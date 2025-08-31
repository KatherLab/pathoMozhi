import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter
import wandb
from tqdm import tqdm

from pathoMozhi.src.helpers import PerceiverResampler
from pathoMozhi.train.train_utils import create_feature_loader

class ClassifierOnPerceiver(nn.Module):
    def __init__(self, perceiver, num_classes, pooling="attn"):
        super().__init__()
        self.perceiver = perceiver
        self.pooling = pooling
        self.latent_dim = perceiver.latents.shape[1]
        if pooling == "attn":
            self.attn_weights = nn.Sequential(
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, 1)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        # print(f"Input shape: {x.shape}")
        latents = self.perceiver(x).squeeze(1)  # shape: (B, N, D)
        if self.pooling == "avg":
            pooled = latents.mean(dim=1)
        elif self.pooling == "attn":
            weights = self.attn_weights(latents).softmax(dim=1)  # (B, N, 1)
            pooled = (latents * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        return self.classifier(pooled)

class PathDataset(Dataset):
    def __init__(self, df, feature_loader):
        self.df = df.reset_index(drop=True)
        self.feature_loader = feature_loader
        self.labels = self.df["organ"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["id"]
        label = self.labels[idx]
        features = self.feature_loader(file_path)["feature"]
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return features, label

def compute_class_weights(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return torch.tensor([weights[i] for i in range(len(counts))], dtype=torch.float)

def train(csv_file, feature_loader, project_name="perceiver_pretrain", run_name="default_run"):
    wandb.init(project=project_name, name=run_name)

    df = pd.read_csv(csv_file)
    assert "id" in df.columns and "organ" in df.columns

    label_encoder = LabelEncoder()
    df["organ"] = label_encoder.fit_transform(df["organ"])
    print(f"Number of classes: {len(label_encoder.classes_)}")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["organ"], random_state=42)

    train_ds = PathDataset(train_df, feature_loader)
    val_ds = PathDataset(val_df, feature_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceiver = PerceiverResampler(dim=768).to(device)
    model = ClassifierOnPerceiver(perceiver, num_classes=len(label_encoder.classes_), pooling="attn").to(device)

    class_weights_tensor = compute_class_weights(train_df["organ"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    best_acc = 0.0
    best_val_loss = float('inf')
    patience = 20
    epochs_without_improvement = 0
    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            wandb.log({"batch_train_loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x = x.to(device)
                logits = model(x)
                loss = loss_fn(logits, y.to(device))
                val_loss += loss.item()
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                trues.extend(y.tolist())
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(trues, preds)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": acc
        })
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            checkpoint_path = f"./pathMozhi/pathoMozhi/best_perceiver_classifier_epoch{epoch+1}_valacc{acc:.4f}_trainloss{avg_train_loss:.4f}_valloss{avg_val_loss:.4f}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in val loss.")
                break

    wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--vision_features", type=str, required=True, help="Path template like /path/to/features/epoch_{epoch}")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="perceiver_pretrain")
    parser.add_argument("--run_name", type=str, default="perceiverClassifier")

    args = parser.parse_args()

    train_feature_loader = create_feature_loader(args.vision_features, epoch=args.epoch, augment=False)
    train(args.csv_file, train_feature_loader, project_name=args.project_name, run_name=args.run_name)
