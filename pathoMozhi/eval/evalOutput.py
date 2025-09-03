import os
import torch
from pathoMozhi import create_model_and_transforms
from torch.cuda.amp import autocast
from helper import load_feats_to_tensor, load_pt_feats_to_tensor

cache_dir="./huggingface"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, tokenizer = create_model_and_transforms(
    lang_encoder_path="microsoft/BioGPT-Large",
    tokenizer_path="microsoft/BioGPT-Large",
    cross_attn_every_n_layers=1,
    freeze_lm_embeddings=False,
    use_local_files=False,
    cache_dir=cache_dir,
)
tokenizer.padding_side = "left"
model.to(device)
checkpoint_path = "./checkpoints/checkpoint_125.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

if "model_state_dict" in checkpoint:
    print('Yup we do have a "model_state_dict" key in the checkpoint')
    checkpoint = checkpoint["model_state_dict"]
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
#model.load_state_dict(checkpoint, strict=False)
state = model.load_state_dict(checkpoint, strict=False)
model.eval()

input_folder = "./regtest2/conchv1_5/STAMP_raw_conchv1_5" # CONCHv1_5 Features
ext = ".h5" 

files = sorted([
    os.path.join(input_folder, f) for f in os.listdir(input_folder)
    if f.endswith(ext)
])

def inferReport(feats, prompt: str = "<image>"):
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad(), autocast(dtype=torch.float32):
        out = model.generate(
            vision_x=feats.to(device),
            lang_x=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            max_new_tokens=320,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith("Final Diagnosis:"):
        decoded = decoded[len("Final Diagnosis:"):].strip()
    decoded = decoded.replace(" + ", "+")
    return decoded

eval_samples = []
for f in files:
    if ext == ".h5":
        feats = load_feats_to_tensor(f)
    else:
        feats = load_pt_feats_to_tensor(f)
    feats = feats.unsqueeze(1)
    eval_samples.append((f, feats))

import json
from pathlib import Path

results = []
for file_path, feats in eval_samples:
    report = inferReport(feats)
    image_id = Path(file_path).with_suffix(".tiff").name
    results.append({
        "id": image_id,
        "report": report.strip()
    })

with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)
