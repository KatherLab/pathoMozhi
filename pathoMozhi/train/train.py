""" Main training script """

import argparse
import glob
import os
import random
import wandb

import torch
torch.cuda.empty_cache()

import numpy as np

from pathoMozhi.train.data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from train_utils import (
    train_one_epoch,
    save_checkpoint,
    create_feature_loader,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import functools

from pathoMozhi import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument("--max_tokens",type=int,default=256,help="Maximum number of tokens to process for each input",)
    parser.add_argument("--tokenizer_path",default="facebook/opt-30b",type=str,help="path to tokenizer",)
    parser.add_argument("--cross_attn_every_n_layers",type=int,default=1,help="how often to add a cross-attention layer after each transformer layer",)

    # training args
    parser.add_argument("--run_name",type=str,default="openflamingo3B",help="used to name saving directory and wandb run",)
    parser.add_argument("--resume_from_checkpoint",type=str,help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",default=None,)
    parser.add_argument("--delete_previous_checkpoint",action="store_true",help="delete previous checkpoint when saving new checkpoint",)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float) # changed from 5e-5 to 1e-4
    parser.add_argument("--lr_scheduler",default="constant",type=str,help="constant, linear, or cosine",)
    parser.add_argument("--gate_learning_rate",type=float,default=None,help=("Absolute learning‑rate for the attn_gate / ff_gate scalars. ""If left unset, we fall back to base_lr * gate_lr_mult."),)
    parser.add_argument("--loss_multiplier", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--precision",choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],default="fp32",help="Floating point precision.",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="whether to train with gradient/activation checkpointing",)
    parser.add_argument("--num_epochs",type=int,default=1,help="we define an 'epoch' as a fixed number of examples, not a pass through the entire dataset",)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--freeze_lm_embeddings",action="store_true",help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    parser.add_argument("--vision_features",type=str,help="path to H5 files with TITAN features",)
    parser.add_argument("--jsonl_file",type=str,help="path to JSONL, which contains file_path and results",)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples", type=int, default=10000)
    parser.add_argument("--new_tokens",nargs="+",default=[],help="List of new tokens to train while freezing existing LM embeddings")

    # distributed training args
    parser.add_argument("--dist-url",default="env://",type=str,help="url used to set up distributed training",)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project",type=str,)
    parser.add_argument("--wandb_entity",type=str,)
    parser.add_argument("--save_checkpoints_to_wandb",default=False,action="store_true",help="save checkpoints to wandb",)

    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    model, tokenizer = create_model_and_transforms(
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )
    # Add new tokens and freeze embeddings as requested
    if args.new_tokens:
        # Add tokens only if not already present
        task_token_ids = tokenizer.convert_tokens_to_ids(args.new_tokens)
        tokens_to_add = []
        for token, token_id in zip(args.new_tokens, task_token_ids):
            if token_id == tokenizer.unk_token_id:  # Token not known to tokenizer
                print(f"Adding new token {token} to tokenizer.")
                tokens_to_add.append(token)
        if tokens_to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            model.resize_token_embeddings(len(tokenizer))
            task_token_ids = tokenizer.convert_tokens_to_ids(tokens_to_add)  # Updated IDs after adding
        # Freeze all embeddings (supporting BioGPT, OPT, GPT2)
        if hasattr(model.lang_encoder, "biogpt") and hasattr(model.lang_encoder.biogpt, "embed_tokens"):
            embedding = model.lang_encoder.biogpt.embed_tokens
        elif hasattr(model.lang_encoder, "decoder") and hasattr(model.lang_encoder.decoder, "embed_tokens"):
            embedding = model.lang_encoder.decoder.embed_tokens  # BioGPT
        elif hasattr(model.lang_encoder, "model") and hasattr(model.lang_encoder.model, "decoder"):
            embedding = model.lang_encoder.model.decoder.embed_tokens  # OPT
        elif hasattr(model.lang_encoder, "embed_tokens"):
            embedding = model.lang_encoder.embed_tokens  # GPT2
        else:
            raise AttributeError("Could not locate the embedding layer in the language model.")
        embedding.weight.requires_grad = False
        # Unfreeze only new task tokens
        for token_id in task_token_ids:
            embedding.weight[token_id].requires_grad = True
    #######################################################
    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    # Load model checkpoint on CPU
    checkpoint_base_path = "/mnt/bulk-mars/vidhya/visionLanguage/CHECKPOINT"
    checkpoint_dir = os.path.join(checkpoint_base_path, args.run_name)

    if os.path.exists(checkpoint_dir) and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
                )[-1]
            print(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1
        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)

    # Initialize DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    # apply weight decay only to params in the xattn layers
    def get_grouped_params(named_params, base_lr, wd, gate_lr=None, gate_lr_mult=5.0):
        """
        Build three parameter groups:
        1. params with weight decay (All multi-dim weights eg xattn params)
        2. params without weight decay (All biases and layernorms)
        3. params with weight decay but with a different learning rate (All gated cross attention params)

        Args:
            named_params (iterable): An iterable of tuples containing parameter names and their corresponding tensors.
            base_lr (float): The base learning rate for the optimizer.
            wd (float): The weight decay to apply to parameters.
            gate_lr (float, optional): Absolute learning rate for gated cross-attention parameters. If None, uses base_lr * gate_lr_mult.
            gate_lr_mult (float, optional): A multiplier for the learning rate of gated cross-attention parameters. Defaults to 5.0.
        """
        decay, no_decay, gate = [], [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if n.endswith("attn_gate") or n.endswith("ff_gate"):
                gate.append(p)
            elif p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        lr_gate = gate_lr if gate_lr is not None else base_lr * gate_lr_mult
        return [
            {"params": decay,"weight_decay": wd,"lr": base_lr},
            {"params": no_decay,"weight_decay": 0.0,"lr": base_lr},
            {"params": gate,"weight_decay": 0.0,"lr": lr_gate},
        ]

    optimizer = torch.optim.AdamW(
        get_grouped_params(params_to_optimize, args.learning_rate, args.weight_decay, gate_lr=args.gate_learning_rate),
        betas=(0.9,0.999)
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        if "optimizer_state_dict" in checkpoint: # Added by me for finetuning only.
            try: # Added by me for finetuning only.
                osd = checkpoint["optimizer_state_dict"] # Pehle yehi line tha right after "if args.resume_from_checkpoint is not None:"
                optimizer.load_state_dict(osd)
            except ValueError:
                print("WARNING: Optimizer state_dict could not be loaded due to mismatch.")
                print("Checkpoint param groups:", len(osd['param_groups']))
                print("Current optimizer param groups:", len(optimizer.param_groups))


    total_training_steps = (
        (args.train_num_samples) // (args.batch_size * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start training!
    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_feature_loader = create_feature_loader(args.vision_features, epoch=epoch)
        train_dataset = get_data(args, train_feature_loader, tokenizer, epoch=epoch)
        train_dataset.set_epoch(epoch)
        train_loader = train_dataset.dataloader

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            device_id=device_id,
            wandb=wandb,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()
