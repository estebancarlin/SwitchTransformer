import os
import shutil
import yaml
import numpy as np
from tqdm import tqdm
from random import randint

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

from transformers import T5Tokenizer
from datasets import load_from_disk, DatasetDict

from switch_transformer import SwitchTransformerConfig, SwitchTransformerLM

# ------------------- DISTRIBUTED SETUP -------------------
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------- T5 SPAN MASKING -------------------
def t5_span_masking(input_ids, tokenizer, noise_density=0.15, mean_span_length=3):
    """Apply span corruption for T5-style denoising objective."""
    input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.size()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    masked_inputs = torch.full_like(input_ids, pad_id)
    targets = torch.full_like(input_ids, -100)

    for i in range(batch_size):
        tokens = input_ids[i].tolist()
        num_mask = int(len(tokens) * noise_density)
        spans, masked = [], 0

        while masked < num_mask:
            span_len = np.random.poisson(mean_span_length)
            start = randint(0, len(tokens) - span_len - 1)
            spans.append((start, start + span_len))
            masked += span_len

        spans = sorted(set(spans))
        input_seq, target_seq, last_idx = [], [], 0

        for idx, (start, end) in enumerate(spans):
            input_seq.extend(tokens[last_idx:start])
            input_seq.append(tokenizer.convert_tokens_to_ids(f"<extra_id_{idx}>"))
            target_seq.append(tokenizer.convert_tokens_to_ids(f"<extra_id_{idx}>"))
            target_seq.extend(tokens[start:end])
            last_idx = end

        input_seq.extend(tokens[last_idx:])
        input_seq = input_seq[:seq_len]
        target_seq.append(eos_id)

        masked_inputs[i, :len(input_seq)] = torch.tensor(input_seq)
        targets[i, :len(target_seq)] = torch.tensor(target_seq)

    return masked_inputs, targets



# ------------------- CONFIGURATION -------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
training_cfg = cfg["training"]
model_name = cfg["model_name"]

config = SwitchTransformerConfig(**model_cfg)

# Training hyperparameters
batch_size = training_cfg["batch_size"]
learning_rate = float(training_cfg["learning_rate"])
max_seq_length = training_cfg["max_seq_length"]
max_steps = training_cfg["max_steps"]
log_interval = training_cfg.get("log_interval", 1000)
eval_interval = training_cfg.get("eval_interval", 5000)
num_workers = training_cfg.get("num_workers", 8)
grad_accum_steps = max(training_cfg.get("grad_accum_steps", 1), 1)
dataset_fraction = training_cfg.get("dataset_fraction", 0.5)
val_fraction = training_cfg.get("val_fraction", 1.0)

# Setup logging and saving
if dist.get_rank() == 0:
    os.makedirs(f"results_{model_name}", exist_ok=True)
    shutil.copy("config.yaml", f"results_{model_name}/config.yaml")
    writer = SummaryWriter(f"results_{model_name}/tensorboard")

save_path = f"results_{model_name}/best_model.pt"

# ------------------- MODEL & TOKENIZER -------------------
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = SwitchTransformerLM(config).to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)



# ------------------- DATASET LOADING -------------------
tokenized_full = load_from_disk("./c4_10p_tokenized_streamed/c4_10p_final").with_format("torch")
train_len = int(len(tokenized_full["train"]) * dataset_fraction)
val_len = int(len(tokenized_full["validation"]) * dataset_fraction * val_fraction)

tokenized = DatasetDict({
    "train": tokenized_full["train"].select(range(train_len)),
    "validation": tokenized_full["validation"].select(range(val_len))
})

train_sampler = DistributedSampler(tokenized["train"], shuffle=True)
valid_sampler = DistributedSampler(tokenized["validation"], shuffle=False)

train_loader = DataLoader(
    tokenized["train"], sampler=train_sampler, batch_size=batch_size,
    pin_memory=True, num_workers=num_workers, drop_last=True,
    prefetch_factor=4, persistent_workers=True
)

valid_loader = DataLoader(
    tokenized["validation"], sampler=valid_sampler, batch_size=batch_size,
    pin_memory=True, num_workers=num_workers, drop_last=True,
    prefetch_factor=4, persistent_workers=True
)

# ------------------- OPTIMIZER & LOSS -------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler("cuda")
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
cumulative_expert_count = torch.zeros(config.num_experts, device=device)
best_val_loss = float("inf")
global_step = 0

if dist.get_rank() == 0:
    print("Starting step-based training...")

optimizer.zero_grad()
model.train()

while global_step < max_steps:
    train_sampler.set_epoch(global_step)

    for step, batch in enumerate(tqdm(train_loader, desc="Batch", disable=dist.get_rank() != 0)):
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        input_ids, labels = t5_span_masking(input_ids, tokenizer)
        input_ids, labels = input_ids.to(device), labels.to(device)

        with autocast("cuda"):
            logits, aux_loss, drop_ratio, expert_load = model(input_ids, attention_mask)
            loss_main = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
            loss = loss_main + aux_loss

        scaler.scale(loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update and normalize expert load stats
        cumulative_expert_count += expert_load
        norm_expert_load = expert_load.float() / expert_load.sum()
        norm_cumulative = cumulative_expert_count.float() / cumulative_expert_count.sum()

        # Logging (only rank 0)
        if dist.get_rank() == 0:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = ((predictions == labels) & mask).sum().item()
                total = mask.sum().item()
                accuracy = correct / total if total > 0 else 0

                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/main/train", loss_main.item(), global_step)
                writer.add_scalar("Loss/aux/train", aux_loss.item(), global_step)
                writer.add_scalar("Accuracy/train", accuracy, global_step)
                writer.add_scalar("Routing/DropRatio/train", drop_ratio, global_step)
                writer.add_histogram("Routing/ExpertLoad/train", norm_expert_load.cpu(), global_step)
                writer.add_histogram("Routing/CumulativeExpertLoad/train", norm_cumulative.cpu(), global_step)

        # Evaluation
        if global_step % eval_interval == 0 and global_step > 0:
            model.eval()
            val_loss = val_main = val_aux = val_acc = torch.zeros(1, device=device)
            expert_load_sum = torch.zeros(config.num_experts, device=device)

            with torch.no_grad():
                for val_batch in tqdm(valid_loader, desc="Validation", disable=dist.get_rank() != 0):
                    val_ids = val_batch["input_ids"].to(device)
                    val_mask = val_batch["attention_mask"].to(device)
                    val_ids, val_labels = t5_span_masking(val_ids, tokenizer)
                    val_ids, val_labels = val_ids.to(device), val_labels.to(device)

                    val_logits, val_aux_loss, drop_ratio, val_expert_load = model(val_ids, val_mask)
                    main_loss = loss_fct(val_logits.view(-1, config.vocab_size), val_labels.view(-1))

                    val_loss += main_loss + val_aux_loss
                    val_main += main_loss
                    val_aux += val_aux_loss

                    preds = torch.argmax(val_logits, dim=-1)
                    mask = val_labels != -100
                    val_acc += ((preds == val_labels) & mask).sum().float() / mask.sum().float()

                    expert_load_sum += val_expert_load

            # Synchronize across GPUs
            for metric in [val_loss, val_main, val_aux, val_acc]:
                dist.all_reduce(metric)
                metric /= dist.get_world_size()

            avg_expert_load = expert_load_sum / max(1, len(valid_loader))
            avg_expert_load /= avg_expert_load.sum()

            if dist.get_rank() == 0:
                writer.add_scalar("Loss/val", val_loss.item(), global_step)
                writer.add_scalar("Loss/main/val", val_main.item(), global_step)
                writer.add_scalar("Loss/aux/val", val_aux.item(), global_step)
                writer.add_scalar("Accuracy/val", val_acc.item(), global_step)
                writer.add_scalar("Routing/DropRatio/val", drop_ratio, global_step)
                writer.add_histogram("Routing/ExpertLoad/val", avg_expert_load.cpu(), global_step)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), save_path)
                    print(f"[Step {global_step}] New best val loss: {best_val_loss:.4f} — model saved.")

        global_step += 1

# Save final model (only rank 0)
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), f"results_{model_name}/final_model.pt")
    writer.close()

print("Training complete.")
dist.destroy_process_group()
