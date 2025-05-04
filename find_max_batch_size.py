import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from torch.amp import autocast

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset and dataloader configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./c4_10p_tokenized_streamed/c4_10p_final",
                        help="Path to the tokenized dataset")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Maximum sequence length (in tokens)")
    parser.add_argument("--max_batch_size", type=int, default=1_048_576,
                        help="Upper bound for global batch size during search")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers per process")
    return parser.parse_args()

def initialize_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize distributed environment and return rank, local_rank, world_size, and device."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device

def load_dataset(data_path: str, rank: int) -> torch.utils.data.Dataset:
    """Load a tokenized dataset from disk and format it for PyTorch."""
    if rank == 0:
        print("Loading dataset...")
    dataset = load_from_disk(data_path)["train"]
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

def test_batch_size(model: torch.nn.Module, dataset, batch_size: int,
                    sampler, device: torch.device, num_workers: int, world_size: int) -> bool:
    """
    Test if a given global batch size can fit in memory across all GPUs.
    Runs a few iterations and catches OOM errors.
    """
    per_gpu_batch = batch_size // world_size
    loader = DataLoader(dataset, sampler=sampler, batch_size=per_gpu_batch,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    try:
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            with autocast("cuda"):
                _ = model(input_ids)
            if i >= 2:  # Only test a few batches
                break
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return False
        else:
            raise e

def find_max_global_batch_size(model: torch.nn.Module, dataset, sampler,
                               device: torch.device, world_size: int,
                               max_batch_limit: int, num_workers: int, rank: int) -> int:
    """
    Perform a distributed binary search to find the maximum global batch size
    that fits in memory across all GPUs.
    """
    low, high = world_size * 4, max_batch_limit
    best = low
    if rank == 0:
        print(f"Searching max global batch size across {world_size} GPUs...")

    while low <= high:
        mid = (low + high) // 2
        success = test_batch_size(model, dataset, mid, sampler, device, num_workers, world_size)
        success_tensor = torch.tensor(int(success), device=device)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)

        if success_tensor.item() == 1:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best

def main():
    args = parse_args()
    rank, local_rank, world_size, device = initialize_distributed()
    dataset = load_dataset(args.data_path, rank)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Lightweight dummy model for batch size testing
    model = torch.nn.Embedding(32128, 768).to(device)

    # Find the largest batch size that can fit across all GPUs
    max_global_batch = find_max_global_batch_size(model, dataset, sampler,
                                                  device, world_size,
                                                  args.max_batch_size,
                                                  args.num_workers, rank)

    if rank == 0:
        print(f"\nMax global batch size: {max_global_batch} examples "
              f"({max_global_batch * args.seq_len} tokens total, "
              f"{max_global_batch // world_size} per GPU)")

if __name__ == "__main__":
    main()