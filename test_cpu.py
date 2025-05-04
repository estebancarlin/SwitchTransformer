import os
import torch

def get_optimal_num_workers():
    num_gpus = torch.cuda.device_count()
    num_cores = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    if num_gpus > 0:
        workers_per_gpu = num_cores // num_gpus
    else:
        workers_per_gpu = num_cores
    return max(2, min(8, workers_per_gpu))

if __name__ == "__main__":
    num_cores = os.cpu_count() or 1
    num_gpus = torch.cuda.device_count()
    num_workers = get_optimal_num_workers()

    print(f"Detected {num_cores} CPU cores, {num_gpus} GPU(s). Using {num_workers} worker(s).")