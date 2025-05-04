Launching Multi-GPU Training for switch_train_eval.py

Using torchrun (Recommended for PyTorch ≥1.9)
```bash
torchrun --nproc_per_node=NUM_GPUS switch_train_eval.py --nproc_per_node: Number of GPUs to use on this machine
```

Example for 4 GPUs:
```bash
torchrun --nproc_per_node=4 switch_train_eval.py
```

Using torch.distributed.run Module (Legacy but supported)
```bash
python -m torch.distributed.run --nproc_per_node=1 switch_train_eval.py
```
Use this if torchrun is unavailable or if you are using legacy launch scripts.

Manually Restrict to Specific GPUs with CUDA_VISIBLE_DEVICES
Run with GPU 0 and GPU 1 only:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 switch_train_eval.py
```

Run with 4 specific GPUs (0 to 3):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 switch_train_eval.py
```

Replace python with the full path to your Python interpreter if needed:
```bash
/E/Personal_AI_Model_Training/M11309813/anaconda3/envs/MoE/bin/python -m torch.distributed.run ...
```
