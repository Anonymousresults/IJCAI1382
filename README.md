# IJCAI1382

Requirements:

pytorch 1.7+ timm 0.4.2 cudatoolkit 10.1

Test TerViT-Swin-S:

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1382 --use_env main.py --model ter_swin_small --batch-size 128 --num_workers 10 --data-path ~/ImageNet --output_dir ./tmp --data-set IMNET --resume ter_swin_s.pth --eval

Checkpoint files can be fetched in 
