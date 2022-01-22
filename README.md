# IJCAI1382

Requirements:

pytorch 1.7+ timm 0.4.2 cudatoolkit 10.1


Test TerViT-Deit-T: (66.6% Top-1 Acc.)

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1382 --use_env main.py --model ter_deit_tiny --batch-size 128 --num_workers 10 --data-path ~/ImageNet --output_dir ./tmp --data-set IMNET --resume ter_deit_t.pth --eval

Test TerViT-Deit-B: (76.1% Top-1 Acc.)

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1382 --use_env main.py --model ter_deit_base --batch-size 128 --num_workers 10 --data-path ~/ImageNet --output_dir ./tmp --data-set IMNET --resume ter_deit_b.pth --eval

Test TerViT-Swin-S: (79.5% Top-1 Acc.)

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1382 --use_env main.py --model ter_swin_small --batch-size 128 --num_workers 10 --data-path ~/ImageNet --output_dir ./tmp --data-set IMNET --resume ter_swin_s.pth --eval

Checkpoint files can be fetched in https://drive.google.com/drive/folders/1pd5tJ-coeY5qxtNFKqwRgdrTNM6yBgPY?usp=sharing
