# IJCAI1382

Requirements:
pytorch 1.7+
timm 0.4.2
cudatoolkit 10.1

Test TerVit-Swin-S:
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1382 --use_env main.py --model ter_swin_small --batch-size 128 --warmup-epochs 0 --num_workers 10 --data-path ~/ImageNet --output_dir ./tmp --data-set IMNET --resume ter_swin_s.pth --eval

checkpoints can be fetched in 
