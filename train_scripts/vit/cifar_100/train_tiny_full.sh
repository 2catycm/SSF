CUDA_VISIBLE_DEVICES=1,5, /home/ycm/program_files/miniconda3/envs/ssf/bin/python  -m torch.distributed.launch --nproc_per_node=2  --master_port=12346  \
	train.py --data_dir /home/ycm/datasets/peft/cifar-100-python --dataset torch/cifar100 --num-classes 100 --model vit_tiny_patch16_224_in21k \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-5 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/vit_base_patch16_224_in21k/cifar_100/tiny_full \
	--amp --tuning-mode None --yuequ-method None --pretrained --log-wandb