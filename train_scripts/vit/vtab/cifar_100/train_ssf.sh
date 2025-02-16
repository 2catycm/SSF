CUDA_VISIBLE_DEVICES=2,6 python  -m torch.distributed.launch --nproc_per_node=2  --master_port=19547  \
	train.py ~/datasets/peft/vtab-1k/cifar  --dataset cifar100 --num-classes 100  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vit_base_patch16_224_in21k/vtab/cifar_100/ssf \
	--amp  --tuning-mode ssf --pretrained  \
