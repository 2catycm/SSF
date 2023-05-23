CUDA_VISIBLE_DEVICES=$1  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=$2  \
	train.py /data/VTAB/vtab-1k/svhn  --dataset svhn --num-classes 10  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  /data/work_dirs/PEFT/vit_base_patch16_224_in21k/vtab/svhn/ssf \
	--amp --tuning-mode ssf --pretrained  