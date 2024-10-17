# export NCCL_P2P_DISABLE=1
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
	--config-file "/opt/data/private/ContourNet/configs/ic/r50_baseline.yaml" \
	--skip-test