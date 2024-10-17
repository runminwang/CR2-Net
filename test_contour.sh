export NGPUS=0
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
	--config-file "/opt/data/private/ContourNet/configs/tpd/r50_baseline.yaml"

