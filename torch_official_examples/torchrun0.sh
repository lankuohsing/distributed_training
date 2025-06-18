CUDA_VISIBLE_DEVICES=4,5 torchrun \
--nproc_per_node=2 \
--nnodes=2 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=10.205.92.13:29603 \
multinode.py 50 10 2>&1 | tee master.log