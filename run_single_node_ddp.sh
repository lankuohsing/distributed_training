CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee ddp_torchrun.log