CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=. torchrun \
    --nnodes=1 \                # 节点数量（单机）
    --nproc_per_node=2 \         # 每节点进程数（GPU数）
    --rdzv_id=12345 \            # 唯一作业ID
    --rdzv_backend=c10d \        # 后端
    --rdzv_endpoint=localhost:0 \# 主节点地址（localhost表示单机）
    scripts/gpu/single_node_multi_gpu_train_ddp_torchrun.py 2>&1 | tee ddp_torchrun.log