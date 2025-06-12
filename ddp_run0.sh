# ddp_train_node0.sh
export CUDA_VISIBLE_DEVICES=0,1       # 使用本机前两块GPU
export MASTER_ADDR='10.205.92.13'     # 主节点IP
export MASTER_PORT='29501'            # 统一端口
export WORLD_SIZE=4                   # 总进程数=2节点×2GPU
export NODE_RANK=0                    # 当前节点序号
PYTHONPATH=. # 使用torchrun自动管理进程（替代mp.spawn）
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    scripts/gpu/custom_ddp_train.py  2>&1 | tee  ddp_train.log