# ddp_train_node1.sh
export CUDA_VISIBLE_DEVICES=0,1       # 使用本机前两块GPU
export MASTER_ADDR='10.205.92.13'     # 保持与主节点一致
export MASTER_PORT='29500'            # 统一端口
export WORLD_SIZE=4                   # 与主节点一致
export NODE_RANK=1                    # 当前节点序号
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python scripts/gpu/custom_ddp_train.py  2>&1 | tee  ddp_train.log