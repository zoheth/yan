import time
import torch
import logging

import torch.distributed as dist
from torch.profiler import ProfilerActivity
import torch.multiprocessing as mp
import os

batch_size_list = [1, 1, 2, 4, 8, 16, 32]
hidden_size = 7168
retry_times = 10

def workitem(offset: int, batch_size: int, send_buffer: torch.Tensor, recv_buffer: torch.Tensor, rank: int, world_size: int):    
    start_time = time.time()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function("model_inference"):
            for i in range(retry_times):
                local_offset = offset
                if rank == 0:
                    ops = []
                    for i in range(1, world_size):
                        recv_tensor = recv_buffer[local_offset:local_offset+batch_size, :]
                        recv_op = dist.P2POp(dist.irecv, recv_tensor, i)
                        ops.append(recv_op)
                        local_offset += batch_size
                    local_offset = offset
                    for i in range(1, world_size):
                        send_tensor = send_buffer[local_offset:local_offset+batch_size, :]
                        send_op = dist.P2POp(dist.isend, send_tensor, i)
                        ops.append(send_op)
                        local_offset += batch_size

                    # 执行批量异步接收
                    reqs = dist.batch_isend_irecv(ops)
                    for req in reqs:
                        req.wait()
                else:
                    ops = []
                    send_tensor = send_buffer[offset:offset+batch_size, :]
                    send_op = dist.P2POp(dist.isend, send_tensor, 0)
                    ops.append(send_op)
                    recv_tensor = recv_buffer[offset:offset+batch_size, :]
                    recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
                    ops.append(recv_op)
                    reqs = dist.batch_isend_irecv(ops)
                    for req in reqs:
                        req.wait()
                if rank == 0:
                    offset += batch_size * (world_size - 1)
                else:
                    offset += batch_size
        end_time = time.time()
    logging.info(f"Rank {rank} 完成，用时 {end_time - start_time} 秒")
    sort_by_keyword = "cuda_time_total"
    if rank == 0:
        print(prof.key_averages().table(header="batch_" + str(batch_size), sort_by=sort_by_keyword, row_limit=10))
    

def run_test(rank, world_size):
    logging.basicConfig(
        level="INFO",
        format=f'{rank}-%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['TORCH_DIST_INIT_BARRIER'] = '1'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    logging.info(f"Rank {rank} / {world_size} 正在运行，设备: {device}")

    # 对于rank0来说，每次他要分别给world_size - 1个rank发送batch*hidden的数据， 并且接受同样的数据
    # 对于其他rank来说，每次他要接受来自rank0的batch*hidden的数据， 并且发送同样的数据
    if rank == 0:
        send_buffer = torch.randn([sum(batch_size_list) * (world_size - 1) * retry_times, hidden_size], device=device)
        recv_buffer = torch.zeros([sum(batch_size_list) * (world_size - 1) * retry_times, hidden_size], device=device)
    else:
        send_buffer = torch.randn([sum(batch_size_list) * retry_times, hidden_size], device=device)
        recv_buffer = torch.zeros([sum(batch_size_list) * retry_times, hidden_size], device=device)

    offset: int = 0
    for batch_size in batch_size_list:
        workitem(offset, batch_size, send_buffer, recv_buffer, rank, world_size)
        if rank == 0:
            offset += batch_size * (world_size - 1) * retry_times
        else:
            offset += batch_size * retry_times

    # 同步所有进程
    dist.barrier()
    print(f"Rank {rank}: 测试完成")

    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4

    # 检查CUDA设备数量
    if torch.cuda.device_count() < world_size:
        print(f"警告: 系统只有 {torch.cuda.device_count()} 个CUDA设备，但需要 {world_size} 个")
        print("将使用可用的设备，可能会有设备共享")

    print(f"启动 {world_size} 个进程进行NCCL测试...")

    # 使用multiprocessing启动多个进程
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

    print("所有进程已完成")