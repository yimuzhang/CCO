import torch
import time

def test_gpu_utilization():
    # 检查可用GPU数量
    if not torch.cuda.is_available():
        print("没有检测到可用的GPU")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU:")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试每个GPU
    for i in range(gpu_count):
        print(f"\n正在测试 GPU {i}...")
        
        # 设置当前GPU
        torch.cuda.set_device(i)
        
        # 创建测试张量
        start_time = time.time()
        x = torch.randn(1000, 1000).cuda()  # 创建大张量
        y = torch.randn(1000, 1000).cuda()
        
        # 执行矩阵乘法（计算密集型操作）
        z = torch.matmul(x, y)
        
        # 同步等待计算完成
        torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        print(f"GPU {i} 测试完成，耗时: {elapsed_time:.2f} 秒")
        
        # 检查内存使用情况
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # 转换为GB
        memory_cached = torch.cuda.memory_reserved(i) / 1024**3     # 转换为GB
        print(f"GPU {i} 内存使用: {memory_allocated:.2f} GB (已分配) / {memory_cached:.2f} GB (保留)")

def test_parallel_gpus():
    """测试并行使用所有GPU"""
    if not torch.cuda.is_available():
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"\n开始并行测试所有 {gpu_count} 个GPU...")
    
    # 在每个GPU上并行执行任务
    results = []
    for i in range(gpu_count):
        device = torch.device(f'cuda:{i}')
        
        # 在每个GPU上创建独立的计算任务
        def gpu_task(device_idx):
            device = torch.device(f'cuda:{device_idx}')
            x = torch.randn(5000, 5000, device=device)
            y = torch.randn(5000, 5000, device=device)
            return torch.matmul(x, y)
        
        results.append(gpu_task(i))
    
    # 等待所有GPU完成
    for i, result in enumerate(results):
        torch.cuda.synchronize(i)
        print(f"GPU {i} 并行任务完成")

if __name__ == "__main__":
    print("开始GPU利用率测试...")
    test_gpu_utilization()
    test_parallel_gpus()
    print("\n测试完成！")