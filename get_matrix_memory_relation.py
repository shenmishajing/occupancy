import math
import os
import json
import torch
import pynvml
import psutil


def main():
    pynvml.nvmlInit()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    else:
        gpu = 0
    meminfo = []
    matrix_size = 0
    while True:
        try:
            matrix_size += 1000
            occupied_tensor = torch.rand(matrix_size, matrix_size, device = torch.device(0))
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            running_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in running_processes:
                try:
                    p = psutil.Process(process.pid)
                except Exception as e:
                    continue
                if p.pid == os.getpid():
                    occupied_memory = process.usedGpuMemory / 1024 ** 2
                    occupied_memory_aligned = math.ceil(occupied_memory / 50) * 50
                    cur_info = occupied_memory_aligned, matrix_size
                    meminfo.append(cur_info)
                    print(cur_info)
                    break
            del occupied_tensor
            occupied_tensor = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            break
    output_path = 'relation.json'
    if os.path.isfile(output_path):
        old_meminfo = json.load(open(output_path))
        old_meminfo_dict = {}
        for occupied_memory_aligned, matrix_size in old_meminfo:
            old_meminfo_dict[matrix_size] = occupied_memory_aligned
        match = True
        for occupied_memory_aligned, matrix_size in meminfo:
            if old_meminfo_dict[matrix_size] != occupied_memory_aligned:
                match = False
                print(f'WARNING: Matrix size {matrix_size} not match old meminfo')
        if match:
            print('all matrix sizes match')
    json.dump(meminfo, open(output_path, 'w'))


if __name__ == "__main__":
    main()
