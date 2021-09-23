import math
import os
import pickle
import torch
import pynvml
import psutil


def get_cur_memory_size(gpu):
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
            return occupied_memory, occupied_memory_aligned


def get_matrix_size_to_memory(duplicate, gpu):
    matrix_size_to_memory = {}
    matrix_size = 0
    while True:
        try:
            matrix_size += 100
            [torch.rand(matrix_size, matrix_size, device = torch.device(0)) for _ in range(duplicate)]
            occupied_memory, occupied_memory_aligned = get_cur_memory_size(gpu)
            matrix_size_to_memory[matrix_size] = occupied_memory_aligned
            print(occupied_memory_aligned, matrix_size)
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            break
    return matrix_size_to_memory


def main():
    pynvml.nvmlInit()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    else:
        gpu = 0
    meminfo = {
        'matrix_size_to_memory': get_matrix_size_to_memory(1, gpu),
        'cuda_matrix_size_to_memory': get_matrix_size_to_memory(3, gpu)
    }
    output_path = 'meminfo.pkl'
    if os.path.isfile(output_path):
        old_meminfo = pickle.load(open(output_path, 'rb'))
        match = True
        for name in meminfo:
            for k in set(meminfo[name].keys()) & set(old_meminfo[name].keys()):
                if meminfo[name][k] != old_meminfo[name][k]:
                    match = False
                    print(f'WARNING: {name} Matrix size {k} not match old meminfo')
        if match:
            print('all matrix sizes match')
    pickle.dump(meminfo, open(output_path, 'wb'))


if __name__ == "__main__":
    main()
