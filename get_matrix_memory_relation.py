import math
import os
import pickle
from collections import OrderedDict

import psutil
import pynvml
import torch


def get_cur_memory_size(gpu):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    running_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    for process in running_processes:
        try:
            p = psutil.Process(process.pid)
        except Exception as e:
            continue
        if p.pid == os.getpid():
            occupied_memory = process.usedGpuMemory / 1024**2
            occupied_memory_aligned = math.ceil(occupied_memory / 50) * 50
            return occupied_memory, occupied_memory_aligned


def get_matrix_size_to_memory(gpu, duplicate=1):
    matrix_size_to_memory = OrderedDict()
    matrix_size = 0
    while True:
        try:
            matrix_size += 100
            [
                torch.rand(matrix_size, matrix_size, device=torch.device(0))
                for _ in range(duplicate)
            ]
            occupied_memory, occupied_memory_aligned = get_cur_memory_size(gpu)
            matrix_size_to_memory[matrix_size] = occupied_memory_aligned
            print(
                {
                    "occupied_memory_aligned": occupied_memory_aligned,
                    "matrix_size": matrix_size,
                    "duplicate": duplicate,
                }
            )
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            break
    return matrix_size_to_memory


def main():
    pynvml.nvmlInit()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    else:
        gpu = 0
    cur_meminfo = get_matrix_size_to_memory(gpu)
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    device_name = pynvml.nvmlDeviceGetName(handle)
    output_path = "meminfo.pkl"
    if os.path.isfile(output_path):
        meminfo = pickle.load(open(output_path, "rb"))
        if device_name in meminfo:
            match = True
            for k in set(cur_meminfo.keys()) & set(meminfo[device_name].keys()):
                if cur_meminfo[k] != meminfo[device_name][k]:
                    match = False
                    print(f"WARNING: Matrix size {k} not match old meminfo")
            if match:
                print("all matrix sizes match")
            merged_meninfo = OrderedDict()
            for k in sorted(
                list(set(cur_meminfo.keys()) | set(meminfo[device_name].keys()))
            ):
                if k in cur_meminfo:
                    merged_meninfo[k] = cur_meminfo[k]
                else:
                    merged_meninfo[k] = meminfo[device_name][k]
            cur_meminfo = merged_meninfo
        meminfo[device_name] = cur_meminfo
    else:
        meminfo = {device_name: cur_meminfo}
    pickle.dump(meminfo, open(output_path, "wb"))
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
