import os
import time
import torch
import pynvml
import psutil
import random
import argparse


class Params(object):
    # define cuda memory size
    cuda_matrix_size = 5000
    cuda_matrix_memory_size = 500
    need_to_left_memory_size = 2000
    memory_to_matrix_size = [(9850, 49000), (9450, 48000), (8400, 45000), (6800, 40000), (5350, 35000), (4100, 30000), (3050, 25000),
                             (2200, 20000)]
    matrix_size_to_memory = {}
    for memory_size, matrix_size in memory_to_matrix_size:
        matrix_size_to_memory[matrix_size] = memory_size

    # define time
    sleep_time = 0.1
    cuda_times = 5  # find gpu to occupy every 5 minutes
    cuda_times = int(cuda_times * 60 / sleep_time)


class GPUInfo(object):
    def __init__(self, free_memory = 0, cur_process_occupied_memory = 0, other_process_occupied_memory = 0, occupied_tensor = 0):
        self.free_memory = 0
        self.cur_process_occupied_memory = cur_process_occupied_memory
        self.other_process_occupied_memory = other_process_occupied_memory
        self.occupied_tensor = occupied_tensor


def parse_args():
    parser = argparse.ArgumentParser(description = 'use python main.py to occupy gpus')
    parser.add_argument('-gpus', type = int, default = None, help = 'gpu ids to occupied, default: all gpus')
    parser.add_argument('-n', type = int, default = 4, help = 'number of gpus to occupy')
    args = parser.parse_args()
    return args


def get_matrix_size(free_memory, already_used = 0, matrix_size = 0, need_to_left = 0):
    free_memory = get_memory_size_can_used(free_memory, already_used, matrix_size) - Params.cuda_matrix_memory_size - need_to_left
    for cur_memory_size, cur_matrix_size in Params.memory_to_matrix_size:
        if cur_matrix_size <= matrix_size:
            break
        if free_memory > cur_memory_size:
            return cur_matrix_size
    return None


def get_memory_size_can_used(free_memory, already_used = 0, matrix_size = 0):
    if already_used == 0 and matrix_size != 0:
        already_used = Params.matrix_size_to_memory[matrix_size]
    return free_memory + already_used


def main():
    args = parse_args()
    pynvml.nvmlInit()

    if args.gpus is None:
        all_gpus = list(range(torch.cuda.device_count()))
    else:
        all_gpus = args.gpus

    info = {
        'id': 0,
        'free_memory': 0,
        'cur_process_occupied_memory': 0,
        'other_process_occupied_memory': 0,
        'occupied_tensor': None,
    }
    gpu_info = {gpu: info.copy() for gpu in all_gpus}
    for gpu in gpu_info:
        gpu_info[gpu]['id'] = gpu
    cur_gpus = []
    while True:
        # calculate which gpus to occupy
        for gpu in all_gpus:
            gpu_index = gpu
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[gpu])
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = meminfo.free / 1024 ** 2
            cur_process_occupied_memory = other_process_occupied_memory = 0
            running_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in running_processes:
                try:
                    p = psutil.Process(process.pid)
                except Exception as e:
                    continue
                if p.uids().real == os.getuid():
                    if p.pid == os.getpid():
                        cur_process_occupied_memory += process.usedGpuMemory / 1024 ** 2
                    else:
                        other_process_occupied_memory += process.usedGpuMemory / 1024 ** 2
            gpu_info[gpu]['free_memory'] = free_memory
            gpu_info[gpu]['cur_process_occupied_memory'] = cur_process_occupied_memory
            gpu_info[gpu]['other_process_occupied_memory'] = other_process_occupied_memory
            if other_process_occupied_memory != 0:
                matrix_size = get_matrix_size(free_memory, already_used = cur_process_occupied_memory,
                                              need_to_left = Params.need_to_left_memory_size,
                                              matrix_size = 0 if gpu_info[gpu]['occupied_tensor'] is None else
                                              gpu_info[gpu]['occupied_tensor'].shape[0])
                if matrix_size is not None:
                    del gpu_info[gpu]['occupied_tensor']
                    gpu_info[gpu]['occupied_tensor'] = torch.rand(matrix_size, matrix_size, device = torch.device(gpu))

        other_process_occupied_gpus = [info['id'] for info in gpu_info.values() if info['other_process_occupied_memory'] != 0]
        can_occupied_gpu = [info for info in gpu_info.values() if info['other_process_occupied_memory'] == 0]
        number_gpus_to_occupied = args.n - len(other_process_occupied_gpus)
        if number_gpus_to_occupied > 0:
            new_gpus = sorted(can_occupied_gpu, reverse = True,
                              key = lambda x: get_memory_size_can_used(x['free_memory'], x['cur_process_occupied_memory']))
            new_gpus = [gpu['id'] for gpu in new_gpus[: number_gpus_to_occupied]]
        else:
            new_gpus = []

        # occupy memory
        if new_gpus != cur_gpus:
            cur_gpus = new_gpus
            occupied_gpus = cur_gpus
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                occupied_gpus = [int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[gpu]) for gpu in occupied_gpus]
            print(f'occupied gpus: {occupied_gpus}, press ctrl-c to exit')
        for gpu in cur_gpus:
            occupied_matrix_size = get_matrix_size(gpu_info[gpu]['free_memory'], gpu_info[gpu]['cur_process_occupied_memory'],
                                                   0 if gpu_info[gpu]['occupied_tensor'] is None else
                                                   gpu_info[gpu]['occupied_tensor'].shape[0])
            if occupied_matrix_size is not None:
                del gpu_info[gpu]['occupied_tensor']
                gpu_info[gpu]['occupied_tensor'] = torch.rand(occupied_matrix_size, occupied_matrix_size, device = torch.device(gpu))

        # occupy cuda
        for _ in range(Params.cuda_times):
            gpus = random.choices(cur_gpus, k = len(cur_gpus))
            for gpu in gpus:
                torch.rand(Params.cuda_matrix_size, Params.cuda_matrix_size, device = torch.device(gpu)).mm(
                    torch.rand(Params.cuda_matrix_size, Params.cuda_matrix_size, device = torch.device(gpu)))
            time.sleep(Params.sleep_time)


if __name__ == "__main__":
    main()
