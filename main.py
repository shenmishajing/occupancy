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
    need_to_left_memory_size = 1000
    memory_to_matrix_size = [(9850, 49000), (9450, 48000), (8400, 45000), (6800, 40000), (5350, 35000), (4100, 30000), (3050, 25000),
                             (2200, 20000)]
    matrix_size_to_memory = {}
    for memory_size, matrix_size in memory_to_matrix_size:
        matrix_size_to_memory[matrix_size] = memory_size

    # define time
    sleep_time = 0.1


class GPUInfo(object):
    @classmethod
    def calculate_cur_gpus(cls, gpu_info, n):
        can_occupied_gpu = [info for info in gpu_info.values() if not info.other_process_occupied_memory]
        number_gpus_to_occupied = n - len(gpu_info) + len(can_occupied_gpu)
        if number_gpus_to_occupied > 0:
            gpus = sorted(can_occupied_gpu, reverse = True, key = lambda x: x.memory_size_can_used)
            gpus = [gpu.index for gpu in gpus[: number_gpus_to_occupied]]
        else:
            gpus = []
        return gpus

    @classmethod
    def get_real_gpus(cls, gpus):
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpus = [int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[gpu]) for gpu in gpus]
        return gpus

    @classmethod
    def update_cur_gpu(cls, gpu_info, new_gpus, old_gpus):
        if old_gpus is not None and set(new_gpus) == set(old_gpus):
            return new_gpus
        if old_gpus is not None and set(old_gpus) - set(new_gpus):
            drop_gpus = sorted(list(set(old_gpus) - set(new_gpus)))
            drop_real_gpus = cls.get_real_gpus(drop_gpus)
            for gpu in drop_gpus:
                gpu_info[gpu].drop_tensor()
        else:
            drop_real_gpus = None
        new_gpus = sorted(new_gpus)
        occupied_gpus = cls.get_real_gpus(new_gpus)
        print(f'occupied gpus: {occupied_gpus}, ' + (f'dropped gpus: {drop_real_gpus} ' if drop_real_gpus else '') + 'press ctrl-c to exit')
        return new_gpus

    def __init__(self, index = 0, times_to_drop = 60, free_memory = 0, cur_process_occupied_memory = 0,
                 other_process_occupied_memory = None, occupied_tensor = None):
        self.index = index
        self.times_to_drop = times_to_drop
        self.free_memory = free_memory
        self.cur_process_occupied_memory = cur_process_occupied_memory
        self.other_process_grown = 0
        self.other_process_occupied_memory = other_process_occupied_memory
        self.occupied_tensor = occupied_tensor
        self.cuda_occupied_tensor = [None for _ in range(3)]

    @property
    def other_process_occupied(self):
        return self.other_process_occupied_memory != 0

    @property
    def memory_size_can_used(self):
        return self.free_memory + self.cur_process_occupied_memory

    def drop_tensor(self):
        del self.occupied_tensor
        del self.cuda_occupied_tensor
        torch.cuda.empty_cache()
        self.occupied_tensor = None
        self.cuda_occupied_tensor = [None for _ in range(3)]

    def get_matrix_size(self, need_to_left = 0, cuda_occupy = True):
        free_memory = self.memory_size_can_used - need_to_left
        if cuda_occupy:
            free_memory -= Params.cuda_matrix_memory_size
        matrix_size = 0 if self.occupied_tensor is None else self.occupied_tensor.shape[0]
        for cur_memory_size, cur_matrix_size in Params.memory_to_matrix_size:
            if cur_matrix_size <= matrix_size:
                break
            if free_memory > cur_memory_size:
                return cur_matrix_size
        return None

    def update_gpu_menory_info(self):
        gpu_index = self.index
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_index = int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[gpu_index])
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
        self.free_memory = free_memory
        self.cur_process_occupied_memory = cur_process_occupied_memory
        if self.other_process_occupied_memory is not None:
            if other_process_occupied_memory > self.other_process_occupied_memory:
                self.other_process_grown = self.times_to_drop
            elif self.other_process_grown > 0:
                self.other_process_grown -= 1
        self.other_process_occupied_memory = other_process_occupied_memory

    def malloc_memory(self, need_to_left = 0, cuda_occupy = True, max_try = 10):
        matrix_size = self.get_matrix_size(need_to_left, cuda_occupy)
        if matrix_size is not None:
            del self.occupied_tensor
            for _ in range(max_try):
                try:
                    self.occupied_tensor = torch.rand(matrix_size, matrix_size, device = torch.device(self.index))
                    break
                except Exception as e:
                    self.update_gpu_menory_info()
                    matrix_size = self.get_matrix_size(Params.need_to_left_memory_size)
                    if matrix_size is None:
                        break


def parse_args():
    parser = argparse.ArgumentParser(description = 'use python main.py to occupy gpus')
    parser.add_argument('-gpus', nargs = '+', type = int, default = None, help = 'gpu ids to occupied, default: all gpus')
    parser.add_argument('-n', type = int, default = 4, help = 'number of gpus to occupy')
    parser.add_argument('-t', type = float, default = 5, help = 'time to update gpu memory info, in seconds, default: 5 seconds')
    parser.add_argument('-T', type = float, default = 5,
                        help = 'time to drop memory when other process requires gpu, in minutes, default: 5 minutes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pynvml.nvmlInit()

    if args.gpus is None:
        all_gpus = list(range(torch.cuda.device_count()))
    else:
        all_gpus = args.gpus

    gpu_info = {gpu: GPUInfo(gpu, int(args.T * 60 / args.t)) for gpu in all_gpus}
    cur_gpus = None
    while True:
        # calculate which gpus to occupy
        for gpu in gpu_info.values():
            gpu.update_gpu_menory_info()
            if gpu.other_process_grown != 0:
                if gpu.occupied_tensor is not None:
                    gpu.drop_tensor()
            elif gpu.other_process_occupied_memory:
                gpu.malloc_memory(Params.need_to_left_memory_size, False)

        new_gpus = GPUInfo.calculate_cur_gpus(gpu_info, args.n)

        # update cur gpu
        cur_gpus = GPUInfo.update_cur_gpu(gpu_info, new_gpus, cur_gpus)

        # occupy memory
        for gpu in cur_gpus:
            gpu_info[gpu].malloc_memory()

        # occupy cuda
        for _ in range(int(args.t / Params.sleep_time)):
            if cur_gpus:
                gpus = random.choices(cur_gpus, k = len(cur_gpus))
                for gpu in gpus:
                    if gpu_info[gpu].cuda_occupied_tensor[0] is None:
                        for i in range(2):
                            gpu_info[gpu].cuda_occupied_tensor[i] = torch.rand(Params.cuda_matrix_size, Params.cuda_matrix_size,
                                                                               device = torch.device(gpu))
                    gpu_info[gpu].cuda_occupied_tensor[2] = gpu_info[gpu].cuda_occupied_tensor[0].mm(gpu_info[gpu].cuda_occupied_tensor[1])
            time.sleep(Params.sleep_time)


if __name__ == "__main__":
    main()
