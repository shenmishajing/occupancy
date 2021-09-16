import os
import json
import time
import torch
import pynvml
import psutil
import argparse
import multiprocessing


class Params(object):
    # define cuda memory size
    cuda_matrix_size = 8000
    cpu_matrix_size = 20
    cuda_matrix_memory_size = 850
    need_to_left_memory_size = 1000
    memory_to_matrix_size_file_path = 'relation.json'
    memory_to_matrix_size = json.load(open(memory_to_matrix_size_file_path))
    matrix_size_to_memory = {}
    for memory_size, matrix_size in memory_to_matrix_size:
        matrix_size_to_memory[matrix_size] = memory_size

    # define time
    sleep_time = 0.1


def occupy_gpu(matrix_size, gpu, occupied_cuda = True, cuda_matrix_size = 5000, cpu_matrix_size = 50, max_try = 10, sleep_time = 0.1):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for _ in range(max_try):
        try:
            occupied_tensor = torch.rand(matrix_size, matrix_size, device = torch.device('cuda'))
            break
        except Exception as e:
            print(e)
            if _ == max_try - 1:
                return

    # occupy cuda
    cpu_num = 1  # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    cuda_occupied_tensor = [None for _ in range(3)]
    cpu_occupied_tensor = [None for _ in range(3)]
    torch.set_num_threads(cpu_num)
    while True:
        if occupied_cuda:
            try:
                if cuda_occupied_tensor[0] is None:
                    for i in range(2):
                        cuda_occupied_tensor[i] = torch.rand(cuda_matrix_size, cuda_matrix_size, device = torch.device('cuda'))
                cuda_occupied_tensor[2] = cuda_occupied_tensor[0].mm(cuda_occupied_tensor[1])
                if cpu_occupied_tensor[0] is None:
                    for i in range(2):
                        cpu_occupied_tensor[i] = torch.rand(cpu_matrix_size, cpu_matrix_size, device = torch.device('cpu'))
                cpu_occupied_tensor[2] = cpu_occupied_tensor[0] + cpu_occupied_tensor[1]
            except Exception as e:
                print(e)
                return
        time.sleep(sleep_time)


class GPUInfo(object):
    old_gpus = None
    old_other_gpus = None
    old_help_occupied_gpus = None

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
    def update_cur_gpu(cls, gpu_info, n):
        new_gpus = sorted(cls.calculate_cur_gpus(gpu_info, n))
        new_other_gpus = sorted([gpu.index for gpu in gpu_info.values() if gpu.other_process_occupied])
        new_help_occupied_gpus = sorted([gpu.index for gpu in gpu_info.values() if gpu.other_process_occupied and gpu.cur_process_occupied])
        if cls.old_gpus is not None and set(new_gpus) == set(cls.old_gpus) and set(new_other_gpus) == set(cls.old_other_gpus) and set(
                new_help_occupied_gpus) == set(cls.old_help_occupied_gpus):
            return new_gpus
        if cls.old_gpus is not None and set(cls.old_gpus) - set(new_gpus):
            drop_gpus = sorted(list(set(cls.old_gpus) - set(new_gpus)))
            drop_real_gpus = cls.get_real_gpus(drop_gpus)
            for gpu in drop_gpus:
                gpu_info[gpu].drop()
        else:
            drop_real_gpus = None
        cls.old_gpus = new_gpus
        cls.old_other_gpus = new_other_gpus
        cls.old_help_occupied_gpus = new_help_occupied_gpus
        occupied_gpus = cls.get_real_gpus(new_gpus)
        other_occupied_gpus = cls.get_real_gpus(new_other_gpus)
        help_occupied_gpus = cls.get_real_gpus(new_help_occupied_gpus)
        print(f'occupied gpus: {occupied_gpus}' + (
            f', help occupied gpus: {help_occupied_gpus}' if help_occupied_gpus else '') + (
                  f', other process occupied gpus: {other_occupied_gpus}' if other_occupied_gpus else '') + (
                  f', dropped gpus: {drop_real_gpus}' if drop_real_gpus else ''))
        return new_gpus

    def __init__(self, index = 0, times_to_drop = 60, free_memory = 0, cur_process_occupied_memory = 0,
                 other_process_occupied_memory = None):
        self.index = index
        self.times_to_drop = times_to_drop
        self.free_memory = free_memory
        self.cur_process_occupied_memory = cur_process_occupied_memory
        self.other_process_grown = 0
        self.other_process_occupied_memory = other_process_occupied_memory
        self.occupied_process = None
        self.occupied_matrix_size = 0

    @property
    def other_process_occupied(self):
        return self.other_process_occupied_memory != 0

    @property
    def cur_process_occupied(self):
        return self.cur_process_occupied_memory > 1000

    @property
    def memory_size_can_used(self):
        return self.free_memory + self.cur_process_occupied_memory

    def occupy(self, need_to_left = 0, cuda_occupy = True, max_try = 10):
        matrix_size = self.get_matrix_size(need_to_left, cuda_occupy)
        if matrix_size is not None:
            self.drop()
            self.occupied_process = multiprocessing.Process(target = occupy_gpu, args = (
                matrix_size, self.index, self.other_process_occupied_memory == 0, Params.cuda_matrix_size, Params.cuda_matrix_size, max_try,
                Params.sleep_time))
            self.occupied_process.start()
            self.occupied_matrix_size = matrix_size

    def drop(self):
        if self.occupied_process is None:
            return
        self.occupied_process.terminate()
        self.occupied_process.join()
        self.occupied_process = None

    def get_matrix_size(self, need_to_left = 0, cuda_occupy = True):
        free_memory = self.memory_size_can_used - need_to_left
        if cuda_occupy:
            free_memory -= Params.cuda_matrix_memory_size
        for cur_memory_size, cur_matrix_size in Params.memory_to_matrix_size:
            if cur_matrix_size <= self.occupied_matrix_size:
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
                if p.pid == os.getpid() or self.occupied_process is not None and p.pid == self.occupied_process.pid:
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


def parse_args():
    parser = argparse.ArgumentParser(description = 'use python main.py to occupy gpus')
    parser.add_argument('-gpus', nargs = '+', type = int, default = None, help = 'gpu ids to occupied, default: all gpus')
    parser.add_argument('-n', type = int, default = 4, help = 'number of gpus to occupy')
    parser.add_argument('-t', type = float, default = 0.5, help = 'time to update gpu memory info, in seconds, default: 0.5 seconds')
    parser.add_argument('-T', type = float, default = 1,
                        help = 'time to drop memory when other process requires gpu, in minutes, default: 1 minute')
    args = parser.parse_args()
    return args


def main():
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    pynvml.nvmlInit()

    if args.gpus is None:
        all_gpus = list(range(torch.cuda.device_count()))
    else:
        all_gpus = args.gpus

    if len(all_gpus) == 0:
        return

    print(f'start occupy gpus {GPUInfo.get_real_gpus(all_gpus)}, press ctrl-c to exit')

    gpu_info = {gpu: GPUInfo(gpu, int(args.T * 60 / args.t)) for gpu in all_gpus}
    while True:
        # calculate which gpus to occupy
        for gpu in gpu_info.values():
            gpu.update_gpu_menory_info()
            if gpu.other_process_grown != 0:
                gpu.drop()
            elif gpu.other_process_occupied_memory:
                gpu.occupy()

        # update cur gpu
        cur_gpus = GPUInfo.update_cur_gpu(gpu_info, args.n)

        # occupy memory
        for gpu in cur_gpus:
            gpu_info[gpu].occupy()

        time.sleep(args.t)


if __name__ == "__main__":
    main()
