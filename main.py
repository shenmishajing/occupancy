import os
import time
import torch
import random
import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
            # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def parse_args():
    parser = argparse.ArgumentParser(description = 'use CUDA_VISIBLE_DEVICES=xxx python main.py to occupy gpus',
                                     formatter_class = SmartFormatter)
    parser.add_argument('-gpus', nargs = '*', type = int, default = None, help = 'gpus to occupied')
    parser.add_argument('-n', type = int, default = 45000, help = 'R|size of matrix for occupy memory, default: 45000\n'
                                                                  'for help, when s is default value, '
                                                                  'the relation between n and memory size as\n'
                                                                  'n\tmemory size(MB)\n'
                                                                  '45000\t11k\n'
                                                                  '40000\t9k3\n'
                                                                  '35000\t7k8.5\n'
                                                                  '30000\t6k6\n'
                                                                  '25000\t5k5\n'
                                                                  '20000\t4k7\n'
                                                                  '15000\t4k\n')
    parser.add_argument('-s', type = int, default = 15000, help = 'size of matrix for occupy cuda')
    parser.add_argument('-t', type = int, default = 5, help = 'time to sleep')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))
    occupied_gpus = [int(os.environ["CUDA_VISIBLE_DEVICES"][g]) for g in args.gpus]
    print(f'occupied gpus: {occupied_gpus}, press ctrl-c to exit')
    occupy_memory = [torch.rand(args.n, args.n, device = torch.device(gpu)) for gpu in args.gpus]
    while True:
        gpu = random.choice(args.gpus)
        torch.rand(args.s, args.s, device = torch.device(gpu)).mm(torch.rand(args.s, args.s, device = torch.device(gpu)))
        time.sleep(random.randint(0, args.t + 1))


if __name__ == "__main__":
    main()
