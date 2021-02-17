import os
import psutil
import multiprocessing as mp


def get_nr_procs(max_processes=None, max_per_proc_usage=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        if max_per_proc_usage is None or type(max_per_proc_usage) not in (int, float):
            max_per_proc_usage = 2.6 * pow(10, 9)
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1
        if isinstance(max_processes, int):
            proc_count = min(proc_count, max_processes)
        return max(proc_count, 1)
