import psutil
import multiprocessing as mp


def get_nr_procs():
    available_mem = psutil.virtual_memory().available
    max_per_proc_usage = 2.6 * pow(10,9)
    proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1
    return max(proc_count,1)
