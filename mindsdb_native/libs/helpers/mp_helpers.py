import os
import psutil
import multiprocessing as mp


def get_nr_procs(max_processes=None, max_per_proc_usage=None, df=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        if max_per_proc_usage is None or type(max_per_proc_usage) not in (int, float):
            try:
                import mindsdb_worker
                import ray
                max_per_proc_usage = 0.2 * pow(10,9)
            except:
                max_per_proc_usage = 3 * pow(10, 9)
            if df is not None:
                max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1
        if isinstance(max_processes, int):
            proc_count = min(proc_count, max_processes)
        return max(proc_count, 1)
