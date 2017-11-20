import numpy as np
import pandas as pd

import multiprocessing as mp
import time

def tot_exec_time_str(time_start):
    """(tot_exec_time_str):
        This function gives out the formatted time string.
    """
    time_end = time.time()
    exec_time = time_end-time_start
    tmp_str = "execution time: %0.2fs (%dh %dm %0.2fs)" %(exec_time, exec_time/3600, (exec_time%3600)/60,(exec_time%3600)%60)
    return tmp_str

def foo_pool(x):
    time.sleep(2)
    return x

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback():
    print 'start search on %i CPUs ...' % mp.cpu_count()
    pool = mp.Pool(2)
    X = np.random.rand(3)
    print 'X: ', X
    for i in X:
        pool.apply_async(foo_pool, args = (i, ), callback = log_result)
    pool.close()
    pool.join()
    print(result_list)

if __name__ == '__main__':
    start_time = time.time()
    apply_async_with_callback()
    print tot_exec_time_str(start_time)