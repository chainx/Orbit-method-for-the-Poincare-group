import time

global total_time_elapsed
total_time_elapsed = 0
start = time.perf_counter()

def time_conversion(s):
    s = int(s)
    hour, r = divmod(s, 3600)
    min, sec = divmod(r, 60)
    out  = f"{hour:d}h " if hour else ""
    out += f"{min}m " if hour or min else ""
    out += f"{sec}s"
    return out

def compute_time_elapsed(msg="", verbose=True):
    global total_time_elapsed
    total_elapsed_new = time.perf_counter() - start
    elapsed = time_conversion(total_elapsed_new-total_time_elapsed)
    if verbose:
        print(f"{msg}. Time elapsed: {elapsed}")
    total_time_elapsed =  total_elapsed_new