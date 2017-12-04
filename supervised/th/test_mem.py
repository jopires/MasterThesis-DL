def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    import sys
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

memory_usage_psutil()
