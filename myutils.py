import time
import sys
import sklearn
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import tensorflow as tf
import subprocess
import platform
import datetime

def versions():
    print('OS: ', platform.platform())
    print('python: ', sys.version)
    print('sklearn: ' ,sklearn.__version__)
    print('numpy: ', np.__version__)
    print('pandas: ', pd.__version__)
    print('matplotlib: ', mpl.__version__)
    print("tensorflow: ", tf.__version__)
    print(datetime.datetime.ctime(datetime.datetime.now()))

def GPU():
    print("GPU:\n", subprocess.getstatusoutput("nvidia-smi")[1])

def CPU():
    print("CPU:\n", subprocess.getstatusoutput("cat /proc/cpuinfo")[1])

def memory():
    print("Memory:\n", subprocess.getstatusoutput("free -h")[1])
    
def sysinfo():
    print('hostname: ', subprocess.getstatusoutput("hostname")[1])
    print('IP: ', subprocess.getstatusoutput("hostname -i")[1])
    print('USERNAME: ', subprocess.getstatusoutput("whoami")[1])
    print('PWD: ', os.getenv('PWD'))
    print('='*10)
    CPU()
    print('='*10)
    memory()
    print('='*10)
    print(subprocess.getstatusoutput("which python")[1])
    print('='*10)
    GPU()
    print('='*10)
    
    
def tic():
    global __start_interval 
    __start_interval = time.perf_counter()
def toc():
    global __start_interval
    duration = time.perf_counter() - __start_interval
    print(f"Duration = {duration:.2f}")
def toc1():
    global __start_interval
    duration = time.perf_counter() - __start_interval
    return duration

versions()

