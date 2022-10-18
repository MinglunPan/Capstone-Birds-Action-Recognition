import os
from config import DATA_DIR
import numpy as np
import imageio
import IPython


def getImagePath(row_record, data_path = DATA_DIR):
    file_path = [str(x) for x in row_record[['day_dir','camera_dir','video_dir','track_dir','image_file']].values]
    file_path.insert(0, data_path)
    return os.path.join(*file_path)

def get_run_logdir(root_logdir = os.path.join(os.curdir, "logs"), log_name = None, overwrite = True):
    import time
    run_id = log_name or time.strftime("run_%Y_%m_%d-%H_%M_%S")
    log_path = os.path.join(root_logdir, run_id)
    if os.path.exists(log_path) and overwrite:
        shutil.rmtree(log_path)
    return log_path