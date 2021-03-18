import random
import os
import torch

import numpy as np
import zipfile
from tqdm import tqdm


from datetime import datetime
from contextlib import contextmanager
from time import time


def set_seed(my_seed):

    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # This can slow down training


def snapshot_code_to_zip(code_path, snapshot_zip_output_dir, snapshot_zip_output_file_name):

    zf = zipfile.ZipFile(os.path.join(snapshot_zip_output_dir, snapshot_zip_output_file_name), "w")
    dirs_to_exclude = ['.git', 'dataset', 'my_debug', 'log']
    for dirname, subdirs, files in os.walk(code_path):
        for dir_to_exclude in dirs_to_exclude:
            if dir_to_exclude in subdirs:
                subdirs.remove(dir_to_exclude)  # If you remove something from the 'subdirs' (second parameter) of os.walk() , os.walk() does not walk into it , that way that entire directory will be skipped. Details at docs.python.org/3/library/os.html#os.walk
        for filename in files:
            if filename == snapshot_zip_output_file_name:
                continue  # skip the output zip file to avoid infinite recursion
            print(filename)
            zf.write(os.path.join(dirname, filename), os.path.relpath(os.path.join(dirname, filename), os.path.join(code_path, '..')))
    zf.close()

@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    ellapsed_time = time() - start

    print(f"{description}: {ellapsed_time}")
