import os
import glob
import numpy as np
import h5py
import multiprocessing
from joblib import Parallel, delayed
import argparse
import sys
import time
sys.path.append("..")
import utils
from cadlib.visualize import vec2CADsolid, CADsolid2pc


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=1000)
args = parser.parse_args()

SAVE_DIR = args.src + '_pc'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_with_timeout(func, args, timeout):
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        print(f"[Timeout] Processing took longer than {timeout} seconds. Skipping...")
        return False
    return True


def process_one(path, SAVE_DIR):
    data_id = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    try:
        start_time = time.time()
        with h5py.File(path, 'r') as fp:
            out_vec = fp["pred_vec"][:].astype(np.float64)
            gt_vec = fp["gt_vec"][:].astype(np.float64)

        # Process prediction
        shape = vec2CADsolid(out_vec)
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
        utils.write_ply(out_pc, os.path.join(SAVE_DIR, data_id + ".ply"))
        
        print(f"[Info] Processed {data_id} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"[Error] Failed to process {data_id}: {e}")


all_paths = glob.glob(os.path.join(args.src, "*.h5"))
print("all_paths", len(all_paths))
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--n_points', type=int, default=1000)
args = parser.parse_args()
    
for path in all_paths:
    SAVE_DIR = path.split("/h5/")[0]+"/h5_pc/"
    process_with_timeout(process_one, (path, SAVE_DIR,), timeout=5)