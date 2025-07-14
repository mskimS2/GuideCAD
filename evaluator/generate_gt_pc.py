import json
import os
import numpy as np
import h5py
import multiprocessing
import argparse
import sys
import time

sys.path.append("..")
import utils
from cadlib.visualize import vec2CADsolid, CADsolid2pc


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--n_points", type=int, default=2000)
args = parser.parse_args()

SAVE_GT_DIR = args.src
if not os.path.exists(SAVE_GT_DIR):
    os.makedirs(SAVE_GT_DIR)

with open(os.path.join("dataset/data/guidecad.json"), "r") as fp:
    all_data = json.load(fp)["test"]

all_paths = []
for p in all_data:
    cadvec_path = "dataset/data/cad_vec/" + p + ".h5"
    all_paths.append(cadvec_path)


def process_with_timeout(func, args, timeout):
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        print(f"[Timeout] Processing took longer than {timeout} seconds. Skipping...")
        return False
    return True


def process_one(path):
    data_id = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(SAVE_GT_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    try:
        start_time = time.time()
        with h5py.File(path, "r") as fp:
            gt_vec = fp["vec"][:].astype(np.float64)

        gt_shape = vec2CADsolid(gt_vec)
        out_gt_pc = CADsolid2pc(gt_shape, args.n_points, data_id)
        utils.write_ply(out_gt_pc, os.path.join(SAVE_GT_DIR, data_id + ".ply"))

        print(f"[Info] Processed {data_id} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"[Error] Failed to process {data_id}: {e}")


for path in all_paths:
    process_with_timeout(process_one, (path,), timeout=5)
