import os
import glob
import h5py
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
import random
from scipy.spatial import cKDTree as KDTree

import time
import sys

sys.path.append("..")
from cadlib.visualize import vec2CADsolid, CADsolid2pc
import utils

SKIP_DATA = [""]


def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, _ = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, _ = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    return points


def process_one(path):
    try:
        data_id = path.split("/")[-1].split(".")[0][:8]
        with h5py.File(path, "r") as fp:
            out_vec = fp["pred_vec"][:].astype(np.float64)
            gt_vec = fp["gt_vec"][:].astype(np.float64)

        shape = vec2CADsolid(out_vec)
        out_pc = CADsolid2pc(shape, args.n_points, data_id)

        gt_shape = vec2CADsolid(gt_vec)
        gt_pc = CADsolid2pc(gt_shape, args.n_points, data_id)

        if np.max(np.abs(out_pc)) > 2:
            out_pc = normalize_pc(out_pc)
            gt_pc = normalize_pc(gt_pc)

        cd = chamfer_dist(gt_pc, out_pc)
    except Exception as e:
        print(f"[Error] Processing failed for {data_id}: {e}")
        return None

    return cd


def run_parallel(args):
    filepaths = sorted(glob.glob(os.path.join(args.src, "*.h5")))
    if args.num != -1:
        filepaths = filepaths[: args.num]

    print("filepaths", len(filepaths))

    save_path = args.src + f"_pc{args.n_points}_statistics_seed{args.random_seed}.txt"
    record_res = None
    if os.path.exists(save_path):
        response = input(save_path + " already exists, overwrite? (y/n) ")
        if response == "y":
            os.system(f"rm {save_path}")
            record_res = None
        else:
            with open(save_path, "r") as fp:
                record_res = fp.readlines()
                n_processed = len(record_res) - 3

    print(f"Starting parallel processing with {args.n_jobs} jobs...")

    with Pool(processes=args.n_jobs) as pool:
        results = pool.map(process_one, filepaths)

    with open(save_path, "w") as fp:
        for i, (filepath, res) in enumerate(zip(filepaths, results)):
            print(f"{i}\t{filepath.split('/')[-1]}\t{res}", file=fp)

    valid_dists = [x for x in results if x is not None]
    valid_dists = sorted(valid_dists)
    print("Top 20 largest errors:")
    print(valid_dists[-20:][::-1])
    n_valid = len(valid_dists)
    n_invalid = len(results) - n_valid

    avg_dist = np.mean(valid_dists)
    trim_avg_dist = np.mean(valid_dists[int(n_valid * 0.1) : -int(n_valid * 0.1)])
    med_dist = np.median(valid_dists)

    print("#####" * 10)
    print(f"Total: {len(filepaths)}, Invalid: {n_invalid}, Invalid Ratio: {n_invalid / len(filepaths):.2f}")
    print(f"Avg Dist: {avg_dist}, Trim Avg Dist: {trim_avg_dist}, Median Dist: {med_dist}")

    with open(save_path, "a") as fp:
        print("#####" * 10, file=fp)
        print(
            f"Total: {len(filepaths)}, Invalid: {n_invalid}, Invalid Ratio: {n_invalid / len(filepaths):.2f}", file=fp
        )
        print(f"Avg Dist: {avg_dist}, Trim Avg Dist: {trim_avg_dist}, Median Dist: {med_dist}", file=fp)


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None, required=True)

parser.add_argument("--random_seed", type=int, default=1234)
parser.add_argument("--n_points", type=int, default=1000)
parser.add_argument("--num", type=int, default=-1)
parser.add_argument("--n_jobs", type=int, default=cpu_count() // 2, help="Number of parallel jobs")
args = parser.parse_args()

utils.set_randomness(args.random_seed)

print(args.src)
print("SKIP DATA:", SKIP_DATA)
since = time.time()
run_parallel(args)
end = time.time()
print(f"Running time: {end - since:.2f} seconds")
