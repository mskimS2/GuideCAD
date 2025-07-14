import h5py
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys

sys.path.append("..")
from cadlib.macro import *
import utils

utils.set_randomness(42)

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None, required=True)
args = parser.parse_args()

TOLERANCE = 3

result_dir = args.src
filenames = sorted(os.listdir(result_dir))

# overall accuracy
avg_cmd_acc = []  # ACC_cmd
avg_param_acc = []  # ACC_param

# accuracy w.r.t. each command type
each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

# accuracy w.r.t each parameter
args_mask = CMD_ARGS_MASK.astype(np.float64)
N_ARGS = args_mask.shape[1]
each_param_cnt = np.zeros([*args_mask.shape])
each_param_acc = np.zeros([*args_mask.shape])


# F1-score
each_cmd_tp = np.zeros((len(ALL_COMMANDS),))
each_cmd_fp = np.zeros((len(ALL_COMMANDS),))
each_cmd_fn = np.zeros((len(ALL_COMMANDS),))

for name in tqdm(filenames):
    path = os.path.join(result_dir, name)
    with h5py.File(path, "r") as fp:
        out_vec = fp["pred_vec"][:].astype(np.int32)
        gt_vec = fp["gt_vec"][:].astype(np.int32)

    out_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = out_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    cmd_acc = (out_cmd == gt_cmd).astype(np.int32)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]
        each_cmd_cnt[cmd] += 1
        each_cmd_acc[cmd] += cmd_acc[j]

        # F1-score True Positive, False Positive, False Negative
        if out_cmd[j] == cmd:  # Correct prediction
            each_cmd_tp[cmd] += 1
        else:  # Incorrect prediction
            each_cmd_fp[out_cmd[j]] += 1  # Predicted as this command but incorrect
            each_cmd_fn[cmd] += 1  # Missed this ground truth command

        if cmd in [SOL_IDX, EOS_IDX]:
            continue

        if out_cmd[j] == gt_cmd[j]:  # Only account param acc for correct cmd
            tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(np.int32)
            # Filter param that do not need tolerance (strict equality)
            if cmd == EXT_IDX:
                tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(np.int32)[-2:]
            elif cmd == ARC_IDX:
                tole_acc[3] = (out_param[j] == gt_param[j]).astype(np.int32)[3]

            valid_param_acc = tole_acc[args_mask[cmd].astype(np.bool)].tolist()
            param_acc.extend(valid_param_acc)

            each_param_cnt[cmd, np.arange(N_ARGS)] += 1
            each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

    param_acc = np.mean(param_acc) if len(param_acc) > 0 else 0
    avg_param_acc.append(param_acc)
    cmd_acc = np.mean(cmd_acc)
    avg_cmd_acc.append(cmd_acc)

# F1-score
precision = each_cmd_tp / (each_cmd_tp + each_cmd_fp + 1e-6)
recall = each_cmd_tp / (each_cmd_tp + each_cmd_fn + 1e-6)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)


save_path = result_dir + "_f1_stat.txt"
with open(save_path, "w") as fp:
    # Overall accuracy (averaged over all data)
    avg_cmd_acc = np.mean(avg_cmd_acc)
    print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
    avg_param_acc = np.mean(avg_param_acc)
    print("avg param acc (ACC_param):", avg_param_acc, file=fp)

    # F1-score for each command type
    print("\nF1-score for each command type:", file=fp)
    for cmd, f1 in zip(ALL_COMMANDS, f1_score):
        print(f"{cmd}: {f1:.4f}", file=fp)

    # Precision and Recall for each command type
    print("\nPrecision and Recall for each command type:", file=fp)
    for cmd, p, r in zip(ALL_COMMANDS, precision, recall):
        print(f"{cmd} - Precision: {p:.4f}, Recall: {r:.4f}", file=fp)

    # Accuracy of each command type
    each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
    print("\nAccuracy for each command type:", file=fp)
    for cmd, acc in zip(ALL_COMMANDS, each_cmd_acc):
        print(f"{cmd}: {acc:.4f}", file=fp)

    # Accuracy of each parameter type
    print("\nParameter accuracy for each command type:", file=fp)
    each_param_acc = each_param_acc * args_mask
    each_param_cnt = each_param_cnt * args_mask
    each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
    for i in range(each_param_acc.shape[0]):
        params = each_param_acc[i][args_mask[i].astype(np.bool)]
        print(f"{ALL_COMMANDS[i]} param acc: {params}", file=fp)

with open(save_path, "r") as fp:
    res = fp.readlines()
    for l in res:
        print(l, end="")
