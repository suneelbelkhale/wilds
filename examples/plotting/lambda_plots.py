# for sweeping lambda

import argparse
import os
import numpy as np

import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--data_folders", type=str, nargs='+')
parser.add_argument("--csvs_to_load", type=str, nargs='+')
parser.add_argument("--keys", type=str, nargs='+')
parser.add_argument("--epoch", type=int, default=4)
args = parser.parse_args()


def lambda_from_filename(name):
    assert "_dl_" in name
    start = str(name).index("_dl_")
    str_lam = name[start + 4:].replace('_', '.')
    return float(str_lam)

# list of (num_folders, num_csvs)
panda_frames = []
data = []
lmds = []
for folder in args.data_folders:
    assert os.path.exists(folder), [folder]
    panda_frames.append([])
    data.append([])
    for csv in args.csvs_to_load:
        fname = os.path.join(folder, csv)
        assert os.path.exists(fname), [folder, csv]
        panda_frames[-1].append(pandas.read_csv(fname))
        data[-1].append([])
        for key in args.keys:
            data[-1][-1].append(panda_frames[-1][-1].loc[args.epoch, key])
    lmds.append(lambda_from_filename(folder))

# (num_folders, num_csvs, num_keys)
data = np.asarray(data)
# (num_folders,)
lmds = np.asarray(lmds)

nf, nc, nk = data.shape

import matplotlib
import matplotlib.pyplot as plt

fig, axis = plt.subplots(figsize=(5 * nc, 6), ncols=nc)

total_width = 0.75
each_width = total_width / nk

for c in range(nc):
    idx_order = np.argsort(lmds)
    for ki, k in enumerate(args.keys):
        shift = - total_width / 2 + each_width * (0.5 + ki / nk)
        axis[c].bar(np.arange(nf) + shift, data[:, c, ki][idx_order], width=each_width, label=k, tick_label=[str(dl) for dl in lmds[idx_order]])
    # axis[c].set_xscale('log')
    axis[c].set_title(args.csvs_to_load[c][:-9])  # remove _eval.csv
    axis[c].set_xlabel("dann_lambda")
    # axis[c].set_xticks(lmds[idx_order] + 1)
    # axis[c].get_xaxis().get_major_formatter().labelOnlyBase = False
    axis[c].set_ylabel("metric")
    axis[c].legend()

plt.show()