import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--dann_lambdas', type=float, nargs='+', required=True)
parser.add_argument('--output_prefix', type=str, default="dann_logs_")
parser.add_argument('--output_folder', type=str)
parser.add_argument('--conda_env', type=str, default="wilds")
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--cpus', type=int, default=4)
parser.add_argument('--mem', type=str, default="8G")
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--partition', type=str, default="iliad")
args = parser.parse_args()

lmds = args.dann_lambdas

print("Generating sbatch scripts for betas: %s" % str(lmds))

# Run your command
file_names = []
for dl in lmds:
    name = args.output_prefix + "dl_%s" % str(dl).replace('.', '_')
    file_names.append(name)
    with open(os.path.join(args.output_folder, "%s.sh" % name), mode='w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --partition=%s\n" % args.partition)
        fh.writelines("#SBATCH --job-name=%s.job\n" % name)
        fh.writelines("#SBATCH --output=/iliad/u/belkhale/logs/%s.out\n" % name)
        fh.writelines("#SBATCH --error=/iliad/u/belkhale/logs/%s.err\n" % name)
        fh.writelines("#SBATCH --time=14-00:00\n")
        fh.writelines("#SBATCH --nodes=%d\n" % args.nodes)
        fh.writelines("#SBATCH --cpus-per-task=%d\n" % args.cpus)
        fh.writelines("#SBATCH --mem=%s\n" % args.mem)
        fh.writelines("#SBATCH --gres=gpu:%d\n" % args.gpu)
        fh.writelines("echo \"SLURM_JOBID=\"$SLURM_JOBID\n")
        fh.writelines("echo \"SLURM_JOB_NODELIST=\"$SLURM_JOB_NODELIST\n")
        fh.writelines("echo \"SLURM_NNODES=\"$SLURM_NNODES\n")
        fh.writelines("echo \"SLURMTMPDIR=\"$SLURMTMPDIR\n")
        fh.writelines("echo \"working directory=\"$SLURM_SUBMIT_DIR\n")
        fh.writelines("echo \"HOSTNAME=\"$HOSTNAME\n")
        fh.writelines("\n")
        fh.writelines(". ~/.bashrc\n")
        fh.writelines("cd /iliad/u/belkhale/wilds || exit\n")
        fh.writelines("conda activate %s\n" % args.conda_env)
        fh.writelines("python examples/run_expt.py --dataset camelyon17 --algorithm DANN "
                      "--root_dir data --n_groups_per_batch 2 --split_scheme in-dist --dann_lambda %f "
                      "--log_dir ./%s/" % (dl, name))

print("done generating files: %s" % file_names)
