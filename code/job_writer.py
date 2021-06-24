# -*- coding: utf-8 -*-
"""Run job submission files for PBS on cluster.

Runs qsub using the - element instead of a submission filename, so the jobfile is read from standard
input and no submission file is needed.

"""

import subprocess


def submit_jobs(start_num, end_num, queue='mventres'):
    """both are inclusive"""
    for i in range(start_num, end_num + 1):
        submit_job(i, queue)

    print(f"jobs {start_num} through {end_num} submitted to {queue}; check qstat to confirm.")


def submit_job(trial_num, queue):
    submission = f"""#!/bin/sh -l
#PBS -q {queue}
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N trial_{trial_num}
#PBS -o /home/mgaree/joboutput/$PBS_JOBNAME.out
#PBS -e /home/mgaree/joboutput/$PBS_JOBNAME.err

source activate entropy

CODEROOT="/home/mgaree/entropy_experiments/code/"
cd $CODEROOT

python3 run_trial.py {trial_num}
"""

    # command = "qsub -z - "  # -z suppresses output of the assigned job number
    subprocess.run(['qsub', '-z', '-'], input=submission, universal_newlines=True)
    # subprocess.run(['echo', submission], input=submission, universal_newlines=True)


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3:
        print("Usage: python <filename> starting_trial_num ending_trial_num [queue name]")
        exit()

    if len(argv) == 3:
        submit_jobs(int(argv[1]), int(argv[2]))
    elif len(argv) == 4:
        submit_jobs(int(argv[1]), int(argv[2]), str(argv[3]))
