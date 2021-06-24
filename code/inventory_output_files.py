"""Take inventory of output data files."""

import os
import glob
from collections import defaultdict


results_directory = os.environ['RCAC_SCRATCH']
# results_directory = "./results"


MIN_TRIAL = 1
MAX_TRIAL = 1800
MIN_REP = 1
MAX_REP = 100


# cleanly convert into number ranges
def range_extract(lst):
    """Yield 2-tuple ranges or 1-tuple single elements from list of increasing ints

    From https://rosettacode.org/wiki/Range_extraction#Python

    """
    lenlst = len(lst)
    i = 0
    while i< lenlst:
        low = lst[i]
        while i <lenlst-1 and lst[i]+1 == lst[i+1]: i +=1
        hi = lst[i]
        if   hi - low >= 2:
            yield (low, hi)
        elif hi - low == 1:
            yield (low,)
            yield (hi,)
        else:
            yield (low,)
        i += 1


def printr(ranges):
    return ','.join( (('%i-%i' % r) if len(r) == 2 else '%i' % r) for r in ranges )


def main():
    """every file is of the form '<label>_<trial>_<replication>.npy' except for trial_results, which omit '_<replication>'' """

    replication_raw_data = defaultdict(lambda: [0]*(MAX_REP+MIN_REP))
    replication_results = defaultdict(lambda: [0]*(MAX_REP+MIN_REP))
    trial_results = [0]*(MAX_TRIAL+MIN_TRIAL)
    unknown_files = list()


    # names = [os.path.basename(x) for x in glob.glob('/your_path')]
    for filename in glob.iglob(results_directory + "/*.npy"):
        filename = os.path.basename(filename)[:-4]  # -4 to remove extension
        chunks = filename.split('_')

        # process filename
        if filename.startswith('replication_results'):
            replication_results[int(chunks[2])][int(chunks[3])] = int(chunks[3])
        elif filename.startswith('replication_raw_data'):
            replication_raw_data[int(chunks[3])][int(chunks[4])] = int(chunks[4])
        elif filename.startswith('trial_results'):
            trial_results[int(chunks[2])] = int(chunks[2])
        else:
            unknown_files.append(filename)

    alltrials = set(list(range(MIN_TRIAL, MAX_TRIAL+1)))
    allreps = set(list(range(MIN_REP, MAX_REP+1)))

    # report on raw data for incompletes
    print()
    print("*** replication_raw_data_x_y.npy ***")
    for trial in alltrials:
        trial_reps = replication_raw_data[trial]  # don't need to try...except KeyError due to defaultdict
        missingno = allreps - set(trial_reps)
        if len(missingno) > 0:
            print(f"missing for trial {trial}: {printr(range_extract(sorted(list(missingno))))}")

    # report on replication results
    print()
    print("*** replication_results_x_y.npy ***")
    for trial in alltrials:
        trial_reps = replication_results[trial]
        missingno = allreps - set(trial_reps)
        if len(missingno) > 0:
            print(f"missing for trial {trial}: {printr(range_extract(sorted(list(missingno))))}")

    # report on trial results
    print()
    print("*** trial_results_x.npy ***")
    missingno = alltrials - set(trial_results)
    if len(missingno) > 0:
        print(f"missing for trials: {printr(range_extract(sorted(list(missingno))))}")


    # report unknown files
    print()
    print("*** unknown files in results directory ***")
    for foo in unknown_files:
        print(foo)

    if len(unknown_files) == 0:
        print('none')


if __name__ == '__main__':
    print("warning: this will count empty files; advise deleting those first!")
    main()
