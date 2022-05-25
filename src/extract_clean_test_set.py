import sys
from os.path import join, exists
import subprocess
from functools import partial
from pathlib import Path
import tqdm


def is_clean(orig_fix):
    with open("repairs/pylint_test.py", "w") as test_pylint:
        test_pylint.write(orig_fix.replace("\\n", '\n')[1:-1])
    pylint_output = subprocess.run(["pylint", "repairs/pylint_test.py", "--errors-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return pylint_output.returncode == 0


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]), samp_1[4], samp_1[5] == "popular", samp_1[6], samp_1[7], samp_1[8])


def do_all_test(data_dir, out_dir, in_file):
    dataset_part_file = join(data_dir, in_file)
    if exists(dataset_part_file):
        with open(dataset_part_file, "r") as inFile:
            dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
    print("# Syntax Errors in original test set:", len(dataset))
    dataset = [(tokns, erules, tok_chgs, user_time, fixed_tokns, popular, orig_prog, orig_fix, actual_tkns) for tokns, erules, tok_chgs, user_time, fixed_tokns, popular, orig_prog, orig_fix, actual_tkns in dataset]
    new_length = 0
    with open(join(out_dir, "clean-test-set-top-20-erules.txt"), "w") as outFile:
        for tokns, erules, tok_chgs, user_time, fixed_tokns, popular, orig_prog, orig_fix, actual_tkns in tqdm.tqdm(dataset):
            if is_clean(orig_fix):
                outFile.write(tokns + " <||> " + " <++> ".join(erules) + " <||> " + str(tok_chgs) + " <||> " + str(user_time) + " <||> " + fixed_tokns + " <||> " + ("popular" if popular else "not_popular") + " <||> " + orig_prog + " <||> " + orig_fix + " <||> " + actual_tkns + "\n")
                new_length += 1
    print("# Syntax Errors in new test set:", new_length)



if __name__ == "__main__":
    dataDir = Path(sys.argv[1])
    outDir = Path(sys.argv[2])
    input_file = sys.argv[3]

    do_all_test(dataDir, outDir, input_file)

