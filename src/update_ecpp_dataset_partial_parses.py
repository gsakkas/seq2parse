import sys
import resource
from os.path import join, exists
from pathlib import Path
# import signal
# from contextlib import contextmanager
import tqdm
import earleyparser_interm_repr

# @contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise TimeoutError("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)


def limit_memory():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft < 0:
        soft = 8 * 1024 * 1024 * 1024
    else:
        soft = soft * 6 // 10
    if hard < 0:
        hard = 32 * 1024 * 1024 * 1024
    else:
        hard = hard * 8 // 10
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def store_dataset(tkns, upd_tkns, erules, tok_chgs, dur, next_tkn, out_file):
    out_file.write(tkns + " <||> " + upd_tkns + " <||> " + " <++> ".join(erules) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + next_tkn + "\n")


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]))


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    for partPath in list(dataDir.glob('part_2018_*')):
        print("#", partPath.name)
        # if partPath.name not in ['part_6', 'part_1']:
        #     continue
        dataset = []
        dataset_part_file = join(partPath, "erule-dataset.txt")
        # if exists(join(partPath, "erule-dataset-partials-probs.txt")):
        #     continue
        if not exists(dataset_part_file):
            continue
        else:
            with open(dataset_part_file, "r") as inFile:
                dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
            with open(join(partPath, "erule-dataset-partials-probs.txt"), "w") as outFile:
                for tokns, eruls, tok_changes, duration in tqdm.tqdm(dataset):
                    upd_tokns, next_token = earleyparser_interm_repr.get_updated_seq(tokns, GRAMMAR)
                    store_dataset(tokns, upd_tokns, eruls, tok_changes, duration, next_token, outFile)
