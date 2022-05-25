import sys
from collections import defaultdict
from copy import deepcopy
import resource
# import timeit
# import multiprocessing.pool
# from multiprocessing import TimeoutError
from os.path import join, exists
from functools import partial
from pathlib import Path
import json
# import signal
# from contextlib import contextmanager
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import tqdm
import earleyparser
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list
from run_extract_token_distribution import return_all_changes
# import run_parse_test_time as rptt

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


def print_results(fails, succs, eruls, avg_erules, bads, out_dir):
    with open(join(out_dir, "ErrorCorrectingEarleyParsesDist.txt"), "w") as dataset_file:
        print("# Dataset size:", succs, "/", fails + succs)
        print("# Parse accuracy within time limit (%):", bads * 100.0 / succs)
        print("# Timed out (%):", fails * 100.0 / (fails + succs))
        print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
        dataset_file.write("Dataset size: " + str(succs) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / succs) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")
    # if True:
    # if fails + succs > 88600:
    with open(join(out_dir, "erules.json"), "w") as out_file:
        # print("#", eruls)
        print("#", avg_erules * 1.0 / succs)
        json.dump(eruls, out_file, indent=4)
        print("# ----------------------------------------")


def store_results(bkts, alls, eruls_bkts, out_dir):
    accuracies = dict()
    for k in alls:
        accuracies[k] = bkts[k] * 100.0 / alls[k]
    avg_erules = dict()
    for k in eruls_bkts:
        avg_erules[k] = eruls_bkts[k] / bkts[k]
    print("#", accuracies)
    # print(alls)
    print("# ----------------------------------------")
    with open(join(out_dir, "accuracies-per-token.json"), "w") as out_file_1, \
        open(join(out_dir, "elems-per-token.json"), "w") as out_file_2, \
        open(join(out_dir, "erules-used-per-token.json"), "w") as out_file_3:
        json.dump(accuracies, out_file_1, indent=4)
        json.dump(alls, out_file_2, indent=4)
        json.dump(avg_erules, out_file_3, indent=4)


def store_dataset(tokns, eruls, tok_chgs, dur, out_file):
    out_file.write(tokns + " <||> " + " <++> ".join(eruls) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + "\n")


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split("<++> ")
    return (samp_1[0], samp_2)


def has_parse(grammar, egrammar, tup):
    i, l = tup
    dct = json.loads(l)
    if dct["index"] in [466, 467, 468] and dct["fixIndex"] == 470 and dct["duration"] in [362, 375, 380]:
        return None
    elif dct["errMsg"] == "SyntaxError":
        if not earleyparser.prog_has_parse(dct['bad'], grammar):
            upd_grammar = deepcopy(egrammar)
            terminals = egrammar.get_alphabet()
            token_changes = return_all_changes(l, terminals)
            if not token_changes:
                return None
            num_of_changes = sum([len(ch[2].split()) if ch[0] == 'added' else len(ch[1].split()) for ch in token_changes])
            if num_of_changes > 40:
                return None
            error_rules = upd_grammar.update_error_grammar(token_changes)
            tokens = get_token_list(dct['bad'], terminals)
            if error_rules == []:
                bparse = False
            else:
                bparse = True
            return (num_of_changes, bparse, error_rules, tokens, dct["duration"])
        return (-1, False, [], "", -1)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    GRAMMAR = earleyparser.read_grammar(grammar_file)
    with ProcessPool(max_workers=14, max_tasks=5) as pool:
        for partPath in list(dataDir.glob('part_*')):
            print("#", partPath.name)
            # if partPath.name not in ['part_15', 'part_8']:
            #     continue
            goodPath = partPath / "goodPairs.jsonl"
            failPath = partPath / "failPairs.jsonl"
            dataset = []
            new_dataset = []
            dataset_part_file = join(partPath, "erule-dataset.txt")
            if not exists(dataset_part_file):
                continue
            else:
                programs = enumerate(goodPath.read_text().strip().split('\n'))
                new_has_parse = partial(has_parse, GRAMMAR, ERROR_GRAMMAR)
                future = pool.map(new_has_parse, programs, chunksize=10)
                it = future.result()
                while True:
                    try:
                        bruh = next(it)
                        if bruh:
                            tok_chs, parse_bad, erules, lexed_prog, duration = bruh
                            if tok_chs > 0:
                                if parse_bad:
                                    new_dataset.append((lexed_prog, list(map(str, erules)), tok_chs, duration))
                    except StopIteration:
                        break
                    except (TimeoutError, ProcessExpired):
                        print("Timeout")
                    except Exception as e:
                        print("WHY here?!", str(e))
                programs = enumerate(failPath.read_text().strip().split('\n'))
                future = pool.map(new_has_parse, programs, chunksize=10)
                it = future.result()
                while True:
                    try:
                        bruh = next(it)
                        if bruh:
                            tok_chs, parse_bad, erules, lexed_prog, duration = bruh
                            if tok_chs > 0:
                                if parse_bad:
                                    new_dataset.append((lexed_prog, list(map(str, erules)), tok_chs, duration))
                    except StopIteration:
                        break
                    except (TimeoutError, ProcessExpired):
                        print("Timeout")
                    except Exception as e:
                        print("WHY here?!", str(e))
                with open(dataset_part_file, "r") as inFile:
                    dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
                with open(join(partPath, "erule-dataset-updated.txt"), "w") as outFile:
                    for tokns, eruls in tqdm.tqdm(dataset):
                        for idx, (lexed_prog, all_erules, tok_chs, duration) in enumerate(new_dataset):
                            if tokns == lexed_prog and all(map(lambda r: r in all_erules, eruls)):
                                store_dataset(tokns, eruls, tok_chs, duration, outFile)
                                del new_dataset[idx]
                                break
