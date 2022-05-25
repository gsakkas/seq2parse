import sys
from collections import defaultdict
from copy import deepcopy
import resource
from os import mkdir
from os.path import join, isdir
# import timeit
from multiprocessing import TimeoutError
import random as rnd
from functools import partial
from pathlib import Path
import json
import signal
from contextlib import contextmanager
# from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
# import tqdm
import earleyparser
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list
import earleyparser_interm_repr
from run_extract_token_distribution import return_all_changes
# import run_parse_test_time as rptt

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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


def print_results(fails, succs, bads, out_dir):
    with open(join(out_dir, "DatasetAugmentationStats.txt"), "w") as dataset_file:
        print("# Dataset size:", succs, "/", fails + succs)
        print("# Parse accuracy within time limit (%):", bads * 100.0 / succs)
        print("# Timed out (%):", fails * 100.0 / (fails + succs))
        print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
        dataset_file.write("Dataset size: " + str(succs) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / succs) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")

def store_dataset(tkns, upd_tkns, erules, tok_chgs, dur, next_tkn, out_file):
    out_file.write(tkns + " <||> " + upd_tkns + " <||> " + " <++> ".join(erules) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + next_tkn +"\n")

def get_parts(in_rule):
    parts = in_rule.split(' -> ')
    return parts[0], parts[1].split()

def generate(grammar, egrammar, rul_dist, tup):
    i, l = tup
    dct = json.loads(l)
    terminals = egrammar.get_alphabet()
    bparse = False
    num_of_changes = -1
    error_rules = []
    while not bparse:
        prog_to_break = get_token_list(dct['fix'], terminals)
        new_prog_to_break = prog_to_break
        # print(prog_to_break)
        all_erules_used = []
        num_of_changes = rnd.choice([1, 1, 1, 1, 1, 1, 2, 2, 2, 3])
        for _ in range(num_of_changes):
            while True:
                erule1 = rnd.choice(rul_dist)
                lhs, rhs = get_parts(erule1)
                if lhs == 'InsertErr':
                    erule2 = rnd.choice(rul_dist)
                    lhs2, rhs2 = get_parts(erule2)
                    while 'H' not in rhs2:
                        erule2 = rnd.choice(rul_dist)
                        lhs2, rhs2 = get_parts(erule2)
                    erules_used = [erule1, erule2]
                    # print("1.", lhs, rhs)
                    # print("2.", lhs2, rhs2)
                    lhs2 = rnd.choice(grammar[lhs2.replace('Err_', '')]).rhs[0]
                    # print("Insert:", rhs[0], ", before:", lhs2)
                    if lhs2 in prog_to_break:
                        while True:
                            idx = rnd.randint(0, len(prog_to_break) - 1)
                            new_prog_to_break = prog_to_break
                            if lhs2 in new_prog_to_break[idx:]:
                                new_prog_to_break = new_prog_to_break[:idx] + new_prog_to_break[idx:].replace(lhs2, rhs[0] + " " + lhs2, 1)
                                break
                        if prog_to_break != new_prog_to_break:
                            all_erules_used.extend(erules_used)
                            # print('OK')
                            prog_to_break = new_prog_to_break
                            break
                elif lhs == 'Err_Tag':
                    erule2 = rnd.choice(rul_dist)
                    lhs2, rhs2 = get_parts(erule2)
                    while 'Err_Tag' not in rhs2:
                        erule2 = rnd.choice(rul_dist)
                        lhs2, rhs2 = get_parts(erule2)
                    erules_used = [erule1, erule2]
                    # print("1.", lhs, rhs)
                    # print("2.", lhs2, rhs2)
                    lhs2 = rnd.choice(grammar[lhs2.replace('Err_', '')]).rhs[0]
                    # print("Replace:", lhs2, ", with:", rhs[0])
                    if lhs2 in prog_to_break:
                        while True:
                            idx = rnd.randint(0, len(prog_to_break) - 1)
                            new_prog_to_break = prog_to_break
                            if lhs2 in new_prog_to_break[idx:]:
                                new_prog_to_break = new_prog_to_break[:idx] + new_prog_to_break[idx:].replace(lhs2, rhs[0], 1)
                                break
                        if prog_to_break != new_prog_to_break:
                            all_erules_used.extend(erules_used)
                            # print('OK')
                            prog_to_break = new_prog_to_break
                            break
                elif lhs.startswith('Err_') and rhs == []:
                    lhs = rnd.choice(grammar[lhs.replace('Err_', '')]).rhs[0]
                    erules_used = [erule1]
                    # print("1.", lhs, rhs)
                    # print("Delete:", lhs)
                    if lhs in prog_to_break:
                        while True:
                            idx = rnd.randint(0, len(prog_to_break) - 1)
                            new_prog_to_break = prog_to_break
                            if lhs in new_prog_to_break[idx:]:
                                new_prog_to_break = new_prog_to_break[:idx] + new_prog_to_break[idx:].replace(lhs, "", 1)
                                break
                        if prog_to_break != new_prog_to_break:
                            all_erules_used.extend(erules_used)
                            # print('OK')
                            prog_to_break = new_prog_to_break
                            break
                elif lhs.startswith('Err_') and 'Err_Tag' in rhs:
                    erule2 = rnd.choice(rul_dist)
                    lhs2, rhs2 = get_parts(erule2)
                    while 'Err_Tag' != lhs2:
                        erule2 = rnd.choice(rul_dist)
                        lhs2, rhs2 = get_parts(erule2)
                    erules_used = [erule1, erule2]
                    # print("1.", lhs, rhs)
                    # print("2.", lhs2, rhs2)
                    lhs = rnd.choice(grammar[lhs.replace('Err_', '')]).rhs[0]
                    # print("Replace:", lhs, ", with:", rhs2[0])
                    if lhs in prog_to_break:
                        while True:
                            idx = rnd.randint(0, len(prog_to_break) - 1)
                            new_prog_to_break = prog_to_break
                            if lhs in new_prog_to_break[idx:]:
                                new_prog_to_break = new_prog_to_break[:idx] + new_prog_to_break[idx:].replace(lhs, rhs2[0], 1)
                                break
                        if prog_to_break != new_prog_to_break:
                            all_erules_used.extend(erules_used)
                            # print('OK')
                            prog_to_break = new_prog_to_break
                            break
                else:
                    erule2 = rnd.choice(rul_dist)
                    lhs2, rhs2 = get_parts(erule2)
                    while 'InsertErr' != lhs2:
                        erule2 = rnd.choice(rul_dist)
                        lhs2, rhs2 = get_parts(erule2)
                    erules_used = [erule1, erule2]
                    # print("1.", lhs, rhs)
                    # print("2.", lhs2, rhs2)
                    lhs = rnd.choice(grammar[lhs.replace('Err_', '')]).rhs[0]
                    # print("Insert:", rhs2[0], ", before:", lhs)
                    if lhs in prog_to_break:
                        while True:
                            idx = rnd.randint(0, len(prog_to_break) - 1)
                            new_prog_to_break = prog_to_break
                            if lhs in new_prog_to_break[idx:]:
                                new_prog_to_break = new_prog_to_break[:idx] + new_prog_to_break[idx:].replace(lhs, rhs2[0] + " " + lhs, 1)
                                break
                        if prog_to_break != new_prog_to_break:
                            all_erules_used.extend(erules_used)
                            # print('OK')
                            prog_to_break = new_prog_to_break
                            break
        upd_grammar = deepcopy(egrammar)
        upd_grammar.update_error_grammar_with_erules(all_erules_used)
        error_rules = list(set(map(str, prog_error_rules(prog_to_break, upd_grammar, terminals))))
        if error_rules == []:
            bparse = False
        else:
            bparse = True
        if bparse and len(prog_to_break.split()) <= 42:
            print(bparse)
            print("--" * 42)
            print(all_erules_used)
            print("==" * 42)
            print(get_token_list(dct['fix'], terminals))
            print("--" * 42)
            print(prog_to_break)
            print("--" * 42)
            print(error_rules)
            print("--" * 42)
            # sys.exit(0)
        if not bparse and len(prog_to_break.split()) <= 42:
            print(bparse)
            print("--" * 42)
            print(all_erules_used)
            print("==" * 42)
            print(get_token_list(dct['fix'], terminals))
            print("--" * 42)
            print(prog_to_break)
            print("--" * 42)
            upd_grammar = deepcopy(egrammar)
            upd_grammar.update_error_grammar_with_erules(rul_dist)
            error_rules = list(set(map(str, prog_error_rules(prog_to_break, upd_grammar, terminals))))
            print(error_rules)
            print("--" * 42)
            # sys.exit(0)
    return (num_of_changes, bparse, error_rules, prog_to_break, dct["duration"])


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    ERROR_GRAMMAR = read_grammar(grammar_file)
    GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    all_rules = defaultdict(int)
    TIMEOUT = 60 * 10 + 5
    parses_bad = 0
    done = 0
    failed = 0
    all_buckets = defaultdict(int)
    parsed_progs_buckets = defaultdict(int)
    rules_per_token_changes = defaultdict(int)
    avrg_rules_used = 0
    top_rules = []
    with open(join(outDir, "parts-order.txt"), "w") as out_file:
        for partPath in list(dataDir.glob('part_*')):
            out_file.write(partPath.name + '\n')
    # with open(join(outDir, "top-erules.txt"), "r") as in_file:
    #     top_erules = in_file.read().split('\n')
    # Use onlt Top-N erules
    erules_cnt = defaultdict(int)
    with open(join(outDir, "erules-augmented.json"), "r") as fin:
        erules_cnt = json.load(fin)
    all_erules = sorted([(erl, erules_cnt[erl]) for erl in erules_cnt], key=lambda x: x[1], reverse=True)
    # print("\n".join(map(lambda x: x[0], all_erules[7:150])))
    max_rules = all_erules[7][1]
    min_rules = all_erules[150][1]
    # print(max_rules, min_rules)
    step = (max_rules - (max_rules + min_rules) // 2) // (150 - 7 - 1)
    probs = list(reversed(list(map(lambda x: x // 2000, range((max_rules + min_rules) // 2, max_rules, step)))))
    # print(len(probs))
    # print(len(all_erules[7:150]))
    if len(probs) != len(all_erules[7:150]):
        print("Not enough probs! Exiting...")
        sys.exit(0)
    # print(probs[0], probs[-1])
    distribution = []
    for er, pr in zip(all_erules[7:150], probs):
        distribution.extend([er[0]] * pr)
    # print(distribution)
    with ProcessPool(max_workers=14, max_tasks=5) as pool:
        for partPath in list(dataDir.glob('part_*')):
            print("#", partPath.name)
            if "augmented" in partPath.name:
                continue
            newDir = join(partPath.parent, "part_augmented_" + partPath.name.split('_')[1])
            goodPath = partPath / "goodPairs.jsonl"
            dataset = []
            if not isdir(newDir):
                mkdir(newDir)
            with open(join(newDir, "erule-dataset-partials-probs.txt"), "w") as outFile:
                goods = map(json.loads, goodPath.read_text().strip().split('\n'))
                print("# Program pairs in goodPath:", len(list(goods)))
                programs = enumerate(goodPath.read_text().strip().split('\n'))
                new_generate = partial(generate, GRAMMAR, ERROR_GRAMMAR, distribution)
                future = pool.map(new_generate, programs, chunksize=1, timeout=TIMEOUT)
                it = future.result()
                while True:
                # for line in programs:
                    try:
                        bruh = next(it)
                        # with time_limit(TIMEOUT):
                        #     bruh = generate(GRAMMAR, ERROR_GRAMMAR, distribution, line)
                        if bruh:
                            tok_chs, parse_bad, erules, lexed_prog, duration = bruh
                            if parse_bad:
                                upd_tokns, next_token = earleyparser_interm_repr.get_updated_seq(lexed_prog, GRAMMAR)
                                store_dataset(lexed_prog, upd_tokns, erules, tok_chs, duration, next_token, outFile)
                                parses_bad += 1
                            done += 1
                            if (failed + done) % 50 == 0:
                                print_results(failed, done, parses_bad, outDir)
                            if parses_bad == 10:
                                sys.exit(0)
                    # except StopIteration:
                    #     break
                    except (TimeoutError, ProcessExpired):
                        failed += 1
                        if (failed + done) % 50 == 0:
                            print_results(failed, done, parses_bad, outDir)
                    except Exception as e:
                        print("WHY here?!", str(e))
                        failed += 1
                        if (failed + done) % 50 == 0:
                            print_results(failed, done, parses_bad, outDir)
            print_results(failed, done, parses_bad, outDir)
