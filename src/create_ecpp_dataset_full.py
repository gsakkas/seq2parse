import sys
from collections import defaultdict
from copy import deepcopy
import resource
# import timeit
# import multiprocessing.pool
# from multiprocessing import TimeoutError
from os.path import join
from functools import partial
from pathlib import Path
import json
# import signal
# from contextlib import contextmanager
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
# import tqdm
import earleyparser
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list, get_actual_token_list
from run_extract_token_distribution import return_all_changes
import earleyparser_interm_repr
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


def store_dataset(erule_dataset, out_file):
    for tokns, eruls, tok_chgs, dur, upd_tkns, next_tkn, fix_tkns, origb, origf, orig_tkns in erule_dataset:
        out_file.write(tokns + " <||> " + upd_tkns + " <||> " + " <++> ".join(eruls) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + next_tkn + " <||> " + fix_tkns + " <||> " + repr(origb) + " <||> " + repr(origf) + " <||> " + orig_tkns + "\n")


def has_parse(grammar, egrammar, igrammar, tup):
    # start_time = timeit.default_timer()
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
            upd_grammar.update_error_grammar(token_changes)
            error_rules = list(set(map(str, prog_error_rules(dct['bad'], upd_grammar, terminals))))
            tokens = get_token_list(dct['bad'], terminals)
            fixed_tkns = get_token_list(dct['fix'], terminals)
            orig_tkns = get_actual_token_list(dct['bad'], terminals)
            if error_rules == []:
                bparse = False
            else:
                bparse = True
            upd_tokns, next_tkn = earleyparser_interm_repr.get_updated_seq(tokens, igrammar)
            return (i, num_of_changes, bparse, error_rules, len(error_rules), tokens, dct["duration"], upd_tokns, next_tkn, fixed_tkns, dct['bad'], dct['fix'], orig_tkns)
        return (i, -1, False, [], -1, [], -1, None, None, None, None, None, None)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    GRAMMAR = earleyparser.read_grammar(grammar_file)
    INTERIM_GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    rules_used = {}
    with open("rules_usage.json", "r") as in_file:
        rules_used = json.load(in_file)
    INTERIM_GRAMMAR.update_probs(rules_used)
    all_erules = defaultdict(int)
    TIMEOUT = 60 * 30 + 5
    parses_bad = 0
    done = 0
    failed = 0
    all_buckets = defaultdict(int)
    parsed_progs_buckets = defaultdict(int)
    erules_per_token_changes = defaultdict(int)
    avrg_erules_used = 0
    with ProcessPool(max_workers=24, max_tasks=5) as pool:
    # pool = multiprocessing.pool.Pool(12)
        with open(join(outDir, "parts-order.txt"), "w") as out_file:
            for partPath in list(dataDir.glob('part_*')):
                out_file.write(partPath.name + '\n')
        for partPath in list(dataDir.glob('part_2018_*')):
            print("#", partPath.name)
            if partPath.name not in ['part_2018_1']:
                continue
            goodPath = partPath / "goodPairs.jsonl"
            failPath = partPath / "failPairs.jsonl"
            dataset = []
            good_ids = []
            fail_ids = []
            with open(join(partPath, "erule-dataset-partials-probs-fixes.txt"), "w") as outFile:
                goods = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, goodPath.read_text().strip().split('\n')))
                bugs = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, failPath.read_text().strip().split('\n')))
                print("# Syntax Errors in goodPath:", len(list(goods)))
                print("# Syntax Errors in failPath:", len(list(bugs)))
                programs = enumerate(goodPath.read_text().strip().split('\n'))
                new_has_parse = partial(has_parse, GRAMMAR, ERROR_GRAMMAR, INTERIM_GRAMMAR)
                # it = pool.imap_unordered(new_has_parse, list(programs), 1)
                future = pool.map(new_has_parse, programs, chunksize=1, timeout=TIMEOUT)
                it = future.result()
                while True:
                # for line in goodPath.read_text().strip().split('\n'):
                    try:
                        # bruh = it.next(timeout=TIMEOUT)
                        bruh = next(it)
                        # with time_limit(TIMEOUT):
                        # bruh = has_parse(line)
                        if bruh:
                            idx, tok_chs, parse_bad, erules, num_of_rules, lexed_prog, duration, upd_tokens, next_token, lexed_fixed_prog, orig_bad, orig_fix, orig_tokens = bruh
                            if tok_chs < 0:
                                good_ids.append(idx)
                            else:
                                good_ids.append(idx)
                                if parse_bad:
                                    avrg_erules_used += num_of_rules
                                    for rule in erules:
                                        all_erules[rule] += 1
                                    parses_bad += 1
                                    erules_per_token_changes[tok_chs] += num_of_rules
                                    parsed_progs_buckets[tok_chs] += 1
                                    dataset.append((lexed_prog, erules, tok_chs, duration, upd_tokens, next_token, lexed_fixed_prog, orig_bad, orig_fix, orig_tokens))
                                    store_dataset(dataset, outFile)
                                    dataset = []
                                all_buckets[tok_chs] += 1
                                done += 1
                                if (failed + done) % 200 == 0:
                                    print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                                    # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                    except StopIteration:
                        break
                    except (TimeoutError, ProcessExpired):
                        failed += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                            # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                    except Exception as e:
                        print("WHY here?!", str(e))
                        failed += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                            # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                good_ids = set(good_ids)
                programs = enumerate(failPath.read_text().strip().split('\n'))
                # it = pool.imap_unordered(new_has_parse, programs, 1)
                future = pool.map(new_has_parse, programs, chunksize=1, timeout=TIMEOUT)
                it = future.result()
                while True:
                # for line in failPath.read_text().strip().split('\n'):
                    try:
                        # bruh = it.next(timeout=TIMEOUT)
                        bruh = next(it)
                        # with time_limit(TIMEOUT):
                        # bruh = has_parse(line)
                        if bruh:
                            idx, tok_chs, parse_bad, erules, num_of_rules, lexed_prog, duration, upd_tokens, next_token, lexed_fixed_prog, orig_bad, orig_fix, orig_tokens = bruh
                            if tok_chs < 0:
                                fail_ids.append(idx)
                            else:
                                fail_ids.append(idx)
                                if parse_bad:
                                    avrg_erules_used += num_of_rules
                                    for rule in erules:
                                        all_erules[rule] += 1
                                    parses_bad += 1
                                    erules_per_token_changes[tok_chs] += num_of_rules
                                    dataset.append((lexed_prog, erules, tok_chs, duration, upd_tokens, next_token, lexed_fixed_prog, orig_bad, orig_fix, orig_tokens))
                                    store_dataset(dataset, outFile)
                                    dataset = []
                                all_buckets[tok_chs] += 1
                                done += 1
                                if (failed + done) % 200 == 0:
                                    print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                                    # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                    except StopIteration:
                        break
                    except (TimeoutError, ProcessExpired):
                        failed += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                            # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                    except Exception as e:
                        print("WHY here?!", str(e))
                        failed += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                            # store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
                fail_ids = set(fail_ids)
                programs = list(enumerate(goodPath.read_text().strip().split('\n')))
                for pr in filter(lambda pr: pr[0] not in good_ids, programs):
                    p = json.loads(pr[1])
                    if not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError":
                        num_of_changes = sum([len(ch[2].split()) if ch[0] == 'added' else len(ch[1].split()) for ch in return_all_changes(pr[1], terminals)])
                        # print('# ================================')
                        # print('# Token Changes =', num_of_changes)
                        # print('# --------------------------')
                        # print(p['bad'])
                        # print('# --------------------------')
                        # print(p['fix'])
                        # print('# ================================')
                        if num_of_changes <= 40:
                            all_buckets[num_of_changes] += 1
                programs = list(enumerate(failPath.read_text().strip().split('\n')))
                for pr in filter(lambda pr: pr[0] not in fail_ids, programs):
                    p = json.loads(pr[1])
                    if not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError":
                        num_of_changes = sum([len(ch[2].split()) if ch[0] == 'added' else len(ch[1].split()) for ch in return_all_changes(pr[1], terminals)])
                        if num_of_changes <= 40:
                            all_buckets[num_of_changes] += 1
            print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
            store_results(parsed_progs_buckets, all_buckets, erules_per_token_changes, outDir)
    # l = json.dumps(pr)
    # dct = json.loads(l)
    # print(dct['bad'])
    # print(dct['fix'])
    # print(has_parse((0, l)))
    # print(prog_has_parse(dct['bad'], grammar))
    # print(prog_has_parse(dct['fix'], grammar))
    # failPath = dataDir
    # goodPath = outDir
    # bad = failPath.read_text()
    # fix = goodPath.read_text()
    # print(bad)
    # print(fix)
    # pr = {'bad': bad, 'fix': fix, 'errMsg': 'SyntaxError', 'index': 1, 'fixIndex': 1, 'duration': 1}
    # results = has_parse(GRAMMAR, ERROR_GRAMMAR, (0, json.dumps(pr)))
    # print(results)

# 2/12/2021 - 1:58 => Bad: 10.70 %, Fix: 100.00 %
