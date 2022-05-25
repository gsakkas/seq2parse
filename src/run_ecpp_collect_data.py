import sys
from collections import defaultdict
from copy import deepcopy
import resource
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
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules
from run_extract_token_distribution import return_all_changes

# pr = {"bad": "def check_rush_hour(day, hour, minute):\n    if day== 'Monday' or day=='Tuesday' or day=='Wednesday' or day=='Thursday' or day=='Friday':\n        if hour>= 5:\n            if hour<= 9:\n                if minute= 0 or minute=1 or minute=2 or minute=3 or minute=4 or minute=5 or minute=6:\n                    return True\n                else:\n                    return False\n        if hour>= 15:\n            if hour<= 19:\n                elif minute= 0:\n                    return True\n                else:\n                    return False\n    else:\n        return False\n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "fix": "def check_rush_hour(day, hour, minute):\n    if day== \"Monday\" or day==\"Tuesday\" or day==\"Wednesday\" or day==\"Thursday\" or day==\"Friday\":\n        if hour>= 5:\n            if hour<= 9:\n                if minute >=0:\n                    if minute<=30:\n                        return True\n                    else:\n                        return False\n        elif hour>= 15:\n            if hour<= 19:\n               return True\n            else:\n                return False\n    else:\n        return False\n        \n    \n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "index": 4818, "fixIndex": 4830, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 1205}

# pr = {"bad": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print i\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "fix": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print (i)\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "index": 6, "fixIndex": 7, "errMsg": "SyntaxError", "isConsecutive": True, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 15}

# pr = {"bad": "#PROMEDIO FINAL\n>>> #PROMEDIO FINAL\n>>> print(\"\u00bfCuantas unidades son?:\")\n\u00bfCuantas unidades son?:\n>>> n=int(input())\nsuma=0\n>>> for i in range (n):\n    print(\"Dame la calificacion\",i+1)\n    cal=int(input())\n#Next i", "fix": "#PROMEDIO FINAL\n#PROMEDIO FINAL\nprint(\"\u00bfCuantas unidades son?:\")\n\nn=int(input())\nsuma=0\nfor i in range (n):\n    print(\"Dame la calificacion\",i+1)\n    cal=int(input())\n#Next i", "index": 575, "fixIndex": 586, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": ["7", "10", "10", "8", "8", "9", "8", "8"], "mergedInput": [], "duration": 137}


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


def print_results(fails, succs, erules, avg_erules, pbad, out_dir):
    total_parses = len(pbad)
    bads = len(list(filter(lambda x: x, pbad)))
    with open(join(out_dir, "ErrorCorrectingEarleyParsesDist.txt"), "w") as dataset_file:
        print("# Dataset size:", total_parses, "/", fails + succs)
        print("# Parse accuracy within time limit (%):", bads * 100.0 / total_parses)
        print("# Timed out (%):", fails * 100.0 / (fails + succs))
        print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
        dataset_file.write("Dataset size: " + str(total_parses) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / total_parses) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")
    with open(join(out_dir, "erules.json"), "w") as out_file:
        # print("#", erules)
        print("#", avg_erules * 1.0 / done)
        json.dump(erules, out_file, indent=4)
        print("# ----------------------------------------")


def store_results(bkts, alls, out_dir):
    accuracies = dict()
    for k in alls:
        accuracies[k] = bkts[k] * 100.0 / alls[k]
    print("#", accuracies)
    # print(alls)
    print("# ----------------------------------------")
    with open(join(out_dir, "accuracies-per-token.json"), "w") as out_file:
        json.dump(accuracies, out_file, indent=4)
    with open(join(out_dir, "elems-per-token.json"), "w") as out_file:
        json.dump(alls, out_file, indent=4)


def has_parse(grammar, egrammar, tup):
    i, l = tup
    dct = json.loads(l)
    if dct["index"] in [466, 467, 468] and dct["fixIndex"] == 470 and dct["duration"] in [362, 375, 380]:
        return None
    elif dct["errMsg"] == "SyntaxError":
        if not earleyparser.prog_has_parse(dct['bad'], grammar):
            upd_grammar = deepcopy(egrammar)
            terminals = egrammar.get_alphabet()
            # print(terminals)
            token_changes = return_all_changes(l, terminals)
            if not token_changes:
                return None
            num_of_changes = sum([len(ch[2].split()) if ch[0] == 'added' else len(ch[1].split()) for ch in token_changes])
            if num_of_changes > 40:
                return None
            # print(token_changes)
            upd_grammar.update_error_grammar(token_changes)
            # print(dct["index"], dct["fixIndex"], len(str(upd_grammar).split('\n')))
            # print(len(str(egrammar).split('\n')))
            error_rules = list(set(map(str, prog_error_rules(dct['bad'], upd_grammar, terminals))))
            # print(dct["index"], dct["fixIndex"], error_rules)
            if error_rules == []:
                bparse = False
            else:
                bparse = True
            # fparse = prog_has_parse(dct['fix'], egrammar, terminals)
            # if num_of_changes < 8 and not bparse:
            #     print('# ================================')
            #     print('# Token Changes =', num_of_changes)
            #     print('# --------------------------')
            #     print(dct['bad'])
            #     print('# --------------------------')
            #     print(dct['fix'])
            #     print('# ================================')
            return (i, num_of_changes, bparse, error_rules, len(error_rules))
        return (i, -1, False, [], -1)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    GRAMMAR = earleyparser.read_grammar(grammar_file)
    parses_bad = []
    all_erules = defaultdict(int)
    TIMEOUT = 60 * 25 + 5
    done = 0
    failed = 0
    good_ids = []
    fail_ids = []
    buckets = defaultdict(int)
    all_buckets = defaultdict(int)
    avrg_erules_used = 0
    with ProcessPool(max_workers=14, max_tasks=5) as pool:
    # pool = multiprocessing.pool.Pool(12)
        for partPath in list(dataDir.glob('part_15')):
            goodPath = partPath / "goodPairs.jsonl"
            failPath = partPath / "failPairs.jsonl"
            goods = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, goodPath.read_text().strip().split('\n')))
            bugs = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, failPath.read_text().strip().split('\n')))
            print("# Syntax Errors in goodPath:", len(list(goods)))
            print("# Syntax Errors in failPath:", len(list(bugs)))
            programs = enumerate(goodPath.read_text().strip().split('\n'))
            new_has_parse = partial(has_parse, GRAMMAR, ERROR_GRAMMAR)
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
                        idx, tok_chs, parse_bad, erules, num_of_rules = bruh
                        if tok_chs < 0:
                            good_ids.append(idx)
                        else:
                            avrg_erules_used += num_of_rules
                            parses_bad.append(parse_bad)
                            for rule in erules:
                                all_erules[rule] += 1
                            good_ids.append(idx)
                            if parse_bad:
                                buckets[tok_chs] += 1
                            all_buckets[tok_chs] += 1
                            done += 1
                            if (failed + done) % 10 == 0:
                                print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                                # store_results(buckets, all_buckets, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                        # store_results(buckets, all_buckets, outDir)
                except Exception as e:
                    print("WHY here?!", str(e))
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                        # store_results(buckets, all_buckets, outDir)
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
                        idx, tok_chs, parse_bad, erules, num_of_rules = bruh
                        if tok_chs < 0:
                            fail_ids.append(idx)
                        else:
                            avrg_erules_used += num_of_rules
                            parses_bad.append(parse_bad)
                            for rule in erules:
                                all_erules[rule] += 1
                            fail_ids.append(idx)
                            if parse_bad:
                                buckets[tok_chs] += 1
                            all_buckets[tok_chs] += 1
                            done += 1
                            if (failed + done) % 10 == 0:
                                print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                                # store_results(buckets, all_buckets, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                        # store_results(buckets, all_buckets, outDir)
                except Exception as e:
                    print("WHY here?!", str(e))
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, all_erules, avrg_erules_used, parses_bad, outDir)
                        # store_results(buckets, all_buckets, outDir)
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
            store_results(buckets, all_buckets, outDir)

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
