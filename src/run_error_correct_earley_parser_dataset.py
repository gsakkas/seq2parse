import sys
from collections import defaultdict
# import multiprocessing.pool
# from multiprocessing import TimeoutError
from os.path import join
from pathlib import Path
import json
# import signal
# from contextlib import contextmanager
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
# import tqdm
import earleyparser
from ecpp_dist import read_grammar, prog_has_parse
from run_extract_token_distribution import return_changes

# pr = {"bad": "def check_rush_hour(day, hour, minute:\n    if day== 'Monday' or day=='Tuesday' or day=='Wednesday' or day=='Thursday' or day=='Friday':\n        if hour>= 5:\n            if hour<= 9:\n                if minute= 0 or minute=1 or minute=2 or minute=3 or minute=4 or minute=5 or minute=6:\n                    return True\n                else:\n                    return False\n        if hour>= 15:\n            if hour<= 19:\n                elif minute= 0:\n                    return True\n                else:\n                    return False\n    else:\n        return False\n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "fix": "def check_rush_hour(day, hour, minute):\n    if day== \"Monday\" or day==\"Tuesday\" or day==\"Wednesday\" or day==\"Thursday\" or day==\"Friday\":\n        if hour>= 5:\n            if hour<= 9:\n                if minute >=0:\n                    if minute<=30:\n                        return True\n                    else:\n                        return False\n        elif hour>= 15:\n            if hour<= 19:\n               return True\n            else:\n                return False\n    else:\n        return False\n        \n    \n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "index": 4818, "fixIndex": 4830, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 1205}

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


def print_results(fails, succs, pfixed, pbad, out_dir):
    total_parses = len(pfixed)
    bads = len(list(filter(lambda x: x, pbad)))
    fixs = len(list(filter(lambda x: x, pfixed)))
    with open(join(out_dir, "ErrorCorrectingEarleyParsesDist.txt"), "w") as dataset_file:
        print("Dataset size:", total_parses, "/", fails + succs)
        print("Bad Dataset Parsed (%):", bads * 100.0 / total_parses)
        print("Fixed Dataset Parsed (%):", fixs * 100.0 / total_parses)
        print("Timed out (%):", fails * 100.0 / (fails + succs))
        print("----------------------------------------")
        dataset_file.write("Dataset size: " + str(total_parses) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Bad Dataset Parsed (%): " + str(bads * 100.0 / total_parses) + "\n")
        dataset_file.write("Fixed Dataset Parsed (%): " + str(fixs * 100.0 / total_parses) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")


def store_results(bkts, alls, out_dir):
    accuracies = dict()
    for k in alls:
        accuracies[k] = bkts[k] * 100.0 / alls[k]
    print(accuracies)
    # print(alls)
    print("----------------------------------------")
    with open(join(out_dir, "accuracies-per-token.json"), "w") as out_file:
        json.dump(accuracies, out_file, indent=4)
    with open(join(out_dir, "elems-per-token.json"), "w") as out_file:
        json.dump(alls, out_file, indent=4)


def has_parse(tup):
    i, l = tup
    dct = json.loads(l)
    if dct["index"] in [466, 467, 468] and dct["fixIndex"] == 470 and dct["duration"] in [362, 375, 380]:
        pass
    elif dct["errMsg"] == "SyntaxError":
        if not earleyparser.prog_has_parse(dct['bad'], GRAMMAR):
            bparse = prog_has_parse(dct['bad'], ERROR_GRAMMAR)
            fparse = prog_has_parse(dct['fix'], ERROR_GRAMMAR)
            token_changes = len(return_changes(l))
            # if token_changes == 0 and not bparse:
            #     print(dct['bad'])
            #     print('--------------------------')
            #     print(dct['fix'])
            return (i, token_changes, bparse, fparse)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    ERROR_GRAMMAR = read_grammar(grammar_file)
    GRAMMAR = earleyparser.read_grammar(grammar_file)

    parses_bad = []
    parses_fix = []
    TIMEOUT = 60 * 5 + 5
    done = 0
    failed = 0
    good_ids = []
    fail_ids = []
    buckets = defaultdict(int)
    all_buckets = defaultdict(int)
    with ProcessPool(max_workers=12) as pool:
        # pool = multiprocessing.pool.Pool(12)
        for partPath in list(dataDir.glob('part_15')):
            goodPath = partPath / "goodPairs.jsonl"
            failPath = partPath / "failPairs.jsonl"
            goods = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, goodPath.read_text().strip().split('\n')))
            bugs = filter(lambda p: not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError", map(json.loads, failPath.read_text().strip().split('\n')))
            print("Syntax Errors in goodPath:", len(list(goods)))
            print("Syntax Errors in failPath:", len(list(bugs)))
            # it = pool.imap_unordered(has_parse, goodPath.read_text().strip().split('\n'), 1)
            programs = enumerate(goodPath.read_text().strip().split('\n'))
            future = pool.map(has_parse, programs, chunksize=1, timeout=TIMEOUT)
            it = future.result()
            while True:
            # for line in goodPath.read_text().strip().split('\n'):
                try:
                    # bruh = it.next(timeout=TIMEOUT)
                    bruh = next(it)
                    # with time_limit(TIMEOUT):
                    # bruh = has_parse(line)
                    if bruh:
                        idx, tok_chs, parse_bad, parse_fix = bruh
                        parses_bad.append(parse_bad)
                        parses_fix.append(parse_fix)
                        good_ids.append(idx)
                        if parse_bad:
                            buckets[tok_chs] += 1
                        all_buckets[tok_chs] += 1
                        done += 1
                        if (failed + done) % 10 == 0:
                            print_results(failed, done, parses_fix, parses_bad, outDir)
                            store_results(buckets, all_buckets, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, parses_fix, parses_bad, outDir)
                        store_results(buckets, all_buckets, outDir)
            good_ids = set(good_ids)
            # it = pool.imap_unordered(has_parse, failPath.read_text().strip().split('\n'), 1)
            programs = enumerate(failPath.read_text().strip().split('\n'))
            future = pool.map(has_parse, programs, chunksize=1, timeout=TIMEOUT)
            it = future.result()
            while True:
            # for line in failPath.read_text().strip().split('\n'):
                try:
                    # bruh = it.next(timeout=TIMEOUT)
                    bruh = next(it)
                    # with time_limit(TIMEOUT):
                    # bruh = has_parse(line)
                    if bruh:
                        idx, tok_chs, parse_bad, parse_fix = bruh
                        parses_bad.append(parse_bad)
                        parses_fix.append(parse_fix)
                        fail_ids.append(idx)
                        if parse_bad:
                            buckets[tok_chs] += 1
                        all_buckets[tok_chs] += 1
                        done += 1
                        if (failed + done) % 10 == 0:
                            print_results(failed, done, parses_fix, parses_bad, outDir)
                            store_results(buckets, all_buckets, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 10 == 0:
                        print_results(failed, done, parses_fix, parses_bad, outDir)
                        store_results(buckets, all_buckets, outDir)
            fail_ids = set(fail_ids)
            fail_ids = filter(lambda pr: pr[0] not in fail_ids, enumerate(failPath.read_text().strip().split('\n')))
    programs = list(enumerate(goodPath.read_text().strip().split('\n')))
    for pr in filter(lambda pr: pr[0] not in good_ids, programs):
        p = json.loads(pr[1])
        if not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError":
            all_buckets[len(return_changes(pr[1]))] += 1
    programs = list(enumerate(failPath.read_text().strip().split('\n')))
    for pr in filter(lambda pr: pr[0] not in fail_ids, programs):
        p = json.loads(pr[1])
        if not (p["index"] in [466, 467, 468] and p["fixIndex"] == 470 and p["duration"] in [362, 375, 380]) and p["errMsg"] == "SyntaxError":
            all_buckets[len(return_changes(pr[1]))] += 1
    print_results(failed, done, parses_fix, parses_bad, outDir)
    store_results(buckets, all_buckets, outDir)
    # l = json.dumps(pr)
    # dct = json.loads(l)
    # print(prog_has_parse(dct['bad'], grammar))
    # print(prog_has_parse(dct['fix'], grammar))

# 2/12/2021 - 1:58 => Bad: 10.70 %, Fix: 100.00 %
