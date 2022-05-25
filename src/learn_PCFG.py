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
import earleyparser_interm_repr


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


def print_results(fails, succs, rules_used, bads, out_dir):
    with open(join(out_dir, "PCFGLearning.txt"), "w") as dataset_file:
        print("# Dataset size:", succs, "/", fails + succs)
        print("# Parse accuracy within time limit (%):", bads * 100.0 / succs)
        print("# Timed out (%):", fails * 100.0 / (fails + succs))
        print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
        dataset_file.write("Dataset size: " + str(succs) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / succs) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")
    with open(join(out_dir, "rules_usage.json"), "w") as out_file:
        json.dump(rules_used, out_file, indent=4)
        print("# ----------------------------------------")


def has_parse(grammar, tup):
    i, l = tup
    dct = json.loads(l)
    rules = list(map(str, earleyparser_interm_repr.get_parse_rules(dct['fix'], grammar)))
    return (i, rules)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    all_rules = defaultdict(int)
    TIMEOUT = 60
    parses_bad = 0
    done = 0
    failed = 0
    with open(join(outDir, "parts-order.txt"), "w") as out_file:
        for partPath in list(dataDir.glob('part_*')):
            out_file.write(partPath.name + '\n')
    with ProcessPool(max_workers=24, max_tasks=5) as pool:
        for partPath in list(dataDir.glob('part_*')):
            print("#", partPath.name)
            # if partPath.name not in ['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7']:
            #     continue
            goodPath = partPath / "goodPairs.jsonl"
            failPath = partPath / "failPairs.jsonl"
            dataset = []
            goods = map(json.loads, goodPath.read_text().strip().split('\n'))
            bugs = map(json.loads, failPath.read_text().strip().split('\n'))
            print("# Program pairs in goodPath:", len(list(goods)))
            print("# Program pairs in failPath:", len(list(bugs)))
            programs = enumerate(goodPath.read_text().strip().split('\n'))
            new_has_parse = partial(has_parse, GRAMMAR)
            future = pool.map(new_has_parse, programs, chunksize=1, timeout=TIMEOUT)
            it = future.result()
            while True:
                try:
                    bruh = next(it)
                    if bruh:
                        _, rules = bruh
                        if rules:
                            for rule in rules:
                                all_rules[rule] += 1
                            parses_bad += 1
                        done += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_rules, parses_bad, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 200 == 0:
                        print_results(failed, done, all_rules, parses_bad, outDir)
                except Exception as e:
                    print("WHY here?!", str(e))
                    failed += 1
                    if (failed + done) % 200 == 0:
                        print_results(failed, done, all_rules, parses_bad, outDir)
            programs = enumerate(failPath.read_text().strip().split('\n'))
            future = pool.map(new_has_parse, programs, chunksize=1, timeout=TIMEOUT)
            it = future.result()
            while True:
                try:
                    bruh = next(it)
                    if bruh:
                        _, rules = bruh
                        if rules:
                            for rule in rules:
                                all_rules[rule] += 1
                            parses_bad += 1
                        done += 1
                        if (failed + done) % 200 == 0:
                            print_results(failed, done, all_rules, parses_bad, outDir)
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    failed += 1
                    if (failed + done) % 200 == 0:
                        print_results(failed, done, all_rules, parses_bad, outDir)
                except Exception as e:
                    print("WHY here?!", str(e))
                    failed += 1
                    if (failed + done) % 200 == 0:
                        print_results(failed, done, all_rules, parses_bad, outDir)
        print_results(failed, done, all_rules, parses_bad, outDir)

# 2/12/2021 - 1:58 => Bad: 10.70 %, Fix: 100.00 %
