import sys
from collections import defaultdict
from copy import deepcopy
import resource
import timeit
from statistics import median_high, median_low, mean
# import multiprocessing.pool
# from multiprocessing import TimeoutError
from os.path import join, exists
from functools import partial
from pathlib import Path
# import signal
# from contextlib import contextmanager
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
# import tqdm
from ecpp_individual_grammar import read_grammar, lexed_prog_has_parse

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


def rate(secs, times):
    in_set = list(filter(lambda x: x <= secs, times))
    return len(in_set) * 100.0 / len(times)


def print_results(fails, succs, bads, avg_time, parse_times, tpsize, time_gs, out_dir, results_file, max_time=30):
    positives = len(list(filter(lambda dt: dt > 0, time_gs)))
    print("# Dataset size:", succs, "/", fails + succs)
    print("# Parse accuracy within time limit (%):", bads * 100.0 / succs)
    print("# Timed out (%):", fails * 100.0 / (fails + succs))
    print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
    print("# => Mean parse time (sec):", avg_time / succs)
    print("# => Median parse time (sec):", median_low(parse_times))
    print("# => Avg. parse time / 50 tokens (sec):", avg_time * 50 / tpsize)
    print("# => Dataset parsed faster than user (%):", positives * 100 / succs)
    print("# => Mean parse time speedup (sec):", mean(time_gs))
    print("# => Median parse time speedup (sec):", median_high(time_gs))
    rates = defaultdict(float)
    for dt in range(1, max_time + 1):
        rates[dt] = rate(dt, parse_times)
        if dt % 5 == 0 or dt == 1:
            print(dt, "sec: Parse accuracy =", rates[dt])
    print("---------------------------------------------------")
    with open(join(out_dir, results_file), "w") as dataset_file:
        dataset_file.write("Dataset size: " + str(succs) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / succs) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Mean parse time (sec): " + str(avg_time / succs) + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(parse_times)) + "\n")
        dataset_file.write("=> Avg. parse time / 50 tokens (sec): " + str(avg_time * 50 / tpsize) + "\n")
        dataset_file.write("=> Dataset parsed faster than user (%): " + str(positives * 100 / succs) + "\n")
        dataset_file.write("=> Mean parse time speedup (sec): " + str(mean(time_gs)) + "\n")
        dataset_file.write("=> Median parse time speedup (sec): " + str(median_high(time_gs)) + "\n")
        for dt in range(1, max_time + 1):
            dataset_file.write(str(dt) + " sec: Parse accuracy = " + str(rates[dt]) + "\n")


def has_parse(egrammar, tup):
    start_time = timeit.default_timer()
    tokns, eruls, user_time = tup
    upd_grammar = deepcopy(egrammar)
    upd_grammar.update_error_grammar_with_erules(eruls)
    error_rules = list(set(map(str, lexed_prog_has_parse(tokns, upd_grammar))))
    if error_rules == []:
        bparse = False
    else:
        bparse = True
    run_time = timeit.default_timer() - start_time
    prog_size = len(tokns.split())
    dt = user_time - run_time
    return (bparse, run_time, dt, prog_size)


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]))


def do_all_test(grammar_file, data_dir, out_dir, top_n, results_file):
    ERROR_GRAMMAR = read_grammar(grammar_file)
    TIMEOUT = 60 * 5 + 5
    parses_bad = 0
    done = 0
    failed = 0
    dataset = []
    avg_run_time = 0.0
    total_size = 0
    parsed_progs_times = []
    time_gains = []
    with ProcessPool(max_workers=22, max_tasks=5) as pool:
        dataset_part_file = join(data_dir, "test-set-top-50-preds.txt")
        if exists(dataset_part_file):
            with open(dataset_part_file, "r") as inFile:
                dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
        dataset = [(tokns, erules, user_time) for tokns, erules, _, user_time in dataset]
        print("# Syntax Errors to repair:", len(dataset))
        new_has_parse = partial(has_parse, ERROR_GRAMMAR)
        future = pool.map(new_has_parse, dataset, chunksize=1, timeout=TIMEOUT)
        it = future.result()
        while True:
            try:
                bruh = next(it)
                if bruh:
                    parse_bad, run_time, dt, size = bruh
                    if parse_bad:
                        parses_bad += 1
                    avg_run_time += run_time
                    parsed_progs_times.append(run_time)
                    total_size += size
                    time_gains.append(dt)
                    done += 1
                    if (failed + done) % 50 == 0:
                        print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, out_dir, results_file, max_time=180)
            except StopIteration:
                break
            except (TimeoutError, ProcessExpired):
                failed += 1
                if (failed + done) % 50 == 0:
                    print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, out_dir, results_file, max_time=180)
            except Exception as e:
                print("WHY here?!", str(e))
                failed += 1
                if (failed + done) % 50 == 0:
                    print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, out_dir, results_file, max_time=180)
        print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, out_dir, results_file, max_time=180)


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])
    num_of_tops = int(sys.argv[4])

    limit_memory()
    do_all_test(grammarFile, dataDir, outDir, num_of_tops, "ECPP-runtime-test-top-" + str(num_of_tops) + ".txt")

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
