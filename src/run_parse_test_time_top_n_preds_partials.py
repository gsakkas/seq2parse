import sys
from collections import defaultdict
from copy import deepcopy
import json
import resource
import timeit
import traceback
from statistics import median_high, median_low, mean
import difflib as df
import re
import subprocess
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
from ecpp_individual_grammar import read_grammar, fixed_lexed_prog, get_token_list, get_actual_token_list, repair_prog

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


def print_results(fails, succs, bads, not_pop_bads, not_pops, accs_per_chgs, all_tok_chngs, avg_time, parse_times, tpsize, sizes, time_gs, user_sames, all_ls, any_ls, out_dir, results_file, repaired_tuplas, repair_csts, used_erls, max_used_erls, max_time=30):
    positives = len(list(filter(lambda dt: dt > 0, time_gs)))
    print("# Dataset size:", succs, "/", fails + succs)
    print("# Parse accuracy within time limit (%):", bads * 100.0 / succs)
    print("# Timed out (%):", fails * 100.0 / (fails + succs))
    print("# => Total parse accuracy (%):", bads * 100.0 / (fails + succs))
    print("# => Not popular parse accuracy (%):", not_pop_bads * 100.0 / not_pops)
    print("# => Mean parse time (sec):", avg_time / (fails + succs))
    print("# => Median parse time (sec):", median_low(parse_times))
    temp_87 = list(map(lambda x: x[0], filter(lambda d: d[1] > 87, zip(parse_times, sizes))))
    print("# => Median parse time (sec):", median_low(temp_87), "for", len(temp_87), "programs with length > 87")
    temp_100 = list(map(lambda x: x[0], filter(lambda d: d[1] > 100, zip(parse_times, sizes))))
    print("# => Median parse time (sec):", median_low(temp_100), "for", len(temp_100), "programs with length > 100")
    temp_250 = list(map(lambda x: x[0], filter(lambda d: d[1] > 250, zip(parse_times, sizes))))
    print("# => Median parse time (sec):", median_low(temp_250), "for", len(temp_250), "programs with length > 250")
    temp_500 = list(map(lambda x: x[0], filter(lambda d: d[1] > 500, zip(parse_times, sizes))))
    print("# => Median parse time (sec):", median_low(temp_500), "for", len(temp_500), "programs with length > 500")
    print("# => Avg. parse time / 50 tokens (sec):", avg_time * 50 / tpsize)
    print("# => Dataset parsed faster than user (%):", positives * 100 / succs)
    print("# => Mean parse time speedup (sec):", mean(time_gs))
    print("# => Median parse time speedup (sec):", median_high(time_gs))
    print("# => Same as user accuracy (%):", user_sames * 100.0 / (fails + succs))
    print("# => All locations fixed accuracy (%):", all_ls * 100.0 / (fails + succs))
    print("# => Any locations fixed accuracy (%):", any_ls * 100.0 / (fails + succs))
    total_erules_used = 0
    for er in used_erls:
        total_erules_used += used_erls[er]
    print("# => Average used error rule position:", sum([used_erls[er] * er for er in used_erls]) / total_erules_used)
    rates = defaultdict(float)
    for dt in range(1, max_time + 1):
        rates[dt] = rate(dt, parse_times)
        if dt <= 60 and (dt % 5 == 0 or dt == 1):
            print(dt, "sec: Parse accuracy =", rates[dt])
    costs_rates = defaultdict(float)
    for dt in range(1, 11):
        costs_rates[dt] = rate(dt, repair_csts)
        if dt <= 5:
            print("Cost =", str(dt) + ": Test set % =", costs_rates[dt])
    erules_rates = defaultdict(float)
    for dt in range(5, 21, 5):
        erules_rates[dt] = rate(dt, max_used_erls)
        print("Top", dt, "error rules: Test set % =", erules_rates[dt])
    for tok_chgs in sorted(accs_per_chgs.keys())[:5]:
        parsed, total_progs = accs_per_chgs[tok_chgs]
        print(tok_chgs, "token changes accuracy (%) =", parsed * 100.0 / total_progs)
    median_times_per_tok_chngs = defaultdict(float)
    for tok_chgs in sorted(accs_per_chgs.keys()):
        temp = list(map(lambda x: x[0], filter(lambda d: d[1] == tok_chgs, zip(parse_times, all_tok_chngs))))
        median_times_per_tok_chngs[tok_chgs] = median_low(temp) if len(temp) > 0 else None
        if tok_chgs <= 5:
            print(tok_chgs, "token changes median parse time (sec) =", median_times_per_tok_chngs[tok_chgs])
    print("---------------------------------------------------")
    with open(join(out_dir, results_file), "w") as dataset_file:
        dataset_file.write("Dataset size: " + str(succs) + "/" + str(fails + succs) + "\n")
        dataset_file.write("Parse accuracy within time limit (%): " + str(bads * 100.0 / succs) + "\n")
        dataset_file.write("Timed out (%): " + str(fails * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Total parse accuracy (%): " + str(bads * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Not popular parse accuracy (%): " + str(not_pop_bads * 100.0 / not_pops) + "\n")
        dataset_file.write("=> Mean parse time (sec): " + str(avg_time / (fails + succs)) + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(parse_times)) + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(temp_87)) + " for " + str(len(temp_87)) + "programs with length > 87" + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(temp_100)) + " for " + str(len(temp_100)) + "programs with length > 100" + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(temp_250)) + " for " + str(len(temp_250)) + "programs with length > 250" + "\n")
        dataset_file.write("=> Median parse time (sec): " + str(median_low(temp_500)) + " for " + str(len(temp_500)) + "programs with length > 500" + "\n")
        dataset_file.write("=> Avg. parse time / 50 tokens (sec): " + str(avg_time * 50 / tpsize) + "\n")
        dataset_file.write("=> Dataset parsed faster than user (%): " + str(positives * 100 / succs) + "\n")
        dataset_file.write("=> Mean parse time speedup (sec): " + str(mean(time_gs)) + "\n")
        dataset_file.write("=> Median parse time speedup (sec): " + str(median_high(time_gs)) + "\n")
        dataset_file.write("=> Same as user accuracy (%): " + str(user_sames * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> All locations fixed accuracy (%): " + str(all_ls * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Any locations fixed accuracy (%): " + str(any_ls * 100.0 / (fails + succs)) + "\n")
        dataset_file.write("=> Average used error rule position: " + str(sum([used_erls[er] * er for er in used_erls]) / total_erules_used) + "\n")
        for dt in range(1, max_time + 1):
            dataset_file.write(str(dt) + " sec: Parse accuracy = " + str(rates[dt]) + "\n")
        for dt in range(1, 11):
            dataset_file.write("Cost = " + str(dt) + ": Test set % = " + str(costs_rates[dt]) + "\n")
        for dt in range(5, 21, 5):
            dataset_file.write("Top " + str(dt) + " error rules: Test set % = " + str(erules_rates[dt]) + "\n")
        for tok_chgs in sorted(accs_per_chgs.keys()):
            parsed, total_progs = accs_per_chgs[tok_chgs]
            dataset_file.write(str(tok_chgs) + " token changes accuracy (%) = " + str(parsed * 100.0 / total_progs) + "\n")
        for tok_chgs in sorted(accs_per_chgs.keys()):
            dataset_file.write(str(tok_chgs) + " token changes median parse time (sec) = "  + str(median_times_per_tok_chngs[tok_chgs]) + "\n")
    with open(join(out_dir, "repaired_prog_pairs.jsonl"), 'w') as repaired_progs_file:
        for pair in repaired_tuplas:
            repaired_progs_file.write(json.dumps(pair) + "\n")


def get_changes(diff):
    line_changes = []
    line_num = 0
    for i, change in enumerate(diff):
        line = change[2:]
        if change[0] == '-':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '+' and line_changes != [] and line_changes[-1][0] == 'added':
                    prev_line = line_changes.pop()[-2]
                    line_changes.append(('replaced', line_num, prev_line, line))
                else:
                    line_changes.append(('deleted', line_num, None, line))
            elif i-1 >= 0 and diff[i-1][0] == '+' and line_changes != [] and line_changes[-1][0] == 'added':
                prev_line = line_changes.pop()[-2]
                line_changes.append(('replaced', line_num, prev_line, line))
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                line_changes.append(('deleted', line_num, None, line))
            line_num += 1
        elif change[0] == '+':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '-' and line_changes != [] and line_changes[-1][0] == 'deleted':
                    prev_line = line_changes.pop()[-1]
                    line_changes.append(('replaced', line_num-1, line, prev_line))
                else:
                    line_changes.append(('added', line_num, line, None))
            elif i-1 >= 0 and diff[i-1][0] == '-' and line_changes != [] and line_changes[-1][0] == 'deleted':
                prev_line = line_changes.pop()[-1]
                line_changes.append(('replaced', line_num-1, line, prev_line))
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                line_changes.append(('added', line_num, line, None))
        elif change[0] == ' ':
            if change[2:].strip() == '':
                line_num += 1
                continue
            line_changes.append(('no_change', line_num, line, line))
            line_num += 1
    return [(ch_type, k) for ch_type, k, _, _ in line_changes if ch_type != 'no_change']


def return_all_changes(bad, fix):
    # bad = bad.split('_NEWLINE_')
    # fix = fix.split('_NEWLINE_')
    diff = list(df.ndiff(bad, fix))
    # print('------------------------------------')
    # print("\n".join(diff))
    # print('------------------------------------')

    line_changes = get_changes(diff)
    # print(line_changes)
    changes = []
    for line_ch in line_changes:
        if line_ch[0] == 'replaced':
            # These are changes within a line
            changes.append(line_ch[1])
        # elif line_ch[0] == 'deleted':
        #     # This are whole line changes (deletions)
        #     changes.append(line_ch[1])
        # else:
        #     # This are whole line changes (additions)
        #     changes.append(line_ch[1])
    return changes


def has_parse(egrammar, max_cost, tup):
    tokns, eruls, tok_chgs, user_time, fixed_tokns, popul, orig_prg, orig_fix, actual_tokns = tup
    # print('=' * 42 + '\n')
    # print(orig_prg.replace("\\n", '\n'))
    # print(orig_fix.replace("\\n", '\n'))
    # print(eruls)
    # print('=' * 42 + '\n')
    upd_grammar_empty = deepcopy(egrammar)
    upd_grammar_empty.update_error_grammar_with_erules([])
    abstr_orig_fixed_seq, orig_fixed_seq, _, _, _  = fixed_lexed_prog(fixed_tokns, upd_grammar_empty, max_cost)

    start_time = timeit.default_timer()
    upd_grammar = deepcopy(egrammar)
    # if 'Err_Colon -> Err_Tag' in eruls:
    #     eruls.remove('Err_Colon -> Err_Tag')
    upd_grammar.update_error_grammar_with_erules(eruls)
    abstr_fixed_seq, fixed_seq, fixed_seq_ops, used_erules, repair_cost = fixed_lexed_prog(tokns, upd_grammar, max_cost)
    repaired_prog = None
    if fixed_seq is None:
        bparse = False
    else:
        repaired_prog = repair_prog(actual_tokns, fixed_seq_ops)
        bparse = True
    # debug_out = '=' * 42 + '\n'
    # debug_out += tokns.replace('_NEWLINE_ ', '\n')
    # debug_out += '\n' + '*' * 42 + '\n'
    # debug_out += fixed_seq_ops.replace('_NEWLINE_ ', '\n')
    # debug_out += '\n' + '*' * 42 + '\n'
    # debug_out += str(eruls)
    # debug_out += '\n' + '*' * 42 + '\n'
    # debug_out += actual_tokns.replace('_NEWLINE_ ', '\n')
    # debug_out += '\n' + '*' * 42 + '\n'
    # debug_out += repaired_prog
    # debug_out += '\n' + '=' * 42 + '\n'
    # print(debug_out)
    run_time = timeit.default_timer() - start_time

    prog_size = len(tokns.split())
    if bparse:
        tokns_lines = tokns.split('_NEWLINE_')
        fixed_orig_lines = orig_fixed_seq.split('_NEWLINE_')
        fixed_seq_lines = fixed_seq.split('_NEWLINE_')
        orig_fixed_lines = return_all_changes(tokns_lines, fixed_orig_lines)
        our_fixed_lines = return_all_changes(tokns_lines, fixed_seq_lines)
        used_erules = [eruls.index(str(er)) for er in used_erules if str(er) in eruls]
        all_correct_lines = all(map(lambda l: l in orig_fixed_lines, our_fixed_lines)) if our_fixed_lines else True
        any_correct_lines = any(map(lambda l: l in orig_fixed_lines, our_fixed_lines)) if our_fixed_lines else True
    dt = user_time - run_time
    if bparse:
        return (bparse, run_time, dt, tok_chgs, prog_size, abstr_orig_fixed_seq == abstr_fixed_seq, all_correct_lines, any_correct_lines, popul, {"orig": orig_prg, "repaired": repaired_prog, "fix": orig_fix}, repair_cost, used_erules)
    else:
        return (bparse, run_time, dt, tok_chgs, prog_size, False, False, False, popul, None, -1, None)


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]), samp_1[4], samp_1[5] == "popular", samp_1[6], samp_1[7], samp_1[8])


def do_all_test(grammar_file, data_dir, out_dir, top_rules_num, ecpp_max_cost, results_file, in_file):
    ERROR_GRAMMAR = read_grammar(grammar_file)
    TIMEOUT = 60 * 5
    parses_bad = 0
    finds_all_lines = 0
    finds_any_lines = 0
    same_as_users = 0
    not_popular_parses = 0
    all_not_populars = 0
    done = 0
    failed = 0
    dataset = []
    avg_run_time = 0.0
    total_size = 0
    all_sizes = []
    all_tok_chngs = []
    parsed_progs_times = []
    time_gains = []
    all_tuplas = []
    repair_costs = []
    all_used_erules = defaultdict(int)
    max_used_erules = []
    accs_per_changes = defaultdict(lambda : (0, 0))
    with ProcessPool(max_workers=28, max_tasks=5) as pool:
        dataset_part_file = join(data_dir, in_file)
        if exists(dataset_part_file):
            with open(dataset_part_file, "r") as inFile:
                dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
        dataset = [(tokns, erules[:top_rules_num], tok_chgs, user_time, fixed_tokns, popular, orig_prog, orig_fix, actual_tkns) for tokns, erules, tok_chgs, user_time, fixed_tokns, popular, orig_prog, orig_fix, actual_tkns in dataset[:15000]]
        for _, _, tok_chgs, _, _, popul, _, _, _ in dataset:
            parsed, total_progs = accs_per_changes[tok_chgs]
            accs_per_changes[tok_chgs] = (parsed, total_progs + 1)
            if not popul:
                all_not_populars += 1
        print("# Syntax Errors to repair:", len(dataset))
        new_has_parse = partial(has_parse, ERROR_GRAMMAR, ecpp_max_cost)
        future = pool.map(new_has_parse, dataset, chunksize=1, timeout=TIMEOUT)
        it = future.result()
        while True:
            try:
                bruh = next(it)
                if bruh:
                    parse_bad, run_time, dt, tok_chgs, size, user_same, all_lines, any_lines, popular, tupla, repair_cst, used_erls = bruh
                    if parse_bad:
                        parses_bad += 1
                        if all_lines:
                            finds_all_lines += 1
                        if any_lines:
                            finds_any_lines += 1
                        if user_same:
                            same_as_users += 1
                        if not popular:
                            not_popular_parses += 1
                        repair_costs.append(repair_cst)
                        for er in used_erls:
                            all_used_erules[er] += 1
                        max_used_erules.append(max(used_erls))
                        parsed, total_progs = accs_per_changes[tok_chgs]
                        accs_per_changes[tok_chgs] = (parsed + 1, total_progs)
                        all_tok_chngs.append(tok_chgs)
                    avg_run_time += run_time
                    parsed_progs_times.append(run_time)
                    total_size += size
                    all_sizes.append(size)
                    time_gains.append(dt)
                    all_tuplas.append(tupla)
                    done += 1
                    if (failed + done) % 50 == 0:
                        print_results(failed, done, parses_bad, not_popular_parses, all_not_populars, accs_per_changes, all_tok_chngs, avg_run_time, parsed_progs_times, total_size, all_sizes, time_gains, same_as_users, finds_all_lines, finds_any_lines, out_dir, results_file, all_tuplas, repair_costs, all_used_erules, max_used_erules, max_time=TIMEOUT+5)
            except StopIteration:
                break
            except (TimeoutError, ProcessExpired):
                failed += 1
                run_time = TIMEOUT
                avg_run_time += run_time
                parsed_progs_times.append(run_time)
                if (failed + done) % 50 == 0:
                    print_results(failed, done, parses_bad, not_popular_parses, all_not_populars, accs_per_changes, all_tok_chngs, avg_run_time, parsed_progs_times, total_size, all_sizes, time_gains, same_as_users, finds_all_lines, finds_any_lines, out_dir, results_file, all_tuplas, repair_costs, all_used_erules, max_used_erules, max_time=TIMEOUT+5)
            except Exception as e:
                print("WHY here?!", str(e))
                traceback.print_tb(e.__traceback__)
                failed += 1
                run_time = TIMEOUT
                avg_run_time += run_time
                parsed_progs_times.append(run_time)
                if (failed + done) % 50 == 0:
                    print_results(failed, done, parses_bad, not_popular_parses, all_not_populars, accs_per_changes, all_tok_chngs, avg_run_time, parsed_progs_times, total_size, all_sizes, time_gains, same_as_users, finds_all_lines, finds_any_lines, out_dir, results_file, all_tuplas, repair_costs, all_used_erules, max_used_erules, max_time=TIMEOUT+5)
        print_results(failed, done, parses_bad, not_popular_parses, all_not_populars, accs_per_changes, all_tok_chngs, avg_run_time, parsed_progs_times, total_size, all_sizes, time_gains, same_as_users, finds_all_lines, finds_any_lines, out_dir, results_file, all_tuplas, repair_costs, all_used_erules, max_used_erules, max_time=TIMEOUT+5)


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])
    input_file = sys.argv[4]
    num_of_tops = int(sys.argv[5])
    max_cost = int(sys.argv[6])

    limit_memory()
    do_all_test(grammarFile, dataDir, outDir, num_of_tops, max_cost, "ECPP-runtime-clean-test-top-1-repair-" + str(num_of_tops) + "-popular-cost-" + str(max_cost) + ".txt", input_file)

    # # For individual testing using:
    # # >>> time python run_parse_test_time_top_n_all_states.py python-grammar.txt repairs/orig_0.py repairs/fix_0.py test-set-top-20-partials-probs.txt 20
    # failPath = dataDir
    # goodPath = outDir
    # bad = failPath.read_text()
    # fix = goodPath.read_text()
    # # print('*' * 42)
    # # print(bad)
    # # print('*' * 42)
    # # print(fix)
    # # print('*' * 42)
    # ERROR_GRAMMAR = read_grammar(grammarFile)
    # terminals = ERROR_GRAMMAR.get_alphabet()
    # # erules0 = ["Err_Break_Stmt -> ", "Err_Close_Paren -> ", "Err_Colon -> ", "Err_Dedent -> ", "Err_For_Keyword -> H For_Keyword", "Err_Indent -> ", "Err_Indent -> Err_Tag", "Err_Literals -> ", "Err_Literals -> H Literals", "Err_Newline -> H Newline", "Err_Open_Paren -> ", "Err_Return_Keyword -> ", "Err_Return_Keyword -> H Return_Keyword", "InsertErr -> )", "InsertErr -> :", "InsertErr -> _NAME_", "InsertErr -> def", "InsertErr -> for", "InsertErr -> if", "InsertErr -> while"]
    # # erules0 = ["Err_Dedent -> ", "Err_Indent -> ", "Err_Indent -> Err_Tag", "Err_Return_Keyword -> ", "Err_Return_Keyword -> H Return_Keyword"]
    # # erules0 = ["Err_Dedent -> ", "Err_Indent -> ", "Err_Literals -> "]
    # # erules0 = ["InsertErr -> )", "Err_Colon -> H Colon", "Err_Colon -> "]
    # utime1 = 4532.0
    # erules1 = ['Err_Break_Stmt -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> ', 'Err_Dedent -> ', 'Err_Indent -> ', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Pass_Stmt -> ', 'Err_Return_Keyword -> ', 'InsertErr -> )', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> def', 'InsertErr -> else', 'InsertErr -> for', 'InsertErr -> if', 'InsertErr -> while']
    # utime2 = 10.0
    # erules2 = ['Err_Arith_Op -> Err_Tag', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> H Colon', 'Err_Comma -> Err_Tag', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> :', 'Err_Tag -> _NUMBER_', 'Err_Tag -> _UNKNOWN_', 'Err_Tag -> else', 'Err_Vfpdef -> Err_Tag', 'InsertErr -> :', 'InsertErr -> =', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_', 'InsertErr -> _UNKNOWN_']
    # utime3 = 12.0
    # erules3 = ['Err_Close_Paren -> H Close_Paren', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_MulDiv_Op -> H MulDiv_Op', 'Err_Newline -> H Newline', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> _NUMBER_', 'Err_Tag -> _STRING_', 'Err_Vfpdef -> Err_Tag', 'Err_Vfpdef -> H Vfpdef', 'InsertErr -> )', 'InsertErr -> +', 'InsertErr -> ,', 'InsertErr -> .', 'InsertErr -> :', 'InsertErr -> =', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> return']
    # utime4 = 8.0
    # erules4 = ['Err_Break_Stmt -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> ', 'Err_Dedent -> ', 'Err_Endmarker -> H Endmarker', 'Err_Indent -> ', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Return_Keyword -> ', 'InsertErr -> )', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> def', 'InsertErr -> else', 'InsertErr -> for', 'InsertErr -> if', 'InsertErr -> while']
    # utime5 = 12.0
    # erules5 = ['Err_Arith_Op -> Err_Tag', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> _NUMBER_', 'Err_Tag -> _UNKNOWN_', 'Err_Unary_Op -> Err_Tag', 'InsertErr -> (', 'InsertErr -> .', 'InsertErr -> :', 'InsertErr -> _DEDENT_', 'InsertErr -> _INDENT_', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_', 'InsertErr -> _UNKNOWN_', 'InsertErr -> return']
    # utime6 = 35.0
    # erules6 = ['Err_Close_Curl_Bracket -> ', 'Err_Close_Paren -> Err_Tag', 'Err_Close_Paren -> H Close_Paren', 'Err_Close_Sq_Bracket -> ', 'Err_Comma -> Err_Tag', 'Err_Comma -> H Comma', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Curl_Bracket -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Open_Sq_Bracket -> ', 'Err_Tag -> :', 'InsertErr -> (', 'InsertErr -> )', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_']
    # utime7 = 9.0
    # erules7 = ['Err_Close_Paren -> H Close_Paren', 'Err_Colon -> H Colon', 'Err_Else_Keyword -> Err_Tag', 'Err_If_Keyword -> Err_Tag', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Return_Keyword -> H Return_Keyword', 'Err_Tag -> _NUMBER_', 'Err_Tag -> _STRING_', 'Err_Vfpdef -> Err_Tag', 'InsertErr -> :', 'InsertErr -> _DEDENT_', 'InsertErr -> _INDENT_', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> def', 'InsertErr -> return', 'InsertErr -> try']
    # utime8 = 6.0
    # erules8 = ['Err_Arith_Op -> ', 'Err_Arith_Op -> H Arith_Op', 'Err_Assign_Op -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> ', 'Err_Comma -> ', 'Err_Comma -> H Comma', 'Err_Literals -> H Literals', 'Err_MulDiv_Op -> ', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Return_Keyword -> Err_Tag', 'Err_Simple_Name -> ', 'Err_Tag -> _NAME_', 'InsertErr -> (', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_']
    # utime9 = 12.0
    # erules9 = ['Err_Arith_Op -> ', 'Err_Arith_Op -> H Arith_Op', 'Err_Assign_Op -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Comma -> ', 'Err_Comma -> H Comma', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_MulDiv_Op -> H MulDiv_Op', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> )', 'InsertErr -> )', 'InsertErr -> .', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_']
    # utime10 = 25.0
    # erules10 = ['Err_Close_Paren -> H Close_Paren', 'Err_Colon -> H Colon', 'Err_Comp_Op -> Err_Tag', 'Err_Comp_Op -> H Comp_Op', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Tag -> =', 'InsertErr -> +=', 'InsertErr -> ,', 'InsertErr -> <', 'InsertErr -> =', 'InsertErr -> [', 'InsertErr -> ]', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> and', 'InsertErr -> is', 'InsertErr -> return']
    # utime11 = 24.0
    # erules11 = ['Err_Assign_Op -> H Assign_Op', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> H Colon', 'Err_Comp_Op -> ', 'Err_Comp_Op -> Err_Tag', 'Err_Comp_Op -> H Comp_Op', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> =', 'Err_Tag -> _NAME_', 'InsertErr -> (', 'InsertErr -> =', 'InsertErr -> [', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> return']
    # utime12 = 80.0
    # erules12 = ['Err_Colon -> ', 'Err_Colon -> H Colon', 'Err_Comp_Op -> H Comp_Op', 'Err_If_Keyword -> Err_Tag', 'Err_In_Keyword -> ', 'Err_In_Keyword -> Err_Tag', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Tag -> _NAME_', 'Err_Tag -> for', 'Err_While_Keyword -> Err_Tag', 'InsertErr -> )', 'InsertErr -> ,', 'InsertErr -> _DEDENT_', 'InsertErr -> _INDENT_', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> for']
    # utime13 = 105.0
    # erules13 = ['Err_Close_Paren -> ', 'Err_Colon -> ', 'Err_Dedent -> ', 'Err_Def_Keyword -> ', 'Err_Endmarker -> H Endmarker', 'Err_Indent -> ', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Return_Keyword -> H Return_Keyword', 'Err_Simple_Name -> ', 'InsertErr -> :', 'InsertErr -> _DEDENT_', 'InsertErr -> _INDENT_', 'InsertErr -> _NAME_', 'InsertErr -> def', 'InsertErr -> for', 'InsertErr -> if', 'InsertErr -> while']
    # utime14 = 15.0
    # erules14 = ['Err_Close_Paren -> ', 'Err_Colon -> ', 'Err_Dedent -> ', 'Err_Dedent -> H Dedent', 'Err_Def_Keyword -> ', 'Err_Endmarker -> H Endmarker', 'Err_For_Keyword -> H For_Keyword', 'Err_In_Keyword -> Err_Tag', 'Err_Indent -> ', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Return_Keyword -> H Return_Keyword', 'Err_Simple_Name -> ', 'InsertErr -> _DEDENT_', 'InsertErr -> _INDENT_', 'InsertErr -> def', 'InsertErr -> for', 'InsertErr -> while']
    # utime15 = 20.0
    # erules15 = ['Err_Arith_Op -> ', 'Err_Assign_Op -> ', 'Err_Assign_Op -> Err_Tag', 'Err_Close_Paren -> ', 'Err_Close_Paren -> Err_Tag', 'Err_Comma -> ', 'Err_Comma -> Err_Tag', 'Err_Comma -> H Comma', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> Err_Tag', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> (', 'Err_Tag -> _NAME_', 'Err_Tag -> _STRING_', 'InsertErr -> (', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_']
    # utime16 = 30.0
    # erules16 = ['Err_Arith_Op -> H Arith_Op', 'Err_Assign_Op -> ', 'Err_Assign_Op -> Err_Tag', 'Err_Close_Paren -> ', 'Err_Close_Paren -> Err_Tag', 'Err_Close_Sq_Bracket -> ', 'Err_Comma -> ', 'Err_Comma -> Err_Tag', 'Err_Comma -> H Comma', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> Err_Tag', 'Err_Open_Paren -> H Open_Paren', 'Err_Open_Sq_Bracket -> H Open_Sq_Bracket', 'Err_Tag -> (', 'InsertErr -> (', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_']
    # utime17 = 8.0
    # erules17 = ['Err_Assign_Op -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> ', 'Err_Colon -> H Colon', 'Err_Comma -> H Comma', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> _NAME_', 'InsertErr -> (', 'InsertErr -> )', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_', 'InsertErr -> def', 'InsertErr -> for']
    # utime19 = 196.0
    # erules19 = ['Err_Break_Stmt -> ', 'Err_Close_Paren -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Colon -> ', 'Err_Dedent -> ', 'Err_Indent -> ', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Pass_Stmt -> ', 'Err_Return_Keyword -> ', 'InsertErr -> )', 'InsertErr -> :', 'InsertErr -> _NAME_', 'InsertErr -> def', 'InsertErr -> else', 'InsertErr -> for', 'InsertErr -> if', 'InsertErr -> while']
    # utime20 = 81.0
    # erules20 = ['Err_And_Bool_Op -> Err_Tag', 'Err_Assign_Op -> H Assign_Op', 'Err_Close_Paren -> ', 'Err_Colon -> ', 'Err_Colon -> H Colon', 'Err_Comp_Op -> ', 'Err_Comp_Op -> Err_Tag', 'Err_Comp_Op -> H Comp_Op', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> =', 'InsertErr -> (', 'InsertErr -> =', 'InsertErr -> [', 'InsertErr -> ]', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> return']
    # utime21 = 25.0
    # erules21 = ['Err_Assign_Op -> H Assign_Op', 'Err_Close_Paren -> ', 'Err_Colon -> H Colon', 'Err_Comp_Op -> ', 'Err_Comp_Op -> Err_Tag', 'Err_Comp_Op -> H Comp_Op', 'Err_Literals -> ', 'Err_Literals -> H Literals', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Open_Paren -> H Open_Paren', 'Err_Tag -> =', 'Err_Tag -> _NAME_', 'InsertErr -> (', 'InsertErr -> =', 'InsertErr -> [', 'InsertErr -> _NAME_', 'InsertErr -> _NUMBER_', 'InsertErr -> _STRING_', 'InsertErr -> return']
    # utime_program8 = 48.0
    # erules_program8 = ["Err_Arith_Op -> H Arith_Op", "Err_Assign_Op -> ", "Err_Assign_Op -> Err_Tag", "Err_Close_Paren -> ", "Err_Close_Paren -> Err_Tag", "Err_Comma -> ", "Err_Comma -> Err_Tag", "Err_Comma -> H Comma", "Err_Comp_Op -> Err_Tag", "Err_Literals -> H Literals", "Err_Newline -> H Newline", "Err_Open_Paren -> ", "Err_Open_Paren -> Err_Tag", "Err_Open_Paren -> H Open_Paren", "Err_Open_Sq_Bracket -> H Open_Sq_Bracket", "Err_Tag -> (", "InsertErr -> (", "InsertErr -> _NAME_", "InsertErr -> _NUMBER_", "InsertErr -> _STRING_"]

    # pr = (get_token_list(bad, terminals), erules9, utime9, get_token_list(fix, terminals), True, bad, fix, get_actual_token_list(bad, terminals))
    # all_names = list(set([tk[1] for tk in zip(get_token_list(bad, terminals).split(), get_actual_token_list(bad, terminals).split()) if tk[0] == '_NAME_'])) + ["1"]
    # for name in all_names:
    #     if "def " + name in bad:
    #         all_names.remove(name)
    # # print(all_names)
    # results = has_parse(ERROR_GRAMMAR, max_cost, pr)
    # # print(results)
    # dct = results[-3]
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("-------------Original Buggy Program---------------")
    # print(dct['orig'][:-1].replace("\\n", '\n'))
    # print("-----------------Repaired Program-----------------")
    # print(dct['repaired'][:-3].replace("\\n", '\n'))
    # print("--------------Original Fix Program----------------")
    # print(dct['fix'][:-1].replace("\\n", '\n'))
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
