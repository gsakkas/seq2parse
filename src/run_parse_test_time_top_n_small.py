import sys
from collections import defaultdict
from copy import deepcopy
import timeit
from statistics import median_high, median_low, mean
import difflib as df
import re
from os import environ
from os.path import join
from pathlib import Path
import tqdm
from ecpp_individual_grammar import read_grammar, fixed_lexed_prog, get_token_list, get_actual_token_list, repair_prog
from predict_eccp_classifier_partials import predict_error_rules


def rate(secs, times):
    in_set = list(filter(lambda x: x <= secs, times))
    return len(in_set) * 100.0 / len(times)


def print_results(succs, bads, avg_time, parse_times, time_gs, user_sames, all_ls, any_ls):
    positives = len(list(filter(lambda dt: dt > 0, time_gs)))
    print("---------------------------------------------------")
    print("Dataset size:", succs)
    print("Parse accuracy (%):", bads * 100.0 / succs)
    print("Mean parse time (sec):", avg_time / succs)
    print("Median parse time (sec):", median_low(parse_times))
    print("Dataset repaired faster than user (%):", positives * 100 / succs)
    # print("Mean parse time speedup (sec):", mean(time_gs))
    # print("Median parse time speedup (sec):", median_high(time_gs))
    print("User fix accuracy (%):", user_sames * 100.0 / succs)
    print("All error locations fixed accuracy (%):", all_ls * 100.0 / succs)
    print("Any error locations fixed accuracy (%):", any_ls * 100.0 / succs)
    rates = defaultdict(float)
    for dt in range(1, 61):
        rates[dt] = rate(dt, parse_times)
        if dt <= 60 and (dt % 5 == 0 or dt == 1):
            print(dt, "sec: Parse accuracy =", rates[dt])
    print("---------------------------------------------------")


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
    diff = list(df.ndiff(bad, fix))

    line_changes = get_changes(diff)
    changes = []
    for line_ch in line_changes:
        if line_ch[0] == 'replaced':
            # These are changes within a line
            changes.append(line_ch[1])
    return changes


def has_parse(egrammar, max_cost, tup):
    tokns, eruls, user_time, fixed_tokns, orig_prg, orig_fix, actual_tokns = tup

    upd_grammar_empty = deepcopy(egrammar)
    upd_grammar_empty.update_error_grammar_with_erules([])
    abstr_orig_fixed_seq, orig_fixed_seq, _, _, _  = fixed_lexed_prog(fixed_tokns, upd_grammar_empty, max_cost)

    start_time = timeit.default_timer()
    upd_grammar = deepcopy(egrammar)
    upd_grammar.update_error_grammar_with_erules(eruls)
    abstr_fixed_seq, fixed_seq, fixed_seq_ops, _, _ = fixed_lexed_prog(tokns, upd_grammar, max_cost)
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

    if bparse:
        tokns_lines = tokns.split('_NEWLINE_')
        fixed_orig_lines = orig_fixed_seq.split('_NEWLINE_')
        fixed_seq_lines = fixed_seq.split('_NEWLINE_')
        orig_fixed_lines = return_all_changes(tokns_lines, fixed_orig_lines)
        our_fixed_lines = return_all_changes(tokns_lines, fixed_seq_lines)
        all_correct_lines = all(map(lambda l: l in orig_fixed_lines, our_fixed_lines)) if our_fixed_lines else True
        any_correct_lines = any(map(lambda l: l in orig_fixed_lines, our_fixed_lines)) if our_fixed_lines else True
    dt = user_time - run_time
    if bparse:
        return (bparse, run_time, dt, abstr_orig_fixed_seq == abstr_fixed_seq, all_correct_lines, any_correct_lines, {"orig": orig_prg, "repaired": repaired_prog, "fix": orig_fix})
    else:
        return (bparse, run_time, dt, False, False, False, None)


def do_all_test(grammar_file, data_dir, models_dir, top_rules_num, ecpp_max_cost, do_predict):
    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    parses_bad = 0
    finds_all_lines = 0
    finds_any_lines = 0
    same_as_users = 0
    done = 0
    dataset = []
    user_times = []
    user_fixes = []
    avg_run_time = 0.0
    parsed_progs_times = []
    time_gains = []

    for prog_path in list(data_dir.glob('program_*')):
        if prog_path.name == 'program_stats.txt':
            continue
        prog_file = join(prog_path, "original.py")
        with open(prog_file, "r") as inFile:
            dataset.append(inFile.read())
        time_file = join(prog_path, "user_repair_time.txt")
        with open(time_file, "r") as inFile:
            user_times.append(float(inFile.read()))
        user_fix_file = join(prog_path, "user_fix.py")
        with open(user_fix_file, "r") as inFile:
            user_fixes.append(inFile.read())
    all_error_rules = []
    if do_predict:
        all_error_rules = predict_error_rules(grammar_file, models_dir, '/device:GPU:0', dataset, False, do_sfile=True, max_erules=top_rules_num)
    else:
        top_20_erules = ['InsertErr -> (', 'InsertErr -> )', 'Err_Endmarker -> H Endmarker', 'InsertErr -> :', 'Err_Literals -> ', 'Err_Dedent -> ', 'Err_Indent -> ', 'Err_Close_Paren -> H Close_Paren', 'Err_Literals -> H Literals', 'Err_Colon -> ', 'Err_Close_Paren -> ', 'InsertErr -> _INDENT_', 'Err_Comp_Op -> Err_Tag', 'InsertErr -> _DEDENT_', 'InsertErr -> =', 'InsertErr -> _NAME_', 'Err_Open_Paren -> H Open_Paren', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Tag -> =']
        # top_50_erules = ['InsertErr -> _NUMBER_', 'Err_Endmarker -> H Endmarker', 'Err_Open_Sq_Bracket -> H Open_Sq_Bracket', 'InsertErr -> =', 'Err_Open_Paren -> H Open_Paren', 'Err_Assign_Op -> ', 'InsertErr -> elif', 'Err_Def_Keyword -> ', 'Err_Def_Keyword -> H Def_Keyword', 'InsertErr -> if', 'Err_Colon -> H Colon', 'Err_Colon -> ', 'Err_Close_Paren -> ', 'InsertErr -> :', 'InsertErr -> else', 'Err_Arith_Op -> ', 'InsertErr -> _INDENT_', 'InsertErr -> (', 'InsertErr -> )', 'Err_Newline -> H Newline', 'Err_Return_Keyword -> H Return_Keyword', 'Err_Comp_Op -> ', 'Err_Dedent -> Err_Tag', 'Err_If_Keyword -> ', 'Err_Open_Paren -> ', 'Err_Indent -> Err_Tag', 'InsertErr -> def', 'Err_If_Keyword -> H If_Keyword', 'Err_Close_Sq_Bracket -> ', 'Err_Literals -> H Literals', 'Err_Tag -> _INDENT_', 'InsertErr -> _UNKNOWN_', 'InsertErr -> _DEDENT_', 'InsertErr -> [', 'Err_Literals -> ', 'Err_Dedent -> H Dedent', 'Err_Dedent -> ', 'InsertErr -> _NAME_', 'Err_Tag -> _NAME_', 'Err_Tag -> =', 'Err_Comp_Op -> H Comp_Op', 'InsertErr -> for', 'Err_Close_Paren -> H Close_Paren', 'Err_For_Keyword -> H For_Keyword', 'Err_Tag -> _UNKNOWN_', 'Err_Comp_Op -> Err_Tag', 'Err_Close_Paren -> Err_Tag', 'InsertErr -> _STRING_', 'Err_Comma -> ', 'Err_Indent -> ']
        all_error_rules = [top_20_erules for _ in dataset]
    dataset = [(get_token_list(prog, terminals), erules, user_time, get_token_list(user_fix, terminals), prog, user_fix, get_actual_token_list(prog, terminals))
                for prog, erules, user_time, user_fix in zip(dataset, all_error_rules, user_times, user_fixes)]
    print("Programs to repair:", len(dataset))
    i = 0
    for sample in tqdm.tqdm(dataset):
        i += 1
        parse_bad, run_time, dt, user_same, all_lines, any_lines, _ = has_parse(ERROR_GRAMMAR, ecpp_max_cost, sample)
        if parse_bad:
            parses_bad += 1
            if all_lines:
                finds_all_lines += 1
            if any_lines:
                finds_any_lines += 1
            if user_same:
                same_as_users += 1
        avg_run_time += run_time
        parsed_progs_times.append(run_time)
        time_gains.append(dt)
        done += 1
    print_results(done, parses_bad, avg_run_time, parsed_progs_times, time_gains, same_as_users, finds_all_lines, finds_any_lines)


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    modelsDir = Path(sys.argv[3])
    num_of_tops = int(sys.argv[4])
    max_cost = int(sys.argv[5])
    predict_erules = True
    if len(sys.argv) > 6:
        predict_erules =  sys.argv[6] == 'predict'

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    do_all_test(grammarFile, dataDir, modelsDir, num_of_tops, max_cost, predict_erules)
