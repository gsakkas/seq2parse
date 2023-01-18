import sys
import re
from copy import deepcopy
from os import mkdir, environ
from os.path import join, exists
from pathlib import Path
import difflib as df
import json
import tensorflow as tf
from ecpp_individual_grammar import read_grammar, fixed_lexed_prog, get_token_list, get_actual_token_list, repair_prog
from predict_eccp_classifier_partials import predict_error_rules


def repair(egrammar, max_cost, tokns, eruls, actual_tokns):
    upd_grammar = deepcopy(egrammar)
    upd_grammar.update_error_grammar_with_erules(eruls)
    _, fixed_seq, fixed_seq_ops, _, _ = fixed_lexed_prog(tokns, upd_grammar, max_cost)
    repaired_prog = None
    if fixed_seq is not None:
        repaired_prog = repair_prog(actual_tokns, fixed_seq_ops)
    return repaired_prog


def get_changes(diff):
    line_changes = []
    line_num = 0
    for i, change in enumerate(diff):
        line = change[2:]
        if change[0] == '-':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '+' and line_changes != [] and line_changes[-1][0] == 'Add':
                    prev_line = line_changes.pop()[-2]
                    line_changes.append(('Replace', line_num, prev_line, line))
                else:
                    line_changes.append(('Delete', line_num, None, line))
            elif i-1 >= 0 and diff[i-1][0] == '+' and line_changes != [] and line_changes[-1][0] == 'Add':
                prev_line = line_changes.pop()[-2]
                line_changes.append(('Replace', line_num, prev_line, line))
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                line_changes.append(('Delete', line_num, None, line))
            line_num += 1
        elif change[0] == '+':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '-' and line_changes != [] and line_changes[-1][0] == 'Delete':
                    prev_line = line_changes.pop()[-1]
                    line_changes.append(('Replace', line_num-1, line, prev_line))
                else:
                    line_changes.append(('Add', line_num, line, None))
            elif i-1 >= 0 and diff[i-1][0] == '-' and line_changes != [] and line_changes[-1][0] == 'Delete':
                prev_line = line_changes.pop()[-1]
                line_changes.append(('Replace', line_num-1, line, prev_line))
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                line_changes.append(('Add', line_num, line, None))
        elif change[0] == ' ':
            if change[2:].strip() == '':
                line_num += 1
                continue
            line_changes.append(('no_change', line_num, line, line))
            line_num += 1
    return [(ch_type, k, prev_line, line) for ch_type, k, line, prev_line in line_changes if ch_type != 'no_change']


def get_line_location(orig_line, parsed_line, token_num, option):
    prev_tokens = parsed_line.split()[:token_num]
    column = 0
    for token in prev_tokens:
        orig_line = orig_line.replace(token, '', 1)
        column += len(token)
    column += len(orig_line) - len(orig_line.lstrip(' '))
    return column


if __name__ == "__main__":
    # For single (erroneous) file:
    # >>> python seq2parse.py python-grammar.txt ./models 0 input_prog.py
    grammarFile = sys.argv[1]
    modelsDir = Path(sys.argv[2])
    gpuToUse = '/device:GPU:' + sys.argv[3]
    inputPath = Path(sys.argv[4])

    max_cost = 5
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    input_prog = inputPath.read_text()
    # print('*' * 42)
    # print(input_prog)
    # print('*' * 42)
    ERROR_GRAMMAR = read_grammar(grammarFile)
    terminals = ERROR_GRAMMAR.get_alphabet()

    prog_tokens = get_token_list(input_prog, terminals)
    error_rules = predict_error_rules(grammarFile, modelsDir, gpuToUse, input_prog, True)
    actual_tokens = get_actual_token_list(input_prog, terminals)

    repaired_prog = repair(ERROR_GRAMMAR, max_cost, prog_tokens, error_rules, actual_tokens).replace('_white_space_', ' ').replace('_NEWLINE_', '\n').replace("\\n", '\n')

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("-------------Original Buggy Program---------------")
    print(input_prog)
    print("-----------------Repaired Program-----------------")
    print(repaired_prog[:-3])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    diff_lines = df.ndiff(actual_tokens.split('_NEWLINE_'), get_actual_token_list(repaired_prog, terminals).split('_NEWLINE_'))
    line_changes = [(ch_type, line_num + 1, get_line_location(input_prog.split('\n')[line_num], prev_line.replace('_INDENT_', ''), token_num, ch_type) + 1, len(input_prog.split('\n')[line_num]), (prev, change), ch_line.replace('_INDENT_', ''))
                    for _, line_num, prev_line, ch_line in get_changes(list(diff_lines))
                        for ch_type, token_num, prev, change in get_changes(list(df.ndiff(prev_line.replace('_INDENT_', '').split(), ch_line.replace('_INDENT_', '').split())))]
    # line_changes = [(ch_type, line_num + 1, len(input_prog.split('\n')[line_num]) + 1, change) for ch_type, line_num, prev, change in get_changes(list(diff_lines))]

    result = { "status": "safe"
             , "errors": []
             , "types": {}
             }

    if line_changes:
        result["status"] = "unsafe"
        for ch_type, line_num, start, line_len, (prev, change), ch_line in line_changes:
            if ch_type == 'Add':
                msg = ch_type + " \'" + change.strip() + "\' on line " + str(line_num) + ", column " + str(start) + ":\n\'" + ch_line.strip() + "\'\n"
                column = 1
                length = line_len
            elif ch_type == 'Delete':
                msg = ch_type + " \'" + prev.strip() + "\' on line " + str(line_num) + ", column " + str(start) + ":\n\'" + ch_line.strip() + "\'\n"
                column = start
                length = len(prev)
            else:
                msg = ch_type + " \'" + prev.strip() + "\' with \'" + change.strip() + "\' on line " + str(line_num) + ", column " + str(start) + ":\n\'" + ch_line.strip() + "\'\n"
                column = start
                length = len(prev)
            result["errors"].append({ "message": msg
                                    , "start"  : {"line": line_num, "column": column}
                                    , "stop"   : {"line": line_num, "column": column + length if column + length > 1 else 20}
                                    })
    tmpDir = join(inputPath.parent.absolute(), ".seq2parse")
    if not exists(tmpDir):
        mkdir(tmpDir)
    newInputPath = Path(join(tmpDir, inputPath.name))

    with open(newInputPath.with_suffix(".py.json"), "w") as out_file:
        json.dump(result, out_file, indent=4)
