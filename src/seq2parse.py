import sys
from copy import deepcopy
from pathlib import Path
import json
from ecpp_individual_grammar import read_grammar, fixed_lexed_prog, get_token_list, get_actual_token_list, repair_prog

def has_parse(egrammar, max_cost, tokns, eruls, actual_tokns):
    upd_grammar = deepcopy(egrammar)
    upd_grammar.update_error_grammar_with_erules(eruls)
    _, fixed_seq, fixed_seq_ops, _, _ = fixed_lexed_prog(tokns, upd_grammar, max_cost)
    repaired_prog = None
    if fixed_seq is not None:
        repaired_prog = repair_prog(actual_tokns, fixed_seq_ops)
    return repaired_prog


if __name__ == "__main__":
    # For single (erroneous) file:
    # >>> python seq2parse.py python-grammar.txt input_prog.py repairs/fix_0.py test-set-top-20-partials-probs.txt 20
    grammarFile = sys.argv[1]
    inputPath = Path(sys.argv[2])

    max_cost = 5
    input_prog = inputPath.read_text()
    # print('*' * 42)
    # print(input_prog)
    # print('*' * 42)
    ERROR_GRAMMAR = read_grammar(grammarFile)
    terminals = ERROR_GRAMMAR.get_alphabet()

    prog_tokens = get_token_list(input_prog, terminals)
    error_rules = ['Err_Close_Paren -> H Close_Paren', 'Err_Close_Sq_Bracket -> H Close_Sq_Bracket', 'Err_Colon -> H Colon', 'Err_Comma -> H Comma', 'Err_Comp_Op -> H Comp_Op', 'Err_Literals -> ', 'Err_Literals -> Err_Tag', 'Err_Literals -> H Literals', 'Err_MulDiv_Op -> H MulDiv_Op', 'Err_Newline -> H Newline', 'Err_Open_Paren -> ', 'Err_Return_Keyword -> ', 'InsertErr -> %', 'InsertErr -> )', 'InsertErr -> *', 'InsertErr -> +', 'InsertErr -> ,', 'InsertErr -> -', 'InsertErr -> :', 'InsertErr -> =']
    actual_tokens = get_actual_token_list(input_prog, terminals)

    repaired_prog = has_parse(ERROR_GRAMMAR, max_cost, prog_tokens, error_rules, actual_tokens)

    result = { "status": "safe"
             , "errors": []
             , "types": {}
             }

    if repaired_prog:
        result["status"] = "error"
        result["errors"] = [{ "message": "repaired_prog"
                            , "start"  : {"line": 1, "column": 1}
                            , "stop"   : {"line": 1, "column": 10}
                            }]

    with open(inputPath.with_suffix(".json"), "w") as out_file:
        json.dump(result, out_file, indent=4)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("-------------Original Buggy Program---------------")
    print(input_prog)
    print("-----------------Repaired Program-----------------")
    print(repaired_prog[:-3].replace("\\n", '\n'))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
