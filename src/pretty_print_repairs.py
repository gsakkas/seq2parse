import json
from pathlib import Path
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list, get_actual_token_list

file_path = Path('./repaired_prog_pairs.jsonl')

for line in file_path.read_text().strip().split('\n'):
    if line == "null":
        continue
    dct = json.loads(line)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("-------------Original Buggy Program---------------")
    temp_prog = dct['orig'][1:-1].replace('\\\'','\'').replace("\\\\n", '<--temp-->')
    print(temp_prog.replace("\\n", '\n').replace('<--temp-->', "\\n"))
    print("-----------------Repaired Program-----------------")
    print(dct['repaired'][:-3]) # -3 to delete some extra newlines at the end
    print("--------------Original Fix Program----------------")
    temp_prog = dct['fix'][1:-1].replace('\\\'','\'').replace("\\\\n", '<--temp-->')
    print(temp_prog.replace("\\n", '\n').replace('<--temp-->', "\\n"))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
