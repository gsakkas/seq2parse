import json
from pathlib import Path
from os.path import join, exists
from os import mkdir
import autopep8
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list, get_actual_token_list

file_path = Path('./human_study/repaired_prog_pairs.jsonl')
out_dir = Path('./human_study')

i = 0
for line in file_path.read_text().strip().split('\n'):
    if line == "null":
        continue
    i += 1
    dct = json.loads(line)
    if not exists(join(out_dir, "program_" + str(i))):
        mkdir(join(out_dir, "program_" + str(i)))
    with open(join(out_dir, "program_" + str(i), "original.py"), "w") as fil:
        temp_prog = dct['original'][1:-1].replace('\\\'','\'').replace("\\\\n", '<--temp-->')
        fil.write(temp_prog.replace("\\n", '\n').replace('<--temp-->', "\\n"))
    with open(join(out_dir, "program_" + str(i), "eccp_repair.py"), "w") as fil:
        fil.write(autopep8.fix_code(dct['repair'][0][:-3]))
    with open(join(out_dir, "program_" + str(i), "user_fix.py"), "w") as fil:
        temp_prog = dct['user_fix'][1:-1].replace('\\\'','\'').replace("\\\\n", '<--temp-->')
        fil.write(autopep8.fix_code(temp_prog.replace("\\n", '\n').replace('<--temp-->', "\\n")))
    with open(join(out_dir, "program_" + str(i), "user_repair_time.txt"), "w") as fil:
        fil.write(str(dct['user_time']) + '\n')
