import sys
import re
from functools import reduce
from collections import Counter
from os.path import join
from pathlib import Path
import difflib as df
import ast
from ast import parse, dump
from random import sample
import json
import tqdm
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Text, Operator, Name, Keyword
from repair_tokenization_unbalanced_curly import repair_unbalanced, reconstruct_and_tokenize
from comment_remover import removeComments
from tokenize_reference import tokenize_ref
from ecpp_individual_grammar import Lexer

# pr = {"bad": "def check_rush_hour(day, hour, minute):\n    if day== 'Monday' or day=='Tuesday' or day=='Wednesday' or day=='Thursday' or day=='Friday':\n        if hour>= 5:\n            if hour<= 9:\n                if minute= 0 or minute=1 or minute=2 or minute=3 or minute=4 or minute=5 or minute=6:\n                    return True\n                else:\n                    return False\n        if hour>= 15:\n            if hour<= 19:\n                elif minute= 0:\n                    return True\n                else:\n                    return False\n    else:\n        return False\n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "fix": "def check_rush_hour(day, hour, minute):\n    if day== \"Monday\" or day==\"Tuesday\" or day==\"Wednesday\" or day==\"Thursday\" or day==\"Friday\":\n        if hour>= 5:\n            if hour<= 9:\n                if minute >=0:\n                    if minute<=30:\n                        return True\n                    else:\n                        return False\n        elif hour>= 15:\n            if hour<= 19:\n               return True\n            else:\n                return False\n    else:\n        return False\n        \n    \n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "index": 4818, "fixIndex": 4830, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 1205}

# pr = {"bad": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print i\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "fix": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print (i)\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "index": 6, "fixIndex": 7, "errMsg": "SyntaxError", "isConsecutive": True, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 15}

# pr = {"bad": "def place_one(board,c,color):\n    for index in range(len(board)):\n        if (board[index][c]=='.'):\n    \t\tboard[index][c]==color\n    \t\tbreak\n\treturn True \n\t\nplace_one([['.','.','.','.','B','.','.'],\n\t\t\t['.','.','.','.','B','B','.'],\n\t\t\t['.','.','.','G','G','G','.'],\n\t\t\t['.','.','.','G','G','B','.'],\n\t\t\t['.','G','.','G','B','B','.'],\n\t\t\t['.','B','.','G','B','G','B']],4,'M')", "fix": "def place_one(board,c,color):\n    for index in range(len(board)):\n        if (board[index][c]=='.'):\n    \t    board[index][c]==color\n    \t    break\n    return True \n\t\nplace_one([['.','.','.','.','B','.','.'],\n\t\t\t['.','.','.','.','B','B','.'],\n\t\t\t['.','.','.','G','G','G','.'],\n\t\t\t['.','.','.','G','G','B','.'],\n\t\t\t['.','G','.','G','B','B','.'],\n\t\t\t['.','B','.','G','B','G','B']],4,'M')", "index": 495, "fixIndex": 497, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 35}


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
    return [(ch_type, line, prev_line) for ch_type, k, line, prev_line in line_changes if ch_type != 'no_change']


AUG_ASSIGN = ['+=', '-=', '*=', '@=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//=']
COMP_OP = ['==', '>=', '<=', '<>', '!=', 'not_in', 'in', 'is_not', 'is', '<', '>']
BINARY_OP = ['|', '^', '&', '>>', '<<', '~']
ARITH_OP = ['+', '-', '//', '/', '%', '**', '*', '@']
BOOLEAN_OP = ['not', 'and', 'or']

def return_generic_token(lxd):
    lxd_upd = lxd.replace('>>>', '__prob__')
    for value in AUG_ASSIGN:
        lxd_upd = lxd_upd.replace(value, '__aug_assign__')
    lxd_upd = lxd_upd.replace('>>', '__right_right__')
    lxd_upd = lxd_upd.replace('<<', '__left_left__')
    lxd_upd = lxd_upd.replace('finally', '__temp1__')
    lxd_upd = lxd_upd.replace('continue', '__temp2__')
    for value in COMP_OP:
        lxd_upd = lxd_upd.replace(value, '__comp_op__')
    lxd_upd = lxd_upd.replace('__temp2__', 'continue')
    lxd_upd = lxd_upd.replace('__temp1__', 'finally')
    lxd_upd = lxd_upd.replace('__right_right__', '__binary_op__')
    lxd_upd = lxd_upd.replace('__left_left__', '__binary_op__')
    for value in BINARY_OP:
        lxd_upd = lxd_upd.replace(value, '__binary_op__')
    for value in ARITH_OP:
        lxd_upd = lxd_upd.replace(value, '__arith_op__')
    lxd_upd = lxd_upd.replace('for', '__temp1__')
    lxd_upd = lxd_upd.replace('import', '__temp2__')
    for value in BOOLEAN_OP:
        lxd_upd = lxd_upd.replace(value, '__boolean_op__')
    lxd_upd = lxd_upd.replace('__temp2__', 'import')
    lxd_upd = lxd_upd.replace('__temp1__', 'for')
    lxd_upd = lxd_upd.replace('.', '__dot_op__')
    lxd_upd = lxd_upd.replace('=', '__assign_op__')
    return lxd_upd


def clean_repeats(token):
    new_token = re.sub(r'(_INDENT_ )+', "", token)
    new_token = re.sub(r'(_DEDENT_ )+', "", new_token)
    return new_token


def extract_change(prev_line, new_line):
    prev_tokens = prev_line.split()
    new_tokens = new_line.split()
    # print(prev_tokens)
    # print(new_tokens)
    try:
        diff = list(df.ndiff(prev_tokens, new_tokens))
    except RecursionError:
        return []
    # print("\n".join(diff))
    changes = []
    deleted = ""
    added = ""
    replacement = False
    for change in diff:
        if change[0] == '?':
            continue
        elif change[0] == '-':
            if added != "":
                added = clean_repeats(added.strip())
                if replacement:
                    prev_change = changes.pop()[1]
                    changes.append(('replaced', prev_change, added))
                    replacement = False
                else:
                    changes.append(('added', None, added))
                    replacement = True
                added = ""
            deleted += change[1:]
        elif change[0] == '+':
            if deleted != "":
                deleted = clean_repeats(deleted.strip())
                if replacement:
                    prev_change = changes.pop()[2]
                    changes.append(('replaced', deleted, prev_change))
                    replacement = False
                else:
                    changes.append(('deleted', deleted, None))
                    replacement = True
                deleted = ""
            added += change[1:]
        else:
            if deleted != "":
                deleted = clean_repeats(deleted.strip())
                if replacement:
                    prev_change = changes.pop()[2]
                    changes.append(('replaced', deleted, prev_change))
                else:
                    changes.append(('deleted', deleted, None))
                    replacement = True
                deleted = ""
            if added != "":
                added = clean_repeats(added.strip())
                if replacement:
                    prev_change = changes.pop()[1]
                    changes.append(('replaced', prev_change, added))
                else:
                    changes.append(('added', None, added))
                    replacement = True
                added = ""
            replacement = False
    if deleted != "":
        deleted = clean_repeats(deleted.strip())
        if replacement:
            prev_change = changes.pop()[2]
            changes.append(('replaced', deleted, prev_change))
            replacement = False
        else:
            changes.append(('deleted', deleted, None))
            replacement = True
        deleted = ""
    if added != "":
        added = clean_repeats(added.strip())
        if replacement:
            prev_change = changes.pop()[1]
            changes.append(('replaced', prev_change, added))
            replacement = False
        else:
            changes.append(('added', None, added))
            replacement = True
        added = ""
    # print(changes)
    new_changes = []
    for ch in changes:
        if ch[0] == 'deleted':
            for one_tok in ch[1].split():
                st_idx = 0
                tok_idx = prev_line.find(one_tok, st_idx)
                if tok_idx < 0:
                    new_changes.append(('deleted', one_tok, '_NEWLINE_'))
                while tok_idx >= 0:
                    next_tok = None
                    next_tok_idx = prev_line.find(' ', tok_idx) + 1
                    if next_tok_idx < 1:
                        next_tok = '_NEWLINE_'
                    else:
                        next_tok_idx_end = prev_line.find(' ', next_tok_idx)
                        if next_tok_idx_end < 0:
                            next_tok = prev_line[next_tok_idx:]
                        else:
                            next_tok = prev_line[next_tok_idx:next_tok_idx_end]
                    new_changes.append(('deleted', one_tok, next_tok))
                    if next_tok_idx < 1:
                        break
                    st_idx = next_tok_idx
                    prev_tok_idx = tok_idx
                    tok_idx = prev_line.find(one_tok, st_idx)
                    if tok_idx == prev_tok_idx:
                        print("A BIT WEIRD!!!")
                        print(prev_line)
                        print(ch)
                        break
        else:
            new_changes.append(ch)
    changes = new_changes

    ## DONT FORGET TO CHANGE FOR REGULAR DISTRIBUTION RUN
    return changes


def return_changes(l, terminals):
    dct = json.loads(l)
    if dct["errMsg"] == "SyntaxError":
        lexer = Lexer(terminals)

        bad = return_generic_token(lexer.lex(dct['bad'])).split('_NEWLINE_')
        fix = return_generic_token(lexer.lex(dct['fix'])).split('_NEWLINE_')
        diff = list(df.ndiff(bad, fix))
        # print("\n".join(diff))
        # print('------------------------------------')

        line_changes = get_changes(diff)
        # print(line_changes)
        changes = []
        for line_ch in line_changes:
            if line_ch[0] == 'replaced':
                # These are changes within a line
                changes.extend(extract_change(line_ch[2], line_ch[1]))
            else:
                # This are whole line changes (additions/deletions)
                changes.append((line_ch[0], line_ch[2], line_ch[1]))
        return changes


def return_all_changes(l, terminals):
    dct = json.loads(l)
    if dct["errMsg"] == "SyntaxError":
        lexer = Lexer(terminals)

        # print('# --------------------------')
        # print(dct['bad'])
        # print('# --------------------------')
        # print(dct['fix'])
        bad = lexer.lex(dct['bad']).split('_NEWLINE_')
        fix = lexer.lex(dct['fix']).split('_NEWLINE_')
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
                changes.extend(extract_change(line_ch[2], line_ch[1]))
            elif line_ch[0] == 'deleted':
                # This are whole line changes (deletions)
                toks = line_ch[2].split()
                next_toks = toks[1:] + ['_NEWLINE_']
                for tok, ntok in zip(toks, next_toks):
                    changes.append(('deleted', tok, ntok))
            else:
                # This are whole line changes (additions)
                changes.append((line_ch[0], line_ch[2], line_ch[1]))
        return changes


def return_changes_one_pair(bad_p, fix_p, terminals):
    lexer = Lexer(terminals)

    bad = return_generic_token(lexer.lex(bad_p)).split('_NEWLINE_')
    fix = return_generic_token(lexer.lex(fix_p)).split('_NEWLINE_')
    diff = list(df.ndiff(bad, fix))
    # print("\n".join(diff))
    # print('------------------------------------')

    line_changes = get_changes(diff)
    # print(line_changes)
    changes = []
    for line_ch in line_changes:
        if line_ch[0] == 'replaced':
            # These are changes within a line
            changes.extend(extract_change(line_ch[2], line_ch[1]))
        else:
            # This are whole line changes (additions/deletions)
            changes.append((line_ch[0], line_ch[2], line_ch[1]))
    return changes


if __name__ == "__main__":
    dataDir = Path(sys.argv[1])
    outDir = Path(sys.argv[2])

    pairs = []
    for partPath in tqdm.tqdm(list(dataDir.glob('part_*'))):
        goodPath = partPath / "goodPairs.jsonl"
        for line in tqdm.tqdm(goodPath.read_text().strip().split('\n')):
            pair = return_changes(line, [])
            if pair:
                pairs.extend(pair)
        failPath = partPath / "failPairs.jsonl"
        for line in tqdm.tqdm(failPath.read_text().strip().split('\n')):
            pair = return_changes(line, [])
            if pair:
                pairs.extend(pair)
    total_pairs = len(pairs)
    pairs = sorted(list(filter(lambda y: y[1] > 100, Counter(pairs).items())), key=lambda x: x[1], reverse=True)
    total_useful_pairs = sum([p[1] for p in pairs])
    with open(join(outDir, "TokenChanges.txt"), "w") as dataset_file:
        print("Dataset size:", len(pairs))
        for pair in pairs:
            first_tab = ":\t" if pair[0][0] in ['replaced', 'deleted'] else ":\t\t"
            second_tab = "\t->\t\t" if len(str(pair[0][1])) < 3 else "\t->\t"
            first_part = pair[0][0] + first_tab + str(pair[0][1]) + second_tab + str(pair[0][2]) + "\t\t"
            third_space = 50 // 4 - len(str(first_part)) // 4
            third_tab = "\t" * (third_space if third_space > 0 else 1)
            stri = pair[0][0] + first_tab + str(pair[0][1]) + second_tab + str(pair[0][2]) + third_tab + "Total = " + str(pair[1]) + ",\t% = " + str(pair[1] * 100 / total_useful_pairs) + "\tOrig % =" + str(pair[1] * 100 / total_pairs) + "\n"
            dataset_file.write(stri)
    # print("\n".join(map(str, return_all_changes(json.dumps(pr)))))
