import sys
import resource
from os.path import join, exists
from pathlib import Path
from functools import partial
from copy import deepcopy
import json
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
# import signal
# from contextlib import contextmanager
import tqdm
import earleyparser_interm_repr
from ecpp_individual_grammar import read_grammar, prog_has_parse, prog_error_rules, get_token_list, get_actual_token_list
import earleyparser
from run_extract_token_distribution import return_all_changes

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


def has_parse(grammar, egrammar, tup):
    i, l = tup
    dct = json.loads(l)
    if dct["index"] in [466, 467, 468] and dct["fixIndex"] == 470 and dct["duration"] in [362, 375, 380]:
        return None
    elif dct["errMsg"] == "SyntaxError":
        if not earleyparser.prog_has_parse(dct['bad'], grammar):
            upd_grammar = deepcopy(egrammar)
            terminals = egrammar.get_alphabet()
            token_changes = return_all_changes(l, terminals)
            if not token_changes:
                return None
            num_of_changes = sum([len(ch[2].split()) if ch[0] == 'added' else len(ch[1].split()) for ch in token_changes])
            if num_of_changes > 40:
                return None
            error_rules = upd_grammar.update_error_grammar(token_changes)
            tokens = get_token_list(dct['bad'], terminals)
            fixed_tokens = get_token_list(dct['fix'], terminals)
            orig_tokens = get_actual_token_list(dct['bad'], terminals)
            if error_rules == []:
                bparse = False
            else:
                bparse = True
            return (num_of_changes, bparse, error_rules, tokens, fixed_tokens, dct['bad'], dct['fix'], dct["duration"], orig_tokens)
        return (-1, False, [], "", "", "", "", -1, "")


def store_dataset(tkns, upd_tkns, erules, tok_chgs, dur, next_tkn, fix_tkns, origb, origf, orig_tokens, out_file):
    out_file.write(tkns + " <||> " + upd_tkns + " <||> " + " <++> ".join(erules) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + next_tkn + " <||> " + fix_tkns + " <||> " + repr(origb) + " <||> " + repr(origf) + " <||> " + orig_tokens + "\n")


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[2].split(" <++> ")
    return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]])


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    GRAMMAR = earleyparser.read_grammar(grammar_file)
    for partPath in list(dataDir.glob('part_*')):
        print("#", partPath.name)
        if partPath.name not in ['part_2018_6']:
            continue
        goodPath = partPath / "goodPairs.jsonl"
        failPath = partPath / "failPairs.jsonl"
        new_dataset = []
        with ProcessPool(max_workers=8, max_tasks=5) as pool:
            programs = enumerate(goodPath.read_text().strip().split('\n'))
            new_has_parse = partial(has_parse, GRAMMAR, ERROR_GRAMMAR)
            future = pool.map(new_has_parse, programs, chunksize=10)
            it = future.result()
            while True:
                try:
                    bruh = next(it)
                    if bruh:
                        tok_chs, parse_bad, erules, lexed_prog, lexed_fixed_prog, orig_bad, orig_fix, duration, orig_tokens = bruh
                        if tok_chs > 0:
                            if parse_bad:
                                new_dataset.append((lexed_prog, list(map(str, erules)), tok_chs, lexed_fixed_prog, duration, orig_bad, orig_fix, orig_tokens))
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    print("Timeout")
                except Exception as e:
                    print("WHY here?!", str(e))
            programs = enumerate(failPath.read_text().strip().split('\n'))
            future = pool.map(new_has_parse, programs, chunksize=10)
            it = future.result()
            while True:
                try:
                    bruh = next(it)
                    if bruh:
                        tok_chs, parse_bad, erules, lexed_prog, lexed_fixed_prog, orig_bad, orig_fix, duration, orig_tokens = bruh
                        if tok_chs > 0:
                            if parse_bad:
                                new_dataset.append((lexed_prog, list(map(str, erules)), tok_chs, lexed_fixed_prog, duration, orig_bad, orig_fix, orig_tokens))
                except StopIteration:
                    break
                except (TimeoutError, ProcessExpired):
                    print("Timeout")
                except Exception as e:
                    print("WHY here?!", str(e))

        dataset = []
        dataset_part_file = join(partPath, "erule-dataset-partials-probs.txt")
        if not exists(dataset_part_file):
            continue
        else:
            with open(dataset_part_file, "r") as inFile:
                dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
            with open(join(partPath, "erule-dataset-partials-probs-fixes.txt"), "w") as outFile:
                for old_tokns, tokns, eruls, tok_changes, duration, next_tokns, in tqdm.tqdm(dataset):
                    for idx, (lexed_prog, all_erules, tok_chs, lexed_fixed_prog, dur, orig_bad, orig_fix, orig_tokens) in enumerate(new_dataset):
                        if old_tokns == lexed_prog and duration == dur and all(map(lambda r: r in all_erules, eruls)):
                            # upd_tokns, next_token = earleyparser_interm_repr.get_updated_seq(tokns, GRAMMAR)
                            store_dataset(old_tokns, tokns, eruls, tok_changes, duration, "".join(next_tokns), lexed_fixed_prog, orig_bad, orig_fix, orig_tokens, outFile)
                            del new_dataset[idx]
                            break
