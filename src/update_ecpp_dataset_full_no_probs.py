import resource
import sys
from os.path import exists, join
from pathlib import Path

import tqdm

import earleyparser_interm_repr
from ecpp_individual_grammar import get_token_list, read_grammar


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

def store_dataset(erule_dataset, out_file):
    for tokns, upd_tkns, eruls, tok_chgs, dur, next_tkn, fix_tkns, origb, origf, orig_tkns in erule_dataset:
        out_file.write(tokns + " <||> " + upd_tkns + " <||> " + " <++> ".join(eruls) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + next_tkn + " <||> " + fix_tkns + " <||> " + repr(origb) + " <||> " + repr(origf) + " <||> " + orig_tkns + "\n")

def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[2].split(" <++> ")
    if len(samp_1) < 9:
        return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]], samp_1[6], samp_1[7])
    else:
        return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]], samp_1[6], samp_1[7], samp_1[8], samp_1[9])


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    limit_memory()

    ERROR_GRAMMAR = read_grammar(grammar_file)
    terminals = ERROR_GRAMMAR.get_alphabet()
    INTERIM_GRAMMAR = earleyparser_interm_repr.read_grammar(grammar_file)
    # We use this script to remove the effect of the PCFG in the abstraction
    # rules_used = {}
    # with open("rules_usage.json", "r") as in_file:
    #     rules_used = json.load(in_file)
    # INTERIM_GRAMMAR.update_probs(rules_used)
    for partPath in list(dataDir.glob('part_*')):
        print("#", partPath.name)
        if partPath.name not in ['part_2018_1', 'part_2018_2', 'part_2018_3', 'part_2018_4']:
            continue
        dataset = []
        dataset_part_file = join(partPath, "erule-dataset-partials-probs-fixes.txt")
        if not exists(dataset_part_file):
            continue
        else:
            with open(dataset_part_file, "r") as inFile:
                old_dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
                if len(old_dataset[0]) < 10:
                    print(f">>> WRONG NUMBER OF DATA IN {partPath.name}!!!")
                for old_results in tqdm.tqdm(old_dataset):
                    orig_bad = old_results[7].replace("\\n", '\n').replace("\\r", '\r').replace("\\t", '\t')[1:-1]
                    tokens = get_token_list(orig_bad, terminals)
                    old_results_temp = list(old_results)
                    old_results_temp[1], old_results_temp[5] = earleyparser_interm_repr.get_updated_seq(tokens, INTERIM_GRAMMAR)
                    if old_results_temp[5] is None:
                        dataset.append(old_results)
                    else:
                        dataset.append(tuple(old_results_temp))
            with open(join(partPath, "erule-dataset-partials-no-probs-fixes.txt"), "w") as outFile:
                store_dataset(dataset, outFile)
