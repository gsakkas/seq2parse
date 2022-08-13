import sys
from os import mkdir
from os.path import join, exists
from pathlib import Path
import json
from statistics import mean, median, stdev
from tokenize import TokenError
import difflib as df
import asttokens as toks
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import re
from ecpp_individual_grammar_all_states import read_grammar, get_token_list, get_actual_token_list


def cum_dist_fun(secs, durs):
    in_set = list(filter(lambda x: x <= secs, durs))
    return len(in_set) * 100.0 / len(durs)


def dist_fun(secs, durs, step):
    in_set = list(filter(lambda x: secs - step < x <= secs, durs))
    return len(in_set) * 100.0 / len(durs)


def get_changes(diff):
    tchanges = []
    for i, change in enumerate(diff):
        line = change[2:]
        if change[0] == '-':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '+' and tchanges != [] and tchanges[-1][0] == 'added':
                    tchanges.pop()
                    tchanges.append('replaced')
                else:
                    tchanges.append('deleted')
            elif i-1 >= 0 and diff[i-1][0] == '+' and tchanges != [] and tchanges[-1][0] == 'added':
                tchanges.pop()
                tchanges.append('replaced')
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                tchanges.append('deleted')
        elif change[0] == '+':
            if i-1 >= 0 and diff[i-1][0] == '?':
                if i-2 >= 0 and diff[i-2][0] == '-' and tchanges != [] and tchanges[-1][0] == 'deleted':
                    tchanges.pop()
                    tchanges.append('replaced')
                else:
                    tchanges.append('added')
            elif i-1 >= 0 and diff[i-1][0] == '-' and tchanges != [] and tchanges[-1][0] == 'deleted':
                tchanges.pop()
                tchanges.append('replaced')
            elif len(re.sub(r"[\n\t\s]*", "", line)) > 0:
                tchanges.append('added')
        elif change[0] == ' ':
            if change[2:].strip() == '':
                continue
            tchanges.append('no_change')
    return tchanges


def get_diff(bad, fix):
    try:
        bad_tok_types = get_token_list(bad, terminals).split()
        fix_tok_types = get_token_list(fix, terminals).split()

        bad_tok_strs = get_actual_token_list(bad, terminals).split()
        fix_tok_strs = get_actual_token_list(fix, terminals).split()

        type_diff = len(list(filter(lambda d: d != 'no_change', get_changes(list(df.ndiff(bad_tok_types, fix_tok_types))))))
        str_diff = len(list(filter(lambda d: d != 'no_change', get_changes(list(df.ndiff(bad_tok_strs, fix_tok_strs))))))

        return (type_diff, str_diff)
    except TokenError as err:
        raise err


MAX_HOURS = 1
MAX_SECS = MAX_HOURS * 60 * 60 // 2
if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])
    errs = dict()
    durations = dict()
    type_diffs = dict()
    str_diffs = dict()

    errs["AllErrors"] = 0
    durations["AllErrors"] = []
    type_diffs["AllErrors"] = []
    str_diffs["AllErrors"] = []

    errs["RuntimeErrors"] = 0
    durations["RuntimeErrors"] = []
    type_diffs["RuntimeErrors"] = []
    str_diffs["RuntimeErrors"] = []

    type_diffs["SyntaxError"] = []
    str_diffs["SyntaxError"] = []

    uniqFixes = set()
    errs["UniqueFixes"] = 0

    ERROR_GRAMMAR = read_grammar(grammarFile)
    terminals = ERROR_GRAMMAR.get_alphabet()

    STEP = 20
    dur_groups = [([], [], [], (time-STEP, time)) for time in range(STEP, 301, STEP)]
    change_size_groups = [([], [], (size-1, size)) for size in range(1, 11)]

    def gather_data(l):
        dct = json.loads(l)
        duration = int(dct["duration"])
        if 0 < duration <= MAX_SECS:
            badMsg = dct["errMsg"]
            errs["AllErrors"] += 1
            if badMsg in errs:
                errs[badMsg] += 1
            else:
                errs[badMsg] = 1
            durations["AllErrors"].append(duration)
            if badMsg in durations:
                durations[badMsg].append(duration)
            else:
                durations[badMsg] = [duration]
            uniqFixes.add(str(dct["fixIndex"]) + dct["fix"])
            try:
                tdiff, sdiff = get_diff(dct["bad"], dct["fix"])
                type_diffs["AllErrors"].append(tdiff)
                str_diffs["AllErrors"].append(sdiff)
                if badMsg == "SyntaxError":
                    type_diffs["SyntaxError"].append(tdiff)
                    str_diffs["SyntaxError"].append(sdiff)
                else:
                    type_diffs["RuntimeErrors"].append(tdiff)
                    str_diffs["RuntimeErrors"].append(sdiff)
                if badMsg == "SyntaxError":
                    for i, (examples, tdiffs, sdiffs, span) in enumerate(dur_groups):
                        if span[0] < duration <= span[1]:
                            if tdiff + sdiff > 2.0 and len(examples) < 20:
                                examples.append(dct)
                            tdiffs.append(tdiff)
                            sdiffs.append(sdiff)
                            dur_groups[i] = (examples, tdiffs, sdiffs, span)
                            break
                    for i, (examples, utimes, span) in enumerate(change_size_groups):
                        if tdiff > 10 and span[1] > 10:
                            if len(examples) < 20:
                                examples.append(dct)
                            utimes.append(duration)
                            change_size_groups[i] = (examples, utimes, span)
                            break
                        if span[0] < tdiff <= span[1]:
                            if len(examples) < 20:
                                examples.append(dct)
                            utimes.append(duration)
                            change_size_groups[i] = (examples, utimes, span)
                            break
            except:
                pass

    for partPath in tqdm.tqdm(list(dataDir.glob('data_*/part_*'))):
        goodPath = partPath / "goodPairs.jsonl"
        for line in tqdm.tqdm(goodPath.read_text().strip().split('\n')):
            gather_data(line)
        failPath = partPath / "failPairs.jsonl"
        for line in tqdm.tqdm(failPath.read_text().strip().split('\n')):
            gather_data(line)
    errs["UniqueFixes"] = len(uniqFixes)
    for k in durations:
        if k not in ["SyntaxError", "UniqueFixes", "AllErrors"]:
            all_durs = durations[k]
            durations["RuntimeErrors"].extend(all_durs)
    for k in errs:
        if k not in ["SyntaxError", "UniqueFixes", "AllErrors"]:
            errs["RuntimeErrors"] += errs[k]
    print("0 <", len(type_diffs["AllErrors"]), "<=", errs["AllErrors"])
    print("0 <", len(type_diffs["SyntaxError"]), "<=", errs["SyntaxError"])
    # Plot CDF of fix durations of the dataset
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=34)
    plt.rc('legend', fontsize=24)

    fig, ax = plt.subplots(figsize=(12, 8))
    xs = list(range(0, MAX_SECS // 2 + 1, 10))
    # total_ys = list(map(lambda x: cum_dist_fun(x, durations["AllErrors"]), xs))
    parse_ys = list(map(lambda x: cum_dist_fun(x, durations["SyntaxError"]), xs))
    runtime_ys = list(map(lambda x: cum_dist_fun(x, durations["RuntimeErrors"]), xs))
    parse_times = {}
    for x, y in zip(xs, parse_ys):
        if x <= 240:
            parse_times[x] = y
    runtime_times = {}
    for x, y in zip(xs, runtime_ys):
        if x <= 240:
            runtime_times[x] = y

    # ax.step(xs, total_ys, where='mid', linewidth=3.0, label='All Errors')
    ax.line(xs, parse_ys, where='mid', linewidth=3.0, label='Parse Errors')
    ax.line(xs, runtime_ys, where='mid', linewidth=3.0, label='Runtime Errors')

    ax.set_xlim([0, MAX_SECS // 2])
    ax.set_ylim([0, 105])

    ax.grid(True, which='both')
    ax.legend(loc='upper left')
    ax.set_xlabel('User Fix Duration (sec)')
    ax.set_ylabel('Fixed Dataset Rate (\%)')

    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "fixed-rate.png"))
    plt.savefig(join(outDir, "fixed-rate.pdf"))

    plt.rc('font', family='serif', size=28)
    plt.rc('legend', fontsize=20)

    # Plot DF of fix durations of the dataset
    fig, ax = plt.subplots(figsize=(12, 9))
    labels = list(range(0, MAX_SECS // 6 + 1, 10))
    parse_ys = list(map(lambda x: dist_fun(x, durations["SyntaxError"], 10), labels))
    runtime_ys = list(map(lambda x: dist_fun(x, durations["RuntimeErrors"], 10), labels))
    width = 0.4
    xs1 = np.arange(len(labels))
    xs2 = [x + width for x in xs1]
    xs = [x + width / 2 for x in xs1]

    ax.bar(xs1, parse_ys, width, label='Parse Errors', zorder=3)
    ax.bar(xs2, runtime_ys, width, label='Runtime Errors', zorder=3)

    maxx_percent = ((max(max(parse_ys), max(runtime_ys)) + 5) // 5) * 5
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, maxx_percent])

    ax.grid(True, which='both', zorder=0)
    ax.legend(loc='upper left')
    ax.set_xlabel('User Fix Duration (sec)')
    ax.set_ylabel('Fixed Dataset Rate (\%)')

    plt.grid(True, which='minor', linestyle='--')
    plt.xticks(rotation=60)
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "dist-fixed-rate.png"))

    # Plot Pie chart of error types
    percentages = []
    labels = []
    explode = []
    rest = 0.0
    for k in errs:
        if k not in ["UniqueFixes", "AllErrors", "RuntimeErrors"]:
            percentage = errs[k] / errs["AllErrors"]
            if percentage < 0.01:
                rest += percentage
            else:
                percentages.append(percentage)
                labels.append(k)
                explode.append(0.05 if k == "SyntaxError" else 0.0)
    if rest > 0.0:
        percentages.append(rest)
        labels.append("Other Errors")
        explode.append(0.0)
    sum_perc = sum(percentages)
    if sum_perc < 1 - 1e-6 or sum_perc > 1 + 1e-6:
        print("Something went wrong with percentages:", sum_perc, 1 - 10^(-6))
    percentages, labels, explode = zip(*sorted(zip(percentages, labels, explode), key=lambda t: t[0],reverse=True))

    fig, ax = plt.subplots(figsize=(12, 8))
    # new_labels = list(labels)[:-2] + [""]*2
    wedges, texts, autotexts = ax.pie(percentages, explode=explode, autopct='%.1f\%%', pctdistance=0.75, labeldistance=1.05, labels=labels, startangle=0, normalize=True)
    # ax.legend(wedges, ['{0}: \\bf {1:1.2f}\%'.format(i,j*100) for i,j in zip(labels, percentages)], loc='upper left')
    ax.axis('equal')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "error-pie.png"))
    plt.savefig(join(outDir, "error-pie.pdf"))

    # Plot average and median number of changes per duration span
    labels = list(map(lambda d: str(d[-1][1]), dur_groups))
    tavgs = list(map(lambda d: mean(d[1]), dur_groups))
    tmedians = list(map(lambda d: median(d[1]), dur_groups))
    st_medians = list(map(lambda d: mean([median(d[1]), median(d[2])]), dur_groups))
    savgs = list(map(lambda d: mean(d[2]), dur_groups))
    smedians = list(map(lambda d: median(d[2]), dur_groups))
    width = 0.2
    xs1 = [x + width / 2 for x in np.arange(len(labels))]
    xs2 = [x + width for x in xs1]
    xs3 = [x + width + width / 2 for x in xs2]
    xs4 = [x + width for x in xs3]
    xs = [x + 2 * width - width / 4 for x in xs1]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(xs1, tavgs, width, label='Type Average', zorder=3)
    ax.bar(xs2, tmedians, width, label='Type Median', zorder=3)
    ax.plot(xs, st_medians, color='black', linewidth='2', linestyle='dashed', marker='o', markersize='5', zorder=3, label='Type+String Median')
    ax.bar(xs3, savgs, width, label='String Average', zorder=3)
    ax.bar(xs4, smedians, width, label='String Median', zorder=3)

    ax.grid(zorder=0)
    ax.set_xlabel('User Fix Duration (sec)')
    ax.set_ylabel('Token Changes (\#)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "token-changes.png"))

    # Plot only median number of changes per duration span
    width = 0.36
    xs1 = np.arange(len(labels))
    xs2 = [x + width for x in xs1]
    xs = [x + width / 2 for x in xs1]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(xs1, tmedians, width, label='Type Median', zorder=3)
    ax.plot(xs, st_medians, color='black', linewidth='2.5', linestyle='dashed', marker='o', markersize='6', zorder=3, label='Type+String Median')
    ax.bar(xs2, smedians, width, label='String Median', zorder=3)

    ax.grid(zorder=0)
    ax.set_xlabel('User Fix Duration (sec)')
    ax.set_ylabel('Token Changes (\#)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    plt.minorticks_on()
    plt.grid(True, which='major', linewidth=1.5)
    plt.grid(True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "median-token-changes.png"))

    # Plot only type median number of changes per duration span
    labels = list(map(lambda d: str(d[-1][1]), change_size_groups))
    labels[-1] = labels[-1] + "+"
    dtmedians = list(map(lambda d: median(d[1]), change_size_groups))
    width = 0.42
    xs = np.arange(len(labels))
    fix_durs = {}
    for l, dtm in zip(labels, dtmedians):
        fix_durs[l] = dtm

    plt.rc('font', family='serif', size=34)
    plt.rc('legend', fontsize=24)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(xs, dtmedians, width, zorder=3)
    # ax.plot(xs, dtmedians, color='black', linewidth='2.5', linestyle='dashed', marker='o', markersize='6', zorder=3)

    ax.grid(zorder=0)
    ax.set_xlabel('Token Changes (\#)')
    ax.set_ylabel('User Fix Duration (sec)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    # plt.minorticks_on()
    plt.grid(True, which='major', linewidth=1.5)
    # plt.grid(True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "median-repair-times.png"))
    plt.savefig(join(outDir, "median-repair-times.pdf"))

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = list(map(lambda d: str(d[-1][1]), change_size_groups))
    labels[-1] = labels[-1] + "+"
    ys = list(map(lambda d: len(d[1]) * 100.0 / errs["SyntaxError"], change_size_groups))
    width = 0.42
    xs = np.arange(len(labels))
    fix_ratios = {}
    for l, rat in zip(labels, ys):
        fix_ratios[l] = rat

    ax.bar(xs, ys, width, zorder=3)

    ax.grid(zorder=0)
    ax.set_xlabel('Token Changes (\#)')
    ax.set_ylabel('Fixed Dataset Ratio (\%)')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max(ys) + 5])

    # plt.minorticks_on()
    plt.grid(True, which='major', linewidth=1.5)
    # plt.grid(True, which='minor', linestyle='--')
    plt.tight_layout()
    # plt.show()
    plt.savefig(join(outDir, "dataset-ratio-per-change.png"))
    plt.savefig(join(outDir, "dataset-ratio-per-change.pdf"))

    for k in durations:
        all_durs = durations[k]
        durations[k] = {"avg": mean(all_durs),
                        "stdev": 0.0 if len(all_durs) < 2 else stdev(all_durs),
                        "median": median(all_durs),
                        "min": min(all_durs),
                        "max": max(all_durs)}
    with open(join(outDir, "all-errors.json"), 'w') as outFile:
        json.dump(errs, outFile, indent=4)
    with open(join(outDir, "all-fix-durations.json"), 'w') as outFile:
        json.dump(durations, outFile, indent=4)
    examples_path = join(outDir, "examples_per_dur")
    if not exists(examples_path):
        mkdir(examples_path)
    for st_med, (examples, a, b, span) in zip(st_medians, dur_groups):
        with open(join(examples_path, "examples_" + str(span[1]) + "_sec.jsonl"), 'a') as outFile:
            for ex, aa, bb in zip(examples, a, b):
                if st_med - 2.0 <= mean([aa, bb]) <= st_med + 2.0:
                    outFile.write(json.dumps(ex) + "\n")
    examples_size_path = join(outDir, "examples_per_change_size")
    if not exists(examples_size_path):
        mkdir(examples_size_path)
    for st_med, (examples, tdiffs, span) in zip(st_medians, change_size_groups):
        with open(join(examples_size_path, "examples_" + str(int((span[0] + span[1]) / 2)) + "_size.jsonl"), 'a') as outFile:
            for ex in examples:
                outFile.write(json.dumps(ex) + "\n")
    for k in type_diffs:
        all_tdiff = type_diffs[k]
        type_diffs[k] = {"avg": mean(all_tdiff),
                        "stdev": 0.0 if len(all_tdiff) < 2 else stdev(all_tdiff),
                        "median": median(all_tdiff),
                        "min": min(all_tdiff),
                        "max": max(all_tdiff)}
        all_sdiff = str_diffs[k]
        str_diffs[k] = {"avg": mean(all_sdiff),
                        "stdev": 0.0 if len(all_sdiff) < 2 else stdev(all_sdiff),
                        "median": median(all_sdiff),
                        "min": min(all_sdiff),
                        "max": max(all_sdiff)}
    with open(join(outDir, "type-token-changes.json"), 'w') as outFile:
        json.dump(type_diffs, outFile, indent=4)
    with open(join(outDir, "string-token-changes.json"), 'w') as outFile:
        json.dump(str_diffs, outFile, indent=4)
    with open(join(outDir, "median-repair-times-per-num-of-changes.json"), 'w') as outFile:
        json.dump(fix_durs, outFile, indent=4)
    with open(join(outDir, "dataset-ratio-per-num-of-changes.json"), 'w') as outFile:
        json.dump(fix_ratios, outFile, indent=4)
    with open(join(outDir, "parse-times.json"), 'w') as outFile:
        json.dump(parse_times, outFile, indent=4)
    with open(join(outDir, "runtime-times.json"), 'w') as outFile:
        json.dump(runtime_times, outFile, indent=4)
