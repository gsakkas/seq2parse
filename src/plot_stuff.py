import sys
from os.path import join
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def cum_dist_fun(secs, durs):
    in_set = list(filter(lambda x: x <= secs, durs))
    return len(in_set) * 100.0 / len(durs)


def dist_fun(secs, durs, step):
    in_set = list(filter(lambda x: secs - step < x <= secs, durs))
    return len(in_set) * 100.0 / len(durs)


if __name__ == "__main__":
    dataDir = Path(sys.argv[1])
    outDir = Path(sys.argv[2])
    # Plot CDF of fix durations of the dataset
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    with open(join(dataDir, "accuracies-per-token-human-edits.json"), "r") as in_file, \
         open(join(dataDir, "accuracies-per-token-dist.json"), "r") as in_file_dist, \
         open(join(dataDir, "accuracies-per-token-25min.json"), "r") as in_file_25_min:
        accuracies_25_min = json.load(in_file_25_min)
        accuracies_dist = json.load(in_file_dist)
        accuracies_edits = json.load(in_file)
        fig, ax = plt.subplots(figsize=(12, 9))
        labels = list(sorted(list(set([int(k) for k in accuracies_edits] + [int(k) for k in accuracies_dist] + [int(k) for k in accuracies_25_min]))))[1:]
        # Exclude weird 0 token changes
        accs_25_min = [accuracies_25_min[str(k)] if str(k) in accuracies_25_min else 0.0 for k in labels]
        accs_dist = [accuracies_dist[str(k)] if str(k) in accuracies_dist else 0.0 for k in labels]
        accs_edits = [accuracies_edits[str(k)] if str(k) in accuracies_edits else 0.0 for k in labels]
        width = 0.30
        xs1 = np.arange(len(labels))
        xs2 = [x + width for x in xs1]
        xs3 = [x + 2*width for x in xs1]
        xs = [x + width for x in xs1]

        ax.bar(xs1, accs_25_min, width, label='ECPP-Orig (25 min)', zorder=3)
        ax.bar(xs2, accs_dist, width, label='ECPP-Dist (5 min)', zorder=3)
        ax.bar(xs3, accs_edits, width, label='ECPP-Edits (25 min)', zorder=3)

        maxx_percent = ((max(max(accs_25_min), max(accs_dist), max(accs_edits)) + 5) // 5) * 5
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim([0, maxx_percent])

        ax.grid(True, which='both', zorder=0)
        ax.legend(loc='upper left')
        ax.set_xlabel('Token Changes')
        ax.set_ylabel('Parse Rate (\%)')

        plt.grid(True, which='minor', linestyle='--')
        plt.xticks(rotation=60)
        plt.tight_layout()
        # plt.show()
        plt.savefig(join(outDir, "parse-rates-ecpp.png"))

    with open(join(dataDir, "erules-used-per-token.json"), "r") as in_file:
        erules_per_token_changes = json.load(in_file)
        fig, ax = plt.subplots(figsize=(12, 9))
        labels = list(sorted(list(set([int(k) for k in erules_per_token_changes]))))
        erules = [erules_per_token_changes[str(k)] if str(k) in erules_per_token_changes else 0.0 for k in labels]
        # width = 0.30
        xs = np.arange(len(labels))

        ax.bar(xs, erules, label='ECPP-Edits (25 min)', zorder=3)

        maxx_percent = max(erules) + 1
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim([0, maxx_percent])

        ax.grid(True, which='both', zorder=0)
        ax.legend(loc='upper left')
        ax.set_xlabel('Token Changes')
        ax.set_ylabel('Mean Num. of Error Rules Used (\%)')

        plt.grid(True, which='minor', linestyle='--')
        plt.xticks(rotation=60)
        plt.tight_layout()
        # plt.show()
        plt.savefig(join(outDir, "avg-erules-used-ecpp.png"))
