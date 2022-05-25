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
    plt.rc('font', family='serif', size=24)
    with open(join(dataDir, "erules.json"), "r") as in_file:
        erules = json.load(in_file)
        fig, ax = plt.subplots(figsize=(21, 9))
        erules = sorted([(erl, erules[erl]) for erl in erules], key=lambda x: x[1], reverse=True)
        labels, counts = zip(*erules)
        labels = list(labels)[:201]
        with open(join(dataDir, "top-erules.txt"), "w") as out_file:
            out_file.write("\n".join(labels))
        total_cnts = sum(counts)
        percents = list(map(lambda c: c * 100.0 / total_cnts, counts))[:201]
        print("Error rule coverage for Top-50:", sum(percents[:50]))
        print("Error rule coverage for Top-100:", sum(percents[:100]))
        print("Error rule coverage for Top-150:", sum(percents[:150]))
        xs1 = np.arange(len(labels))

        ax.bar(xs1, percents, zorder=3)

        maxx_percent = 10
        ax.set_xticks(xs1)
        ax.set_xticklabels([str(i) if i % 5 == 0 else "" for i in range(len(labels))])
        ax.set_ylim([0, maxx_percent])

        ax.grid(True, which='major', axis='y', zorder=0)
        # ax.legend(loc='upper left')
        ax.set_xlabel('Error Rule \#')
        ax.set_ylabel('Dataset Coverage (\%)')

        # plt.grid(True, which='minor', linestyle='--')
        plt.xticks(rotation=60)
        plt.tight_layout()
        # plt.show()
        plt.savefig(join(outDir, "erules_dataset_coverage.png"))
