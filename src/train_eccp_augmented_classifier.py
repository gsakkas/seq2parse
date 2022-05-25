import sys
import resource
from os.path import join, exists
import itertools
from collections import defaultdict
import json
from platform import release
from functools import partial
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
from random import shuffle, seed
from sklearn.preprocessing import MultiLabelBinarizer
from statistics import median_high, mean
from numpy import argsort, array
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from transformer_classifier import AugmentedTransformerClassifier
from run_parse_test_time import print_results, rate, has_parse
from ecpp_individual_grammar import read_grammar, lexed_prog_has_parse


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


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[2].split(" <++> ")
    return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]], samp_1[6], samp_1[7])


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])
    gpuToUse = '/device:GPU:' + sys.argv[4]
    if len(sys.argv) > 5:
        do_time_tests = sys.argv[5] == 'true'
    saved_model_file = join(outDir, 'models', 'augmented-transformer-classifier.h5')

    # Dataset preprocessing
    dataset = []
    test_set = []
    xs = []
    ys = []
    num_chngs = []
    user_ts = []
    with open(join(outDir, "parts-order-augmented.txt"), "w") as out_file:
        for partPath in list(dataDir.glob('part_*')):
            out_file.write(partPath.name + '\n')
    # Because of different directory setup on server
    if 'MANJARO' in release():
        data_dir_2017 = dataDir.joinpath('data_2017')
        for partPath in list(data_dir_2017.glob('part_*')):
            dataset_part_file = join(partPath, "erule-dataset-partials-probs-fixes.txt")
            if exists(dataset_part_file):
                print("#", partPath.name)
                with open(dataset_part_file, "r") as inFile:
                    dataset.extend(list(map(read_sample, inFile.read().split('\n')[:-1])))
                print("# Syntax Errors to repair:", len(dataset))
        data_dir_2018 = dataDir.joinpath('data_2018')
        for partPath in list(data_dir_2018.glob('part_*')):
            dataset_part_file = join(partPath, "erule-dataset-partials-probs-fixes.txt")
            if exists(dataset_part_file):
                print("#", partPath.name)
                with open(dataset_part_file, "r") as inFile:
                    if partPath.name != 'part_6':
                        dataset.extend(list(map(read_sample, inFile.read().split('\n')[:-1])))
                    else:
                        test_set.extend(list(map(read_sample, inFile.read().split('\n')[:-1])))
                print("# Syntax Errors to repair:", len(dataset))
    else:
        for partPath in list(dataDir.glob('part_*')):
            dataset_part_file = join(partPath, "erule-dataset-partials-probs-fixes.txt")
            if exists(dataset_part_file):
                print("#", partPath.name)
                with open(dataset_part_file, "r") as inFile:
                    if partPath.name != 'part_2018_6':
                        dataset.extend(list(map(read_sample, inFile.read().split('\n')[:-1])))
                    else:
                        test_set.extend(list(map(read_sample, inFile.read().split('\n')[:-1])))
            print("# Syntax Errors to repair:", len(dataset))

    seed(42)
    shuffle(dataset)
    old_xs, xs, ys, num_chngs, user_ts, next_tkns, fixed_xs, orig_xs = zip(*dataset)
    old_xs_test, xs_test, ys_test, num_chngs_test, user_ts_test, next_tkns_test, fixed_xs_test, orig_xs_test = zip(*test_set)

    tokens = dict()
    reverse_tokens = dict()
    tokens_file = join(outDir, "tokens_ints-augmented.json")
    rev_tokens_file = join(outDir, "tokens_rev_ints-augmented.json")
    if exists(saved_model_file):
        with open(tokens_file, "r") as fin:
            tokens = json.load(fin)
        with open(rev_tokens_file, "r") as fin:
            new_reverse_tokens = json.load(fin)
        for k in new_reverse_tokens:
            reverse_tokens[int(k)] = new_reverse_tokens[k]
    else:
        new_tokens = sorted(set(itertools.chain.from_iterable((map(lambda x: x.split(), xs)))))
        for i, l in enumerate(new_tokens, start=1):
            tokens[l] = i
            reverse_tokens[i] = l
        reverse_tokens[0] = ""
        with open(tokens_file, "w") as fout:
            json.dump(tokens, fout, indent=4)
        with open(rev_tokens_file, "w") as fout:
            json.dump(reverse_tokens, fout, indent=4)
    xs = [list(map(lambda xx: tokens[xx], x.split())) for x in xs]
    xs_test = [list(map(lambda xx: tokens[xx] if xx in tokens else 0, x.split())) for x in xs_test]

    labels = dict()
    reverse_labels = dict()
    labels_file = join(outDir, "erule_labels-augmented.json")
    rev_labels_file = join(outDir, "erule_reverse_labels-augmented.json")
    if exists(saved_model_file):
        with open(labels_file, "r") as fin:
            labels = json.load(fin)
        with open(rev_labels_file, "r") as fin:
            new_reverse_labels = json.load(fin)
        for k in new_reverse_labels:
            reverse_labels[int(k)] = new_reverse_labels[k]
    else:
        new_labels = sorted(set(itertools.chain.from_iterable(ys)))
        for i, l in enumerate(new_labels, start=1):
            labels[l] = i
            reverse_labels[i] = l
        with open(labels_file, "w") as fout:
            json.dump(labels, fout, indent=4)
        with open(rev_labels_file, "w") as fout:
            json.dump(reverse_labels, fout, indent=4)

    # Use onlt Top-N erules
    erules_cnt = defaultdict(int)
    if exists(saved_model_file):
        with open(join(outDir, "erules-augmented.json"), "r") as fin:
            erules_cnt = json.load(fin)
    else:
        for erls in ys:
            for er in erls:
                erules_cnt[er] += 1
        with open(join(outDir, "erules-augmented.json"), "w") as fout:
            json.dump(erules_cnt, fout, indent=4)
    all_erules = sorted([(erl, erules_cnt[erl]) for erl in erules_cnt], key=lambda x: x[1], reverse=True)

    top_N = 150
    top_erules = all_erules[:top_N]
    top_erules = set([er[0] for er in top_erules])
    top_50_erules_ids = set(map(lambda er: labels[er[0]], all_erules[:50]))
    ys = [list(filter(lambda xx: xx != 0, map(lambda yy: labels[yy] if yy in top_erules else 0, y))) for y in ys]
    ys_test = [list(filter(lambda xx: xx != 0, map(lambda yy: labels[yy] if yy in top_erules else 0, y))) for y in ys_test]

    mlb = MultiLabelBinarizer()
    ys = mlb.fit_transform(ys)

    next_tkns_mlb = MultiLabelBinarizer()
    next_tkns_xs = next_tkns_mlb.fit_transform(next_tkns)
    next_tkns_xs_test = next_tkns_mlb.transform(next_tkns_test)

    dist = list(map(len, xs))
    print("Largest program:", max(dist))
    print("Median size of programs:", median_high(dist))
    print("Mean size of programs:", mean(dist))
    print("Prob. of program to be less than 128:", rate(128, dist))
    print("Prob. of program to be less than 256:", rate(256, dist))
    print("Prob. of program to be less than 512:", rate(512, dist))
    print("Prob. of program to be less than 768:", rate(768, dist))
    print("Prob. of program to be less than 1024:", rate(1024, dist))
    erules = list(map(lambda y: len(list(filter(lambda yy: yy > 0, y))), ys))
    print("Largest num. of erules:", max(erules))
    print("Median num. of erules:", median_high(erules))
    print("Mean num. of erules:", mean(erules))
    # exit(0)

    dist = set([x for xx in map(set, xs) for x in xx])
    vocab_size = len(dist) + 1 # Plus 1 needed for 0s of padding
    x_maxlen = 128
    next_tkns_len = next_tkns_xs.shape[1]

    dataset_size = len(dataset)
    print("Total dataset size:", dataset_size)
    train_size = round(dataset_size - 2000)
    print("Training dataset size:", train_size)
    print("Validation dataset size:", dataset_size - train_size)
    print("Testing dataset size:", len(test_set))
    print("Vocabulary size:", vocab_size)
    print("Next tokens options:", next_tkns_len)
    print("Max. sequence size:", x_maxlen)

    x_train = xs[:train_size]
    next_tkns_x_train = next_tkns_xs[:train_size]
    y_train = ys[:train_size]
    x_val = xs[train_size:]
    x_val_orig = old_xs[train_size:]
    next_tkns_x_val = next_tkns_xs[train_size:]
    y_val = ys[train_size:]
    num_chngs = num_chngs[train_size:]
    user_ts = user_ts[train_size:]
    num_of_labels = y_val.shape[1]

    def tokenize(xx):
        return " ".join(list(map(lambda x: "_UNKNOWN_" if x == 0 else reverse_tokens[x], xx))).strip()

    def labelize(yy, num_preds):
        top_preds = [0] * top_N
        for i in argsort(yy)[::-1][:num_preds]:
            top_preds[i] = 1
        top_preds = list(mlb.inverse_transform(array(top_preds).reshape(1, num_of_labels))[0])
        return list(map(lambda r: reverse_labels[r], top_preds))

    if not exists(join(outDir, "erule-dataset-test-set-generic.txt")):
        with open(join(outDir, "erule-dataset-test-set-generic.txt"), "w") as outFile:
            for tokns, eruls, tok_chgs, dur in zip(old_xs_test, ys_test, num_chngs_test, user_ts_test):
                popular = "popular"
                if not any(map(lambda yy: yy in top_50_erules_ids, eruls)):
                    popular = "not_popular"
                outFile.write(tokns + " <||> " + " <++> ".join(labelize(eruls, 25)) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + popular + "\n")

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=x_maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=x_maxlen)
    xs_test = keras.preprocessing.sequence.pad_sequences(xs_test, maxlen=x_maxlen)

    try:
        # Specify an invalid GPU device
        with tf.device(gpuToUse):
            embed_dim = 128  # Embedding size for each token
            num_heads = 12  # Number of attention heads
            ff_dim = 256  # Hidden layer size in feed forward network inside transformer
            transformer_blks = 4 # Number of transformer blocks
            dense_dims = [256, 128] # Dense layer sizes in classifier

            transformerClfr = AugmentedTransformerClassifier(embed_dim, num_heads, ff_dim, transformer_blks, dense_dims, vocab_size, x_maxlen, num_of_labels, next_tkns_len, 'sigmoid')

            transformerClfr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_2=0.98),
                                    loss="binary_crossentropy",
                                    metrics=[keras.metrics.CategoricalAccuracy(name="acc"),
                                            keras.metrics.TopKCategoricalAccuracy(10, name="top-10-acc")])

            if exists(saved_model_file):
                transformerClfr.load_weights(saved_model_file)
            else:
                history = transformerClfr.fit([x_train, next_tkns_x_train], y_train, ([x_val, next_tkns_x_val], y_val), batch_size=128, epochs=30, verbose=1)
            # a = transformerClfr.predict(x_val[42].reshape(1, x_maxlen))[0]
            # print(a)
            # print(y_val[42])
            # top_10 = argsort(a)[::-1][:10]
            # print(top_10)
            # top_10_preds = [0] * top_N
            # for i in top_10:
            #     top_10_preds[i] = 1
            # top_10_preds = array(top_10_preds).reshape(1, num_of_labels)
            # print(list(mlb.inverse_transform(top_10_preds)[0]))
            # print(list(mlb.inverse_transform(y_val[42].reshape(1, num_of_labels))[0]))
            # print(a.shape)
            # print(max(transformerClfr.predict(x_val[0].reshape(1, x_maxlen))[0]))
            # print(max(transformerClfr.predict(x_val[17].reshape(1, x_maxlen))[0]))
            # print(max(transformerClfr.predict(x_val[42].reshape(1, x_maxlen))[0]))
            # print(max(transformerClfr.predict(x_val[4217].reshape(1, x_maxlen))[0]))
            progs_top_10 = 0
            progs_top_10_one = 0
            progs_top_20 = 0
            progs_top_20_one = 0
            progs_top_20_not_pop = 0
            progs_top_20_one_not_pop = 0
            not_pop_tests = 0
            progs_top_50 = 0
            progs_top_50_one = 0
            progs_best = 0
            progs_best_one = 0
            progs_best_not_pop = 0
            progs_best_one_not_pop = 0
            avg_num_of_preds = 0
            all_best_preds_lens = []
            for y_pred, y_true in zip(transformerClfr.predict([xs_test, next_tkns_xs_test]), ys_test):
                top_50 = argsort(y_pred)[::-1][:50]
                top_20 = top_50[:20]
                top_10 = top_20[:10]
                top_50_preds = [0] * top_N
                for i in top_50:
                    top_50_preds[i] = 1
                top_50_preds = list(mlb.inverse_transform(array(top_50_preds).reshape(1, num_of_labels))[0])
                top_20_preds = [0] * top_N
                for i in top_20:
                    top_20_preds[i] = 1
                top_20_preds = list(mlb.inverse_transform(array(top_20_preds).reshape(1, num_of_labels))[0])
                top_10_preds = [0] * top_N
                for i in top_10:
                    top_10_preds[i] = 1
                top_10_preds = list(mlb.inverse_transform(array(top_10_preds).reshape(1, num_of_labels))[0])
                list_of_erules = list(y_true)
                best = list(map(lambda y_tup: y_tup[0], filter(lambda yy: yy[1] > 0.020, enumerate(y_pred))))
                best_preds = [0] * top_N
                for i in best:
                    best_preds[i] = 1
                best_preds = list(mlb.inverse_transform(array(best_preds).reshape(1, num_of_labels))[0])
                if len(best_preds) > 25:
                    top_25 = top_50[:25]
                    top_25_preds = [0] * top_N
                    for i in top_25:
                        top_25_preds[i] = 1
                    best_preds = list(mlb.inverse_transform(array(top_25_preds).reshape(1, num_of_labels))[0])
                avg_num_of_preds += len(best_preds)
                all_best_preds_lens.append(len(best_preds))
                if len(list_of_erules) == 0 or not any(map(lambda yy: yy in top_50_erules_ids, list_of_erules)):
                    not_pop_tests += 1
                if len(list_of_erules) > 0:
                    if all(map(lambda yy: yy in top_10_preds, list_of_erules)):
                        progs_top_10 += 1
                    if any(map(lambda yy: yy in top_10_preds, list_of_erules)):
                        progs_top_10_one += 1
                    if all(map(lambda yy: yy in top_20_preds, list_of_erules)):
                        progs_top_20 += 1
                    if any(map(lambda yy: yy in top_20_preds, list_of_erules)):
                        progs_top_20_one += 1
                    if all(map(lambda yy: yy in top_50_preds, list_of_erules)):
                        progs_top_50 += 1
                    if any(map(lambda yy: yy in top_50_preds, list_of_erules)):
                        progs_top_50_one += 1
                    if all(map(lambda yy: yy in best_preds, list_of_erules)):
                        progs_best += 1
                    if any(map(lambda yy: yy in best_preds, list_of_erules)):
                        progs_best_one += 1
                    if not any(map(lambda yy: yy in top_50_erules_ids, list_of_erules)):
                        if all(map(lambda yy: yy in top_20_preds, list_of_erules)):
                            progs_top_20_not_pop += 1
                        if any(map(lambda yy: yy in top_20_preds, list_of_erules)):
                            progs_top_20_one_not_pop += 1
                        if all(map(lambda yy: yy in best_preds, list_of_erules)):
                            progs_best_not_pop += 1
                        if any(map(lambda yy: yy in best_preds, list_of_erules)):
                            progs_best_one_not_pop += 1
            print(">> Top 10 predictions accuracy:", progs_top_10 * 100.0 / len(ys_test))
            print(">> Top 20 predictions accuracy:", progs_top_20 * 100.0 / len(ys_test))
            print(">> Top 20 predictions acc. (not popular):", progs_top_20_not_pop * 100.0 / not_pop_tests)
            print(">> Top 50 predictions accuracy:", progs_top_50 * 100.0 / len(ys_test))
            print(">> Best predictions accuracy:", progs_best * 100.0 / len(ys_test))
            print(">> Best predictions acc. (not popular):", progs_best_not_pop * 100.0 / len(ys_test))
            print(">> Num. of non-popular (not top 50):", not_pop_tests)
            print(">> Avg. Number of best predictions:", avg_num_of_preds / len(ys_test))
            print(">> Median Number of best predictions:", median_high(all_best_preds_lens))
            print(">> Min Number of best predictions:", min(all_best_preds_lens))
            print(">> Max Number of best predictions:", max(all_best_preds_lens))
            # print(">> Top 10 preds acc. (at least one):", progs_top_10_one * 100.0 / len(ys_test))
            # print(">> Top 20 preds acc. (at least one):", progs_top_20_one * 100.0 / len(ys_test))
            # print(">> Top 20 preds acc. (at least one) (not popular):", progs_top_20_one_not_pop * 100.0 / not_pop_tests)
            # print(">> Top 50 preds acc. (at least one):", progs_top_50_one * 100.0 / len(ys_test))
            # print(">> Best preds acc. (at least one):", progs_best_one * 100.0 / len(ys_test))
            # print(">> Best preds acc. (at least one) (not popular):", progs_best_one_not_pop * 100.0 / len(ys_test))
            with open(join(outDir, "AugmentedClassifierStats.txt"), "w") as out_file:
                out_file.write(">> Parameters:\n")
                out_file.write("embed_dim = " + str(embed_dim) + "\n")
                out_file.write("num_heads = " + str(num_heads) + "\n")
                out_file.write("ff_dim = " + str(ff_dim) + "\n")
                out_file.write("transformer_blks = " + str(transformer_blks) + "\n")
                out_file.write("dense_dims = " + str(dense_dims) + "\n")
                out_file.write("-" * 42 + "\n")
                out_file.write(">> Test set results:\n")
                out_file.write("Top 10 predictions accuracy: " + str(progs_top_10 * 100.0 / len(ys_test)) + "\n")
                out_file.write("Top 20 predictions accuracy: " + str(progs_top_20 * 100.0 / len(ys_test)) + "\n")
                out_file.write("Top 20 predictions acc. (not popular): " + str(progs_top_20_not_pop * 100.0 / not_pop_tests) + "\n")
                out_file.write("Top 50 predictions accuracy: " + str(progs_top_50 * 100.0 / len(ys_test)) + "\n")
                out_file.write("Best predictions accuracy: " + str(progs_best * 100.0 / len(ys_test)) + "\n")
                out_file.write("Best predictions acc. (not popular): " + str(progs_best_not_pop * 100.0 / len(ys_test)) + "\n")
                out_file.write("Num. of non-popular (not top 50): " + str(not_pop_tests) + "\n")
                out_file.write("Avg. Number of best predictions: " + str(avg_num_of_preds / len(ys_test)) + "\n")
                out_file.write("Median Number of best predictions: " + str(median_high(all_best_preds_lens)) + "\n")
                out_file.write("Min Number of best predictions: " + str(min(all_best_preds_lens)) + "\n")
                out_file.write("Max Number of best predictions: " + str(max(all_best_preds_lens)) + "\n")
                out_file.write("Top 10 preds acc. (at least one): " + str(progs_top_10_one * 100.0 / len(ys_test)) + "\n")
                out_file.write("Top 20 preds acc. (at least one): " + str(progs_top_20_one * 100.0 / len(ys_test)) + "\n")
                out_file.write("Top 20 preds acc. (at least one) (not popular): " + str(progs_top_20_one_not_pop * 100.0 / not_pop_tests) + "\n")
                out_file.write("Top 50 preds acc. (at least one): " + str(progs_top_50_one * 100.0 / len(ys_test)) + "\n")
                out_file.write("Best preds acc. (at least one): " + str(progs_best_one * 100.0 / len(ys_test)) + "\n")
                out_file.write("Best preds acc. (at least one) (not popular): " + str(progs_best_one_not_pop * 100.0 / len(ys_test)) + "\n")
            if not exists(saved_model_file):
                transformerClfr.save_weights(saved_model_file)

            if do_time_tests:
                ERROR_GRAMMAR = read_grammar(grammarFile)
                terminals = ERROR_GRAMMAR.get_alphabet()
                TIMEOUT = 60 * 5 + 5
                parses_bad = 0
                done = 0
                failed = 0
                dataset = [(x, labelize(y_pred, 20), user_t) for x, y_pred, user_t in zip(old_xs_test, transformerClfr.predict([xs_test, next_tkns_xs_test]), user_ts_test)]
                avg_run_time = 0.0
                total_size = 0
                parsed_progs_times = []
                time_gains = []
                with ProcessPool(max_workers=14, max_tasks=5) as pool:
                    new_has_parse = partial(has_parse, ERROR_GRAMMAR)
                    future = pool.map(new_has_parse, dataset, chunksize=1, timeout=TIMEOUT)
                    it = future.result()
                    while True:
                        try:
                            bruh = next(it)
                            if bruh:
                                parse_bad, run_time, dt, size = bruh
                                if parse_bad:
                                    parses_bad += 1
                                avg_run_time += run_time
                                parsed_progs_times.append(run_time)
                                total_size += size
                                time_gains.append(dt)
                                done += 1
                                if (failed + done) % 50 == 0:
                                    print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, outDir, "trans_cls_results-augmented.txt", max_time=120)
                        except StopIteration:
                            break
                        except (TimeoutError, ProcessExpired):
                            failed += 1
                            if (failed + done) % 50 == 0:
                                print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, outDir, "trans_cls_results-augmented.txt", max_time=120)
                        except Exception as e:
                            print("WHY here?!", str(e))
                            failed += 1
                            if (failed + done) % 50 == 0:
                                print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, outDir, "trans_cls_results-augmented.txt", max_time=120)
                    print_results(failed, done, parses_bad, avg_run_time, parsed_progs_times, total_size, time_gains, outDir, "trans_cls_results-augmented.txt", max_time=120)
            else:
                with open(join(outDir, "test-set-top-20-augmented.txt"), "w") as outFile:
                    for tokns, y_pred, tok_chgs, dur, eruls, fixed_tokns in zip(old_xs_test, transformerClfr.predict([xs_test, next_tkns_xs_test]), num_chngs_test, user_ts_test, ys_test, fixed_xs_test):
                        # best = list(map(lambda y_tup: y_tup[0], filter(lambda yy: yy[1] > 0.020, enumerate(y_pred))))
                        # best_preds = [0] * top_N
                        # for i in best:
                        #     best_preds[i] = 1
                        # best_preds = list(mlb.inverse_transform(array(best_preds).reshape(1, num_of_labels))[0])
                        # if len(best_preds) > 25:
                        #     top_25 = argsort(y_pred)[::-1][:25]
                        #     top_25_preds = [0] * top_N
                        #     for i in top_25:
                        #         top_25_preds[i] = 1
                        #     best_preds = list(mlb.inverse_transform(array(top_25_preds).reshape(1, num_of_labels))[0])
                        # best_preds = list(map(lambda r: reverse_labels[r], best_preds))
                        popular = "popular"
                        if not any(map(lambda yy: yy in top_50_erules_ids, eruls)):
                            popular = "not_popular"
                        outFile.write(tokns + " <||> " + " <++> ".join(labelize(y_pred, 20)) + " <||> " + str(tok_chgs) + " <||> " + str(dur) + " <||> " + fixed_tokns + " <||> " + popular + "\n")
    except RuntimeError as e:
        print(e)
