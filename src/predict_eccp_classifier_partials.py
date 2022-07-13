import sys
from os import environ
from os.path import join, exists
import itertools
from collections import defaultdict
import json
import pickle
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
from transformer_classifier import TransformerClassifier
from run_parse_test_time import print_results, rate, has_parse
from ecpp_individual_grammar import read_grammar, lexed_prog_has_parse, get_token_list
import earleyparser_interm_repr


def read_sample(samp):
    samp_1 = samp.split(" <||> ")
    samp_2 = samp_1[1].split(" <++> ")
    return (samp_1[0], samp_2, int(samp_1[2]), float(samp_1[3]), samp_1[4] == "popular")


def predict_error_rules(grammarFile, modelsDir, gpuToUse, input_prog, sfile):
    saved_model_file = join(modelsDir, 'transformer-classifier-partial-parses-probs.h5')
    xs_test = input_prog
    if sfile:
        INTERIM_GRAMMAR = earleyparser_interm_repr.read_grammar(grammarFile)
        rules_used = {}
        with open(join(modelsDir, "rules_usage.json"), "r") as in_file:
            rules_used = json.load(in_file)
        INTERIM_GRAMMAR.update_probs(rules_used)

        ERROR_GRAMMAR = read_grammar(grammarFile)
        terminals = ERROR_GRAMMAR.get_alphabet()
        tokens = get_token_list(input_prog, terminals)
        upd_tokens, _ = earleyparser_interm_repr.get_updated_seq(tokens, INTERIM_GRAMMAR)
        xs_test = [upd_tokens]

    tokens = dict()
    reverse_tokens = dict()
    tokens_file = join(modelsDir, "tokens_ints-partials-probs.json")
    rev_tokens_file = join(modelsDir, "tokens_rev_ints-partials-probs.json")
    if exists(saved_model_file):
        with open(tokens_file, "r") as fin:
            tokens = json.load(fin)
        with open(rev_tokens_file, "r") as fin:
            new_reverse_tokens = json.load(fin)
        for k in new_reverse_tokens:
            reverse_tokens[int(k)] = new_reverse_tokens[k]
    else:
        sys.exit(-1)
    xs_test = [list(map(lambda xx: tokens[xx] if xx in tokens else 0, x.split())) for x in xs_test]

    labels = dict()
    reverse_labels = dict()
    labels_file = join(modelsDir, "erule_labels-partials-probs.json")
    rev_labels_file = join(modelsDir, "erule_reverse_labels-partials-probs.json")
    if exists(saved_model_file):
        with open(labels_file, "r") as fin:
            labels = json.load(fin)
        with open(rev_labels_file, "r") as fin:
            new_reverse_labels = json.load(fin)
        for k in new_reverse_labels:
            reverse_labels[int(k)] = new_reverse_labels[k]
    else:
        sys.exit(-1)


    mlb = MultiLabelBinarizer()
    with open(join(modelsDir, 'myMultiLabelBinarizer.pkl'), 'rb') as f:
        mlb = pickle.load(f)
    vocab_size = 182
    x_maxlen = 128
    num_of_labels = 150

    def tokenize(xx):
        return " ".join(list(map(lambda x: "_UNKNOWN_" if x == 0 else reverse_tokens[x], xx))).strip()

    def labelize(yy, num_preds):
        top_preds = [0] * num_of_labels
        for i in argsort(yy)[::-1][:num_preds]:
            top_preds[i] = 1
        top_preds = list(mlb.inverse_transform(array(top_preds).reshape(1, num_of_labels))[0])
        return list(map(lambda r: reverse_labels[r], top_preds))

    xs_test = keras.preprocessing.sequence.pad_sequences(xs_test, maxlen=x_maxlen)

    try:
        # Specify an invalid GPU device
        with tf.device(gpuToUse):
            embed_dim = 128  # Embedding size for each token
            num_heads = 12  # Number of attention heads
            ff_dim = 256  # Hidden layer size in feed forward network inside transformer
            transformer_blks = 6 # Number of transformer blocks
            dense_dims = [256, 128] # Dense layer sizes in classifier

            transformerClfr = TransformerClassifier(embed_dim, num_heads, ff_dim, transformer_blks, dense_dims, vocab_size, x_maxlen, num_of_labels, 'sigmoid')

            transformerClfr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_2=0.98),
                                    loss="binary_crossentropy",
                                    metrics=[keras.metrics.CategoricalAccuracy(name="acc"),
                                            keras.metrics.TopKCategoricalAccuracy(10, name="top-10-acc")])

            if exists(saved_model_file):
                transformerClfr.load_weights(saved_model_file)
            else:
                sys.exit(-1)
            y_pred = transformerClfr.predict(xs_test)
            if sfile:
                return labelize(y_pred[0], 20)
            else:
                return y_pred
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    inputPath = Path(sys.argv[2])
    modelsDir = Path(sys.argv[3])
    gpuToUse = '/device:GPU:' + sys.argv[4]
    single_file = True
    if len(sys.argv) > 5:
        single_file = sys.argv[5] == 'true'
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if single_file:
        input_prog = inputPath.read_text()

        erules = predict_error_rules(grammarFile, modelsDir, gpuToUse, input_prog, True)
        print(erules)
    else:
        dataset = []
        top_rules_num = 20
        num_of_labels = 150
        with open(inputPath, "r") as inFile:
                dataset = list(map(read_sample, inFile.read().split('\n')[:-1]))
        dataset = [(tokns, erules[:top_rules_num], tok_chgs, user_time, popular)
                    for tokns, erules, tok_chgs, user_time, popular in dataset]
        xs_test, ys_test, _, _, popularities = zip(*dataset)
        labels = dict()
        labels_file = join(modelsDir, "erule_labels-partials-probs.json")
        if exists(labels_file):
            with open(labels_file, "r") as fin:
                labels = json.load(fin)
        else:
            sys.exit(-1)
        ys_test = [list(filter(lambda xx: xx != 0, map(lambda yy: labels[yy] if yy in labels else 0, y))) for y in ys_test]

        mlb = MultiLabelBinarizer()
        with open(join(modelsDir, 'myMultiLabelBinarizer.pkl'), 'rb') as f:
            mlb = pickle.load(f)

        progs_top_10 = 0
        progs_top_10_one = 0
        progs_top_10_not_pop = 0
        progs_top_20 = 0
        progs_top_20_one = 0
        progs_top_20_not_pop = 0
        progs_top_20_one_not_pop = 0
        progs_top_50 = 0
        progs_top_50_one = 0
        progs_top_50_not_pop = 0
        not_pop_tests = 0
        all_tests = len(ys_test)
        progs_best = 0
        progs_best_one = 0
        progs_best_not_pop = 0
        progs_best_one_not_pop = 0
        avg_num_of_preds = 0
        all_best_preds_lens = []
        for y_pred, y_true, popular in zip(predict_error_rules(grammarFile, modelsDir, gpuToUse, xs_test, False), ys_test, popularities):
            top_50 = argsort(y_pred)[::-1][:50]
            top_20 = top_50[:20]
            top_10 = top_20[:10]
            top_50_preds = [0] * num_of_labels
            for i in top_50:
                top_50_preds[i] = 1
            top_50_preds = list(mlb.inverse_transform(array(top_50_preds).reshape(1, num_of_labels))[0])
            top_20_preds = [0] * num_of_labels
            for i in top_20:
                top_20_preds[i] = 1
            top_20_preds = list(mlb.inverse_transform(array(top_20_preds).reshape(1, num_of_labels))[0])
            top_10_preds = [0] * num_of_labels
            for i in top_10:
                top_10_preds[i] = 1
            top_10_preds = list(mlb.inverse_transform(array(top_10_preds).reshape(1, num_of_labels))[0])
            list_of_erules = list(y_true)
            best = list(map(lambda y_tup: y_tup[0], filter(lambda yy: yy[1] > 0.01, enumerate(y_pred))))
            best_preds = [0] * num_of_labels
            for i in best:
                best_preds[i] = 1
            best_preds = list(mlb.inverse_transform(array(best_preds).reshape(1, num_of_labels))[0])
            if len(best_preds) > 20:
                top_20_local = top_50[:20]
                top_20_preds_local = [0] * num_of_labels
                for i in top_20_local:
                    top_20_preds_local[i] = 1
                best_preds = list(mlb.inverse_transform(array(top_20_preds_local).reshape(1, num_of_labels))[0])
            avg_num_of_preds += len(best_preds)
            all_best_preds_lens.append(len(best_preds))
            if len(list_of_erules) == 0 or not popular:
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
                if not popular:
                    if all(map(lambda yy: yy in top_10_preds, list_of_erules)):
                        progs_top_10_not_pop += 1
                    if all(map(lambda yy: yy in top_20_preds, list_of_erules)):
                        progs_top_20_not_pop += 1
                    if all(map(lambda yy: yy in top_50_preds, list_of_erules)):
                        progs_top_50_not_pop += 1
                    if any(map(lambda yy: yy in top_20_preds, list_of_erules)):
                        progs_top_20_one_not_pop += 1
                    if all(map(lambda yy: yy in best_preds, list_of_erules)):
                        progs_best_not_pop += 1
                    if any(map(lambda yy: yy in best_preds, list_of_erules)):
                        progs_best_one_not_pop += 1
        print(">> Top 10 predictions accuracy:", progs_top_10 * 100.0 / all_tests)
        print(">> Top 10 predictions acc. (rare):", progs_top_10_not_pop * 100.0 / not_pop_tests)
        print(">> Top 20 predictions accuracy:", progs_top_20 * 100.0 / all_tests)
        print(">> Top 20 predictions acc. (rare):", progs_top_20_not_pop * 100.0 / not_pop_tests)
        print(">> Top 50 predictions accuracy:", progs_top_50 * 100.0 / all_tests)
        print(">> Top 50 predictions acc. (rare):", progs_top_50_not_pop * 100.0 / not_pop_tests)
        print(">> Threshold predictions accuracy:", progs_best * 100.0 / all_tests)
        print(">> Threshold predictions acc. (rare):", progs_best_not_pop * 100.0 / not_pop_tests)
        print(">> Num. of rare programs:", not_pop_tests, "(" + str(not_pop_tests * 100.0 / all_tests) + "%)")
        print(">> Avg. Number of threshold predictions:", avg_num_of_preds / all_tests)
        print(">> Median Number of threshold predictions:", median_high(all_best_preds_lens))
        print(">> Min Number of threshold predictions:", min(all_best_preds_lens))
        print(">> Max Number of threshold predictions:", max(all_best_preds_lens))
