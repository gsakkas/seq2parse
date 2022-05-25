import sys
import resource
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
    if len(samp_1) < 9:
        return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]], samp_1[6], samp_1[7])
    else:
        return (samp_1[0], samp_1[1], samp_2, int(samp_1[3]), float(samp_1[4]), [samp_1[5]], samp_1[6], samp_1[7], samp_1[8], samp_1[9])


if __name__ == "__main__":
    grammarFile = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])
    gpuToUse = '/device:GPU:' + sys.argv[4]
    if len(sys.argv) > 5:
        do_time_tests = sys.argv[5] == 'true'
    saved_model_file = join(outDir, 'models', 'transformer-classifier-partial-parses-probs.h5')

    program_path = Path(dataDir)
    input_program = program_path.read_text()

    INTERIM_GRAMMAR = earleyparser_interm_repr.read_grammar(grammarFile)
    rules_used = {}
    with open("rules_usage.json", "r") as in_file:
        rules_used = json.load(in_file)
    INTERIM_GRAMMAR.update_probs(rules_used)

    ERROR_GRAMMAR = read_grammar(grammarFile)
    terminals = ERROR_GRAMMAR.get_alphabet()
    tokens = get_token_list(input_program, terminals)
    upd_tokens, _ = earleyparser_interm_repr.get_updated_seq(tokens, INTERIM_GRAMMAR)
    xs_test = [upd_tokens]

    tokens = dict()
    reverse_tokens = dict()
    tokens_file = join(outDir, "tokens_ints-partials-probs.json")
    rev_tokens_file = join(outDir, "tokens_rev_ints-partials-probs.json")
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
    labels_file = join(outDir, "erule_labels-partials-probs.json")
    rev_labels_file = join(outDir, "erule_reverse_labels-partials-probs.json")
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
    with open(join(outDir, 'models', 'myMultiLabelBinarizer.pkl'), 'rb') as f:
        mlb = pickle.load(f)
    vocab_size = 182
    x_maxlen = 128
    num_of_labels = 150

    def tokenize(xx):
        return " ".join(list(map(lambda x: "_UNKNOWN_" if x == 0 else reverse_tokens[x], xx))).strip()

    def labelize(yy, num_preds):
        top_preds = [0] * 150
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
            print(labelize(y_pred[0], 20))
    except RuntimeError as e:
        print(e)
