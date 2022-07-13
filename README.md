# Seq2Parse: Neurosymbolic Parse Error Repair

## Getting Started

Let's quickly walk through setting up a development environment.

All programs mentioned below should be able to run on a multicore 16GB RAM
laptop, with or without a dedicated GPU. For training the machine learning (ML)
model, a modern GPU would be required, but since the training data are not
publicly available we provided the pre-trained model. The code requires at least
Python 3.7.5 and ideally Python 3.10.x, since it was tested mainly with the
latest version.

### Getting the source

All **Seq2Parse code** is available at this [GitHub repo] and a **demo website**
that runs Seq2Parse can be found [here]. You can clone the code and the
pre-trained model using:

``` shell
~ $ git clone https://github.com/gsakkas/seq2parse.git
~ $ cd seq2parse/src
```

[GitHub repo]: https://github.com/gsakkas/seq2parse
[here]: http://seq2parse.goto.ucsd.edu/index.html

### Installing

This project uses Python (3.10) for learning/executing the model and generating
the parse error repairs for new programs. We recommend using [virtualenv].

[virtualenv]: https://virtualenv.pypa.io/en/stable/

``` shell
~/seq2parse/src $ virtualenv .venv
~/seq2parse/src $ source .venv/bin/activate
~/seq2parse/src $ pip install -r requirements.txt
```

Our ML models use [tensorflow] and CPU-only support is enough for predictions
and Seq2Parse, but it might run a bit slow. For GPU support and more information
on tensorflow see the online [instructions].

[tensorflow]: https://www.tensorflow.org/
[instructions]:  https://www.tensorflow.org/install/pip#linux

### Testing

Let's run a quick test to make sure everything was installed correctly. We'll
use the pre-trained Transformer classifier to predict a set of error rules and
then generate a parse error repair for the paper example.

**To run Seq2Parse**, execute the following command:

``` shell
~/seq2parse/src $ python seq2parse.py python-grammar.txt ./models 0 human_study/orig_paper.py
```

You should get the following output:

``` shell
=================================================================
Total params: 5,270,550
Trainable params: 5,270,550
Non-trainable params: 0
_________________________________________________________________
1/1 [==============================] - 1s 804ms/step
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def foo(a):
    return a + 42

def bar(a):
    b = foo(a) + 17
    return b +

-----------------Repaired Program-----------------
def foo(a):
    return a + 42

def bar(a):
    b = foo(a) + 17
    return b

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

The command line output should include the ML model architecture at the start,
followed by the lines with `Original Buggy Program` and `Repaired Program` near
the end. Those sections show the original buggy input program `orig_paper.py`
and the repaired program that Seq2Parse produces. We observe that the last line
`return b +` is fixed by removing the extra `+` operator.

## Reproducing the evaluation (Step-by-Step Instructions)

Part of the paper evaluation can not be reproduced due the private dataset that
we used. We provide the full anonymized test set for some parts of the
evaluation and a very small subset of it when the full programs are needed.

### Accuracy (Sec. 7.1)

The first experiment compares the accuracy of our transformer classifiers. We
provide the pre-trained model and all the necessary auxiliary files in the
`models` directory. We also provide an anonymized test set of the abstracted
program token sequences with the ground truth error production rules in
`datasets/erule-test-set-generic.txt` Each line consists of the abstracted
sequences, the error rules, the number of token changes and time the original
author of the program needed to fix it.

The following command will make predictions for the whole test set. It will take
a couple of minutes to finish.

``` shell
~/seq2parse/src $ python predict_eccp_classifier_partials.py python-grammar.txt ./datasets/erule-test-set-generic.txt ./models 0 false
```

The output should look like this:

``` shell
=================================================================
Total params: 5,270,550
Trainable params: 5,270,550
Non-trainable params: 0
_________________________________________________________________
957/957 [==============================] - 46s 47ms/step
>> Top 10 predictions accuracy: 70.65437667516507
>> Top 10 predictions acc. (rare): 53.609083536090836
>> Top 20 predictions accuracy: 79.55481466954305
>> Top 20 predictions acc. (rare): 66.01784266017843
>> Top 50 predictions accuracy: 90.16800679871871
>> Top 50 predictions acc. (rare): 78.02108678021087
>> Threshold predictions accuracy: 77.46616983722298
>> Threshold predictions acc. (rare): 62.530413625304135
>> Num. of rare programs: 1233 (4.030202000392234%)
>> Avg. Number of threshold predictions: 14.086029940511212
>> Median Number of threshold predictions: 14
>> Min Number of threshold predictions: 2
>> Max Number of threshold predictions: 20
```

The `Top N predictions accuracy` lines show the accuracy for the top N predicted
error rules, presented in the *Abstracted* blue bar in the paper's Figure 13.
Additionally, the `Top N predictions acc. (rare)` lines show the the rare
programs accuracy in the *Abstracted* green bar in Figure 13. Finally,
`Threshold predictions accuracy`  and `Threshold predictions acc. (rare)` show
the *Threshold* accuracies of Figure 13.

### Repaired Program Preciseness & Efficiency (Sec. 7.2 & 7.3)

Unfortunately, we can't provide the full dataset to reproduce the program repair
rates. We will show though the repair rate for the *50 public programs* that we
used for our human study, that can be found in the `human_study` directory. Each
`program_N` subdirectory contains the original buggy program `original.py`, the
user fix `user_fix.py`, the Seq2Parse repair `eccp_repair.py` along with other
auxiliary files. All these programs are included at our [website demo], where we
run a smaller version of Seq2Parse.

[website demo]: http://seq2parse.goto.ucsd.edu/index.html

Run the `human_study` evaluation with `run_parse_test_time_top_n_small.py`, a
smaller version of `run_parse_test_time_top_n_preds_partials.py` which runs the
full test set evaluation but requires the private dataset. This evaluation will
take approximately 6 minutes on a modern laptop to complete the automated repair
and data analysis using the pre-trained transformer classifier for the
predicting error rules.

``` shell
~/seq2parse/src $ python run_parse_test_time_top_n_small.py python-grammar.txt ./human_study ./models 20 10 predict
```

and we expect an output similar to this:

``` shell
_________________________________________________________________
2/2 [==============================] - 1s 31ms/step
Programs to repair: 50
100%|████████████████████████████████████████████████████████████| 50/50 [05:27<00:00,  6.56s/it]
---------------------------------------------------
Dataset size: 50
Parse accuracy (%): 100.0
Mean parse time (sec): 6.458792008679957
Median parse time (sec): 1.8143638589972397
Dataset repaired faster than user (%): 96.0
User fix accuracy (%): 10.0
All error locations fixed accuracy (%): 62.0
Any error locations fixed accuracy (%): 90.0
1 sec: Parse accuracy = 22.0
5 sec: Parse accuracy = 80.0
10 sec: Parse accuracy = 86.0
15 sec: Parse accuracy = 90.0
20 sec: Parse accuracy = 94.0
25 sec: Parse accuracy = 96.0
30 sec: Parse accuracy = 96.0
35 sec: Parse accuracy = 96.0
40 sec: Parse accuracy = 96.0
45 sec: Parse accuracy = 96.0
50 sec: Parse accuracy = 96.0
55 sec: Parse accuracy = 96.0
60 sec: Parse accuracy = 96.0
---------------------------------------------------
```

`Parse accuracy` shows how many programs were repaired and parsed by Seq2Parse
and uses the *MinimumCost* approach from Figure 14. `Median parse time` shows
the median time needed to repair and parse a program in this set. The rest of
the results are also self-explanatory.

For the *20 Most Popular* approach on Figure 14, use:

``` shell
~/seq2parse/src $ python run_parse_test_time_top_n_small.py python-grammar.txt ./human_study ./models 20 10 top-20
```

and expect an output similar to:

``` shell
_________________________________________________________________
2/2 [==============================] - 1s 31ms/step
Programs to repair: 50
100%|████████████████████████████████████████████████████████████| 50/50 [06:15<00:00,  7.50s/it]
---------------------------------------------------
Dataset size: 50
Parse accuracy (%): 82.0
Mean parse time (sec): 7.398970724940446
Median parse time (sec): 2.4660917939982028
Dataset repaired faster than user (%): 94.0
User fix accuracy (%): 14.0
All error locations fixed accuracy (%): 64.0
Any error locations fixed accuracy (%): 74.0
1 sec: Parse accuracy = 14.0
5 sec: Parse accuracy = 70.0
10 sec: Parse accuracy = 82.0
15 sec: Parse accuracy = 86.0
20 sec: Parse accuracy = 92.0
25 sec: Parse accuracy = 94.0
30 sec: Parse accuracy = 96.0
35 sec: Parse accuracy = 96.0
40 sec: Parse accuracy = 96.0
45 sec: Parse accuracy = 96.0
50 sec: Parse accuracy = 96.0
55 sec: Parse accuracy = 96.0
60 sec: Parse accuracy = 98.0
---------------------------------------------------
```

### Usefulness (Sec. 7.4)

The rest of our evaluation was based on a user study. We used a web interface
that can be reached at this [link].

[link]: https://dijkstra.eecs.umich.edu/~endremad/APR_HumanEval/

All the program stimuli are also included at our [website demo], where we run a
smaller version of Seq2Parse. Additionally, these programs or any other *Python
program with syntax errors* can use our website demo or the local command:

``` shell
~/seq2parse/src $ [CUDA_VISIBLE_DEVICES=GPU_NUM] python seq2parse.py python-grammar.txt ./models GPU_NUM INPUT_PROG.py
```

where `INPUT_PROG.py` is a Python program file and `GPU_NUM` is the ID of the
local GPU to use (if multiple exist). If the script variable `GPU_NUM` is not
working, try exporting `CUDA_VISIBLE_DEVICES=GPU_NUM` as well at the beginning.
If only one GPU is available set `GPU_NUM` to 0. If no GPUs are available, the
script will default to CPU usage.

The output of the above Seq2Parse command is the original program and the
Seq2Parse repair. The command will also generate a `INPUT_PROG.py.json` file at
a `.seq2parse` directory (generated if it doesn't exist, at the same directory
as `INPUT_PROG.py`). `INPUT_PROG.py.json` has information used by the [website
demo] for line-by-line syntax error feedback (and will also be used on a future
VS Code plugin).

## Conclusion

While the lack of the Python dataset, makes it infeasible to recreate the full
paper evaluation, we believe that the code we have provided can *help future
researchers to build on and compare with our work*.

**Additional scripts** are provided to *train* on *any* programming language
dataset and repair *new* programs, as long as a dataset of fixed buggy programs
and the language grammar are given. For example, `create_ecpp_dataset_full.py`
extracts a machine learning appropriate dataset from the dataset of fixed buggy
program pairs after performing the appropriate program abstraction,
`learn_PCFG.py` learns the probabilistic grammar needed for the abstraction,
`train_eccp_classifier_partials.py` trains the transformer classifier on the
abstracted programs and the ground truth error production rules and, finally,
`run_parse_test_time_top_n_preds_partials.py` run all the relevant repair
experiments on a fixed buggy program test set.