"""
Error Correcting Python Earley Parser.

@author: Georgios Sakkas, Earley Parser based on Hardik's implementation
"""

import argparse
import re
# from ast import parse
from pathlib import Path
from collections import defaultdict, deque
from nltk.tree import Tree
from functools import partial
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Text, Name, Number, String, Punctuation, Operator, Keyword


class Rule():
    """
    Represents a CFG rule.
    """

    def __init__(self, lhs, rhs):
        # Represents the rule 'lhs -> rhs', where lhs is a non-terminal and
        # rhs is a list of non-terminals and terminals.
        self.lhs, self.rhs = lhs, rhs

    def __contains__(self, sym):
        return sym in self.rhs

    def __eq__(self, other):
        if isinstance(other, (Rule, ErrorRule)):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __hash__(self):
        return hash((" ".join(self.lhs), " ".join(self.rhs)))

    def __getitem__(self, i):
        return self.rhs[i]

    def __len__(self):
        return len(self.rhs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.lhs + ' -> ' + ' '.join(self.rhs)

    def error_score(self):
        return 0


class ErrorRule(Rule):

    def error_score(self):
        if self.lhs in ['Err_Dedent', 'Err_Indent', 'Err_Close_Paren', 'Err_Open_Paren']:
            return 0.5
        return 1


class Grammar():
    """
    Represents a CFG.
    """

    def __init__(self):
        # The rules are represented as a dictionary from L.H.S to R.H.S.
        self.rules = defaultdict(list)

    def add(self, rule):
        """
        Adds the given rule to the grammar.
        """
        if rule not in self.rules[rule.lhs]:
            self.rules[rule.lhs].append(rule)

    def get_alphabet(self):
        symbols = set([])
        alphabet = set([])
        for key in self.rules:
            for rule in self.rules[key]:
                for sym in rule.rhs:
                    symbols.add(sym)
        for sym in symbols:
            if self.is_terminal(sym):
                if sym not in alphabet:
                    alphabet.add(sym)
        return alphabet

    def get_tags(self):
        symbols = set([])
        tags = set([])
        for key in self.rules:
            for rule in self.rules[key]:
                for sym in rule.rhs:
                    symbols.add(sym)
        for sym in symbols:
            if self.is_tag(sym):
                if sym not in tags:
                    tags.add(sym)
        return tags

    @staticmethod
    def load_grammar(fpath):
        """
        Loads the grammar from file (from the )
        """
        grammar = Grammar()
        with open(fpath) as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                entries = line.split('->')
                lhs = entries[0].strip()
                for rhs in entries[1].split('<|>'):
                    grammar.add(Rule(lhs, rhs.strip().split()))

        return grammar

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [str(r) for r in self.rules['S']]

        for nt, rule_list in self.rules.items():
            if nt == 'S':
                continue

            s += [str(r) for r in rule_list]

        return '\n'.join(s)

    # Returns the rules for a given Non-terminal.
    def __getitem__(self, nt):
        return self.rules[nt]

    def is_terminal(self, sym):
        """
        Checks is the given symbol is terminal.
        """

        return len(self.rules[sym]) == 0

    def is_tag(self, sym):
        """
        Checks whether the given symbol is a tag, i.e. a non-terminal with rules
        to solely terminals.
        """

        if not self.is_terminal(sym):
            return all(self.is_terminal(s) for r in self.rules[sym] for s in
                r.rhs)

        return False

    def is_nullable(self, sym):
        """
        Checks whether the given symbol is nullable, i.e. a non-terminal with rules
        to null.
        """

        if not self.is_terminal(sym):
            return any(r.rhs==[] for r in self.rules[sym])


class ErrorGrammar(Grammar):
    @staticmethod
    def load_grammar(fpath):
        """
        Loads the grammar from file (from the )
        """

        grammar = ErrorGrammar()

        with open(fpath) as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                entries = line.split('->')
                lhs = entries[0].strip()
                for rhs in entries[1].split('<|>'):
                    rhssymbols = rhs.strip().split()
                    grammar.add(Rule(lhs, rhssymbols))

        # print(len(str(grammar).split('\n')))
        alphabet = grammar.get_alphabet()
        tags = grammar.get_tags()
        alphabet.remove('_ENDMARKER_')
        #first step in Algorithm 1 in AHO's paper
        added_rules_1 = []
        for key in grammar.rules:
            for rule in grammar.rules[key]:
                new_rule_1 = ErrorGrammar.error_rule_1(grammar, rule)
                if new_rule_1 is not None:
                    added_rules_1.append(new_rule_1)
        for rule in added_rules_1:
            grammar.add(rule)

        #second step
        for sym in tags:
            # grammar.add(ErrorRule("Err_" + sym + "_0", [sym]))
            # for sym2 in alphabet:
            #     # if sym2 not in [sym, '_NEWLINE_', '_ENDMARKER_']:
            #     if sym2 not in [sym, '_ENDMARKER_']:
            #         grammar.add(ErrorRule("Err_" + sym + "_0", [sym2]))
            grammar.add(ErrorRule("Err_" + sym, ["Err_Tag"]))
            grammar.add(ErrorRule("Err_" + sym, ["H", sym]))
            # grammar.add(ErrorRule("Err_" + sym, ["InsertErr", "InsertErr", sym]))
            grammar.add(ErrorRule("Err_" + sym, [])) #empty string
            # parsing_test_0: 20:54 mins -> 8.46 mins (58% speedup without deleting)
            # parsing_test_0: 20:54 mins -> 7.52 mins (62% speedup without deleting, with one insertion)
            # parsing_test_0: 20:54 mins -> 7.31 mins (64% speedup without deleting, with one insertion and common Err_Tag for all replacements)
            # parsing_test_0: 20:54 mins -> 5.43 mins (73% speedup without deleting, with one insertion and common Err_Tag for all replacements and 18 max error rules (not 20))
            # parsing_test_0: 20:54 mins -> 5.58 mins (71% speedup with deleting on Err_Tag with one insertion and common Err_Tag for all replacements and 18 max error rules (not 20))
            # parsing_test_0: 20:54 mins -> 7.29 mins (64% speedup with deleting on Err_Tag with two insertions and common Err_Tag for all replacements and 18 max error rules (not 20))
        # grammar.add(ErrorRule("Err_Tag", []))
        for sym in alphabet:
            grammar.add(ErrorRule("Err_Tag", [sym]))
            grammar.add(ErrorRule("InsertErr", [sym]))

        #third step
        grammar.add(Rule("S'", ["S"]))
        grammar.add(ErrorRule("S'", ["S", "InsertErr"]))
        grammar.add(ErrorRule("H", ["H", "InsertErr"]))
        grammar.add(ErrorRule("H", ["InsertErr"]))
        # print(grammar)
        # print(len(str(grammar).split('\n')))
        return grammar

    @staticmethod
    def error_rule_1(grammar,rule):
        if grammar.is_tag(rule.lhs):
            return None
        new_rule_rhs = []
        for sym in rule.rhs:
            if not grammar.is_tag(sym):
            # if not grammar.is_tag(sym) or sym in ['Endmarker']:
                new_rule_rhs.append(sym)
            else:
                new_rule_rhs.append("Err_" + sym)
        if new_rule_rhs != rule.rhs:
            return ErrorRule(rule.lhs,new_rule_rhs)
        else:
            return None


class State():
    """
    Represents a state in the error-correcting Earley algorithm.
    """

    GAM = '<GAM>'

    def __init__(self, rule, dot=0, sent_pos=0, chart_pos=0, error_count=0, back_pointers=[]):
        # CFG Rule.
        self.rule = rule
        # Dot position in the rule.
        self.dot = dot
        # Sentence position.
        self.sent_pos = sent_pos
        # Chart index.
        self.chart_pos = chart_pos
        # Error counter
        self.error_count = error_count
        # Pointers to child states (if the given state was generated using
        # Completer).
        self.back_pointers = back_pointers
        # Hash for the back_pointers for efficient usage
        self.own_hash = hash((self.rule, self.dot, self.sent_pos))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.rule == other.rule and self.dot == other.dot and \
                self.sent_pos == other.sent_pos

        return False

    def __hash__(self):
        return self.own_hash

    def __len__(self):
        return len(self.rule)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        def str_helper(state):
            return ('(' + state.rule.lhs + ' -> ' +
            ' '.join(state.rule.rhs[:state.dot] + ['*'] +
                state.rule.rhs[state.dot:]) +
            (', [%d, %d], %d)' % (state.sent_pos, state.chart_pos, state.error_count)))

        return (str_helper(self) +
            ' (' + ', '.join(str_helper(s) for s in self.back_pointers) + ')')

    def next(self):
        """
        Return next symbol to parse, i.e. the one after the dot
        """

        if self.dot < len(self):
            return self.rule[self.dot]

    def is_complete(self):
        """
        Checks whether the given state is complete.
        """

        return len(self) == self.dot

    @staticmethod
    def init():
        """
        Returns the state used to initialize the chart in the Earley algorithm.
        """

        return State(Rule(State.GAM, ["S'"]))


class ChartEntry():
    """
    Represents an entry in the chart used by the Earley algorithm.
    """

    def __init__(self, states):
        # List of Earley states.
        self.states = deque(states)
        self.other_states = deque([])
        self.seen = dict([])

    def __iter__(self):
        return iter(list(self.seen.keys()) + list(self.states) + list(self.other_states))

    def __len__(self):
        return len(self.seen.keys()) + len(self.states) + len(self.other_states)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n'.join(str(s) for s in ['<<<<Seen>>>>'] + list(self.seen.keys()) + ['<<<<Not seen states>>>>'] + list(self.states) + ['<<<<Not seen other states>>>>'] + list(self.other_states))

    def switch_states(self):
        # print(len(self.states), len(self.other_states))
        self.other_states.extend(self.states)
        self.states = self.other_states
        self.other_states = deque([])
        # print(len(self.states), len(self.other_states))

    def pop_state(self, grammar, cost):
        """
        Return the next unseen state
        """
        while len(self.states) > 0:
            state = self.states.popleft()
            if state.error_count <= cost:
            # print('----------------')
            # print(state)
            # print(state.error_count)
                if state not in self.seen:
                    self.seen[state] = state.error_count
                return state
            if state.is_complete() and grammar.is_nullable(state.rule.lhs):
                self.other_states.append(state)
            else:
                self.other_states.appendleft(state)
        return None

    def add_other_states(self, grammar, state):
        in_other_states = state in self.other_states
        if not in_other_states:
            if state.is_complete() and grammar.is_nullable(state.rule.lhs):
                self.other_states.append(state)
            else:
                self.other_states.appendleft(state)


    def add(self, grammar, state):
        """
        Add the given state (if it hasn't already been added).
        """
        # print(state.error_count)
        # print(state)
        # print('----------------')
        have_not_seen = state not in self.seen
        in_states = state in self.states
        in_other_states = state in self.other_states
        # print('>>>', state)
        # print('have_not_seen', have_not_seen)
        if not in_states and not in_other_states and have_not_seen:
            if state.is_complete() and grammar.is_nullable(state.rule.lhs):
                self.states.append(state)
            else:
                self.states.appendleft(state)
        else:
            same_state = next(x for x in (list(self.states) + list(self.other_states) + list(self.seen.keys())) if x == state)
            if same_state.error_count > state.error_count:
                if in_states:
                    self.states.remove(same_state)
                if in_other_states:
                    self.other_states.remove(same_state)
                if state.is_complete() and grammar.is_nullable(state.rule.lhs):
                    self.states.append(state)
                else:
                    self.states.appendleft(state)


class Chart():
    """
    Represents the chart used in the Earley algorithm.
    """

    def __init__(self, entries):
        # List of chart entries.
        self.entries = entries

    def __getitem__(self, i):
        return self.entries[i]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n\n'.join([("Chart[%d]:\n" % i) + str(entry) for i, entry in
            enumerate(self.entries)])

    @staticmethod
    def init(l):
        """
        Initializes a chart with l entries (Including the dummy start state).
        """

        return Chart([(ChartEntry([]) if i > 0 else
                ChartEntry([State.init()])) for i in range(l)])


class ErrorEarleyParse():
    """
    Represents the Error-correcting Earley-generated parse for a given sentence according to a
    given grammar.
    """

    def __init__(self, sentence, grammar, max_cost=5):
        self.words = sentence.split()
        self.grammar = grammar
        self.chart = Chart.init(len(self.words) + 1)
        # Maximum number of error correcting rules to use
        self.max_cost = max_cost

    def predictor(self, state, pos):
        """
        Error-correcting Earley Predictor.
        """

        # runs = 0
        for rule in self.grammar[state.next()]:
            # This is my optimization to avoid using that many ErrorRules
            self.chart[pos].add(self.grammar, State(rule, dot=0,
                sent_pos=state.chart_pos, chart_pos=state.chart_pos))
            # runs += 1
        # print("===================")
        # print("<<Predictor>>")
        # print("Chart[" + str(pos) + "]")
        # print(self.chart[pos])
        # print(">>>", state)
        # print("===================")
        # return runs

    def scanner(self, state, pos):
        """
        Error-correcting Earley Scanner.
        """
        # runs = 1
        if state.chart_pos < len(self.words):
            word = self.words[state.chart_pos]
            # runs = ([(word in r) for r in self.grammar[state.next()]] + [True]).index(True)
            if any((word in r) for r in self.grammar[state.next()]):
                new_cost = 1 if state.next().startswith('Err_') or state.next() == 'InsertErr' else 0
                if new_cost <= self.max_cost:
                    self.chart[pos + 1].add(self.grammar, State(Rule(state.next(), [word]),
                        dot=1, sent_pos=state.chart_pos,
                        chart_pos=(state.chart_pos + 1),
                        error_count=new_cost))
            # print("===================")
            # print("<<Scanner>>")
            # print("Chart[" + str(pos+1) + "]")
            # print(self.chart[pos+1])
            # print(">>>", state)
            # print("===================")
        # return runs

    def completer(self, state, pos):
        """
        Error-correcting Earley Completer.
        """
        # runs = 0
        # print("===================")
        # print("<<Completer>>")
        # print(">>>", state)
        for prev_state in self.chart[state.sent_pos]:
            if prev_state.next() == state.rule.lhs:
                new_cost = prev_state.error_count + state.error_count + state.rule.error_score()
                if state.rule.lhs == 'Err_Tag':
                    if state.rule.rhs == []:
                        new_cost = 1
                    elif any((state.rule.rhs[0] in r) for r in self.grammar[prev_state.rule.lhs.replace('Err_', '')]):
                        new_cost = prev_state.error_count
                if new_cost <= self.max_cost:
                    self.chart[pos].add(self.grammar, State(prev_state.rule,
                        dot=(prev_state.dot + 1), sent_pos=prev_state.sent_pos,
                        chart_pos=pos,
                        back_pointers=(prev_state.back_pointers + [state]),
                        error_count=new_cost))
            # runs += 1
        # print("Chart[" + str(pos) + "]")
        # print(self.chart[pos])
        # print("===================")
        # return runs

    def parse(self):
        """
        Parses the sentence by running the Earley algorithm and filling out the
        chart.
        """

        # Checks whether the next symbol for the given state is a tag.
        def is_tag(state):
            return self.grammar.is_tag(state.next())

        for cost in range(self.max_cost + 1):
            for i in range(len(self.chart)):
                # print("Chart[" + str(i) + "]")
                # print(len(self.chart[i]))
                # jj = 0
                # print("===================")
                state = self.chart[i].pop_state(self.grammar, cost)
                while state is not None:
                    # print(">>>", state)
                    # print("===================")
                    # jj += 1
                    if not state.is_complete():
                        if is_tag(state) and not self.grammar.is_nullable(state.next()):
                            self.scanner(state, i)
                            self.chart[i].add_other_states(self.grammar, state)
                        else:
                            self.predictor(state, i)
                    else:
                        self.completer(state, i)
                        self.chart[i].add_other_states(self.grammar, state)
                    state = self.chart[i].pop_state(self.grammar, cost)
                #     print("YOOO!", state, state == None)
                # print("HERE", len(self.chart[i].other_states))
                # print(jj)
                # print(self.chart[i])
                # print(len(self.chart[i]))
                # break
                self.chart[i].switch_states()
            if self.has_parse():
                # print("Cost =", cost)
                break
        # for i in range(len(self.chart)):
        #     print("Chart[" + str(i) + "]")
        #     # print("===================")
        #     print(len(self.chart[i]))
        #     # print(self.chart[i])
        # print("Cost =", cost)

    def has_parse(self):
        """
        Checks whether the sentence has a parse.
        """

        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == "S'" and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                return True

        return False

    def get(self):
        """
        Returns the minimum error parse if it exists, otherwise returns None.
        """

        def get_helper(state):
            # print(state)
            if self.grammar.is_tag(state.rule.lhs):
                if state.rule.rhs != []:
                    return Tree(state.rule.lhs, [state.rule.rhs[0]])
                return Tree(state.rule.lhs, [state.rule.lhs.replace('Err_', '')])


            return Tree(state.rule.lhs,
                [get_helper(s) for s in state.back_pointers])

        found_state = None
        errors = float("inf")
        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == "S'" and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                if state.error_count < errors:
                    # get_helper(state).pretty_print()
                    # print(state.error_count)
                    found_state = state
                    errors = state.error_count

        if found_state is not None:
            return get_helper(found_state)

        return None


    def get_rules(self):
        """
        Returns the minimum error parse if it exists, otherwise returns None.
        """

        def get_helper(state):
            # print(state)
            if self.grammar.is_tag(state.rule.lhs):
                return [state.rule]
            result = [state.rule]
            for s in state.back_pointers:
                result.extend(get_helper(s))
            return result

        found_state = None
        errors = float("inf")
        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == "S'" and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                if state.error_count < errors:
                    found_state = state
                    errors = state.error_count

        if found_state is not None:
            return get_helper(found_state)

        return None


    def get_fixed_seq(self):
        """
        Returns the minimum error parse if it exists, otherwise returns None.
        """

        def get_helperrr(state):
            # print(state)
            if self.grammar.is_tag(state.rule.lhs) or \
                (self.grammar.is_tag(state.rule.lhs.replace('Err_', '')) and \
                    (state.rule.rhs == [] or state.rule.rhs[0] != 'H')):
                if state.rule.rhs != []:
                    return Tree(state.rule.lhs, [state.rule.rhs[0]])
                return Tree(state.rule.lhs, [state.rule.lhs.replace('Err_', '')])


            return Tree(state.rule.lhs,
                [get_helperrr(s) for s in state.back_pointers])


        def get_helper(state):
            if self.grammar.is_tag(state.rule.lhs) or \
                (self.grammar.is_tag(state.rule.lhs.replace('Err_', '')) and \
                    (state.rule.rhs == [] or state.rule.rhs[0] != 'H')):
                return [state]
            if state.rule.rhs:
                if state.rule.rhs[0] == 'Err_Tag':
                    return [state]
            result = []
            for s in state.back_pointers:
                result.extend(get_helper(s))
            return result


        def get_erules(state):
            if self.grammar.is_tag(state.rule.lhs):
                return [state.rule]
            result = [state.rule]
            for s in state.back_pointers:
                result.extend(get_erules(s))
            return result


        found_state = None
        errors = float("inf")
        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == "S'" and \
                state.sent_pos == 0 and state.chart_pos == len(self.words):
                if state.error_count < errors:
                    found_state = state
                    errors = state.error_count

        if found_state is not None:
            return (get_helper(found_state), get_erules(found_state), found_state.error_count)

        return None


class Lexer():
    """
    Simple lexer for Python programs
    """

    def __init__(self, terminals):
        self.lexer = get_lexer_by_name("python")
        self.terminals = terminals

    def lex(self, input_program):
        program = input_program
        if len(input_program) > 1:
            if input_program[-1] != '\n':
                program = input_program + '\n'
        program = self.remove_comments_and_strings(program)
        # Some hacks for random errors
        if "’" in program:
            program = program.replace("’", "'")
            program = self.remove_comments_and_strings(program)
        # Clean tabs
        all_lines = []
        for line in program.split('\n'):
            spaces_so_far = 0
            if len(line) > 0:
                if line[0] in [' ', '\t']:
                    for char in line:
                        if char == ' ':
                            spaces_so_far += 1
                        elif char == '\t':
                            spaces_so_far = (spaces_so_far // 4 + 1) * 4
                        else:
                            break
            all_lines.append(' ' * spaces_so_far + line.lstrip().replace('\t', '    '))
        all_lines = list(map(lambda line: list(pygments.lex(line.rstrip(), self.lexer)), all_lines))
        all_lines = self.update_indents_stack(all_lines)
        all_lines = self.update_spaces_and_nls(all_lines)
        all_lines = self.update_tokens(all_lines)
        tokens = [tok for line in all_lines for tok in line]
        tokens = self.final_cleaning(tokens, False)
        return tokens

    def clean_with_lex(self, input_program):
        program = input_program
        # print(program)
        if len(input_program) > 1:
            if input_program[-1] != '\n':
                program = input_program + '\n'
        program = self.remove_comments_and_strings(program)
        # Some hacks for random errors
        if "’" in program:
            program = program.replace("’", "'")
            program = self.remove_comments_and_strings(program)
        # Store strings for later use
        all_strings, all_string_types = self.get_comments_and_strings(input_program, program)
        # print(all_strings, all_string_types)
        # Clean tabs
        all_lines = []
        for line in program.split('\n'):
            spaces_so_far = 0
            if len(line) > 0:
                if line[0] in [' ', '\t']:
                    for char in line:
                        if char == ' ':
                            spaces_so_far += 1
                        elif char == '\t':
                            spaces_so_far = (spaces_so_far // 4 + 1) * 4
                        else:
                            break
            all_lines.append(' ' * spaces_so_far + line.lstrip().replace('\t', '    '))
        all_lines = list(map(lambda line: list(pygments.lex(line.rstrip(), self.lexer)), all_lines))
        all_lines = self.update_indents_stack(all_lines)
        all_lines = self.update_spaces_and_nls(all_lines)
        all_lines = self.update_tokens_with_actual(all_lines)
        tokens = [tok for line in all_lines for tok in line]
        tokens = self.final_cleaning(tokens, True)
        # Put strings back
        # print('-' * 42 + '\n' + input_program + '\n' + '=' * 42 + '\n' + tokens.replace('_NEWLINE_ ', '\n') + '\n' + '*' * 42 + '\n')
        if tokens.count("_STRING_") == len(all_strings):
            # prev_len = len(tokens.split())
            # new_tokens = tokens
            for string, stype in zip(all_strings, all_string_types):
                if ' ' in string:
                    string = string.replace(' ', "_white_space_")
                tokens = tokens.replace('_STRING_', stype + string + stype, 1)
            # print(len(new_tokens.split()), prev_len)
            # print(new_tokens.split())
            # print(tokens.split())
            # if len(new_tokens.split()) == len(tokens.split()):
            #     tokens = new_tokens
            # else:
            #     tokens = tokens.replace('_STRING_', "\"_some_string_\"")
        else:
            tokens = tokens.replace('_STRING_', "\"_some_string_\"")
        # print('-' * 42 + '\n' + '=' * 42 + '\n' + tokens.replace('_NEWLINE_ ', '\n') + '\n' + '*' * 42 + '\n')
        return tokens

    def remove_comments_and_strings(self, input_prog):
        prog = re.sub(re.compile(r"\\\s*?\n") , "" , input_prog)
        # Temporary replacements
        prog = prog.replace("\\\\", "__temporary__")
        prog = prog.replace("\\\"", "__double_quote__")
        prog = prog.replace("\\\'", "__single_quote__")
        prog = prog.replace("__temporary__", "\\\\")
        # String and comment replacements
        prog = re.sub(re.compile(r"\n\s*#.*?\n") , "\n" , prog)
        prog = re.sub(re.compile(r"\"\"\".*?\"\"\"", flags=re.DOTALL) , "__triple_dstring__" , prog)
        prog = re.sub(re.compile(r"\'\'\'.*?\'\'\'", flags=re.DOTALL) , "__triple_sstring__" , prog)
        in_double_quote = False
        in_single_quote = False
        in_comment = False
        new_prog = ""
        for char in prog:
            if not in_comment:
                if not in_double_quote and not in_single_quote and char == "#":
                    in_comment = True
                    new_prog += char
                elif not in_double_quote and not in_single_quote and char == "\"":
                    in_double_quote = True
                    new_prog += char
                elif not in_double_quote and not in_single_quote and char == "\'":
                    in_single_quote = True
                    new_prog += char
                elif in_double_quote and not in_single_quote and char == "\"":
                    in_double_quote = False
                    new_prog += char
                elif in_double_quote and not in_single_quote and char == "\'":
                    new_prog += "__single_quote__"
                elif not in_double_quote and in_single_quote and char == "\'":
                    in_single_quote = False
                    new_prog += char
                elif not in_double_quote and in_single_quote and char == "\"":
                    new_prog += "__double_quote__"
                else:
                    new_prog += char
            else:
                if char ==  "\n":
                    in_comment = False
                    new_prog += char
                elif char == "\'":
                    new_prog += "__single_quote__"
                elif char == "\"":
                    new_prog += "__double_quote__"
                else:
                    new_prog += char
        prog = new_prog
        prog = re.sub(re.compile(r"\"([^(\"|\'|\n)]|\(|\)|\|)*?\"") , "\"__string__\"" , prog)
        prog = re.sub(re.compile(r"\'([^(\"|\'|\n)]|\(|\)|\|)*?\'") , "\'__string__\'" , prog)
        prog = prog.replace("__triple_dstring__", "\"__string__\"")
        prog = prog.replace("__triple_sstring__", "\'__string__\'")
        prog = re.sub(re.compile(r"#.*?\n" ) , "\n" , prog)
        prog = re.sub(re.compile(r"\n\s+\n" ) , "\n" , prog)
        while prog.find('\n\n') >= 0:
            prog = prog.replace('\n\n', '\n')
        return prog

    def get_comments_and_strings(self, input_prog, prog):
        strings = []
        string_types = []
        clean_input_prog = input_prog.replace("’", "'")
        if len(clean_input_prog) > 1:
            if clean_input_prog[-1] != '\n':
                clean_input_prog = clean_input_prog + '\n'
        clean_input_prog = re.sub(re.compile(r"\\\s*?\n") , "" , clean_input_prog)
        clean_input_prog = re.sub(re.compile(r"\n\s*#.*?\n") , "\n" , clean_input_prog)
        clean_input_prog = re.sub(re.compile(r"#[^\'\"]*?\n" ) , "\n" , clean_input_prog)
        clean_input_prog = re.sub(re.compile(r"\n\s+\n" ) , "\n" , clean_input_prog)
        while clean_input_prog.find('\n\n') >= 0:
            clean_input_prog = clean_input_prog.replace('\n\n', '\n')
        # print("=*" * 42 + "=")
        # print(clean_input_prog)
        parts = prog.split("__string__")
        # print(parts)
        while parts:
            part = parts.pop(0)
            clean_input_prog = clean_input_prog.replace(part, '', 1)
            if parts == [] or clean_input_prog == "" or clean_input_prog.split(parts[0]) == []:
                break
            split_prog = clean_input_prog.split(parts[0])
            strings.append(split_prog[0].replace('\n', '_NEWLINE_'))
            clean_input_prog = clean_input_prog.replace(strings[-1], '', 1)
            if len(clean_input_prog) > 2 and clean_input_prog[:3] == "\"\"\"":
                string_types.append("\"\"\"")
            elif len(clean_input_prog) > 2 and clean_input_prog[:3] == "\'\'\'":
                string_types.append("\'\'\'")
            elif len(clean_input_prog) > 0 and clean_input_prog[0] == "\"":
                string_types.append("\"")
            elif len(clean_input_prog) > 0 and clean_input_prog[0] == "\'":
                string_types.append("\'")
            else:
                string_types.append("\"")
        return strings, string_types

    def update_indents_stack(self, all_lines):
        all_line_tokens = []
        lst_token_prev_line = False
        fst_token_this_line = False
        indents = []
        paren_so_far = 0
        curly_so_far = 0
        square_so_far = 0
        for token_list in all_lines:
            fst_token = token_list[0]
            tok_idx = 0
            fst_real_token = token_list[tok_idx]
            while fst_real_token[0] in Text and fst_real_token[1].replace(' ', '') == '':
                tok_idx += 1
                if tok_idx < len(token_list):
                    fst_real_token = token_list[tok_idx]
                else:
                    break
            fst_token_this_line = fst_real_token[0] in Operator and fst_real_token[1] in ['+', '-', '*', '/', '//', '%',  '==', '!=', 'in', 'or', 'and'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Punctuation and fst_real_token[1] in [',', '}', ')', ']'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Punctuation and fst_real_token[1] in ['(', '['] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in Keyword and fst_real_token[1] == 'for' and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            fst_token_this_line |= fst_real_token[0] in String and lst_token_prev_line and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
            if lst_token_prev_line:
                # Checks if previous line ends with an operator, paren etc. and we are within a parenthesis, thus we must not indent
                last_line_tokens = all_line_tokens.pop()
                if len(last_line_tokens) > 1:
                    all_line_tokens.append(last_line_tokens[:-1])
                all_line_tokens.append(token_list)
            elif fst_token_this_line:
                # Checks if line starts with an operator and we are within a parenthesis, thus we must not indent
                last_line_tokens = all_line_tokens.pop()
                if len(last_line_tokens) > 1:
                    all_line_tokens.append(last_line_tokens[:-1])
                all_line_tokens.append(token_list[tok_idx:])
            elif fst_token[0] in Text and fst_token[1].replace(' ', '') == '':
                this_indent = len(fst_token[1])
                if indents == [] and this_indent > 0:
                    indents.append(this_indent)
                    all_line_tokens.append([(fst_token[0], '_INDENT_')] + token_list[1:])
                elif indents == []:
                    all_line_tokens.append(token_list[1:])
                elif indents[-1] == this_indent:
                    all_line_tokens.append(token_list[1:])
                elif indents[-1] < this_indent:
                    indents.append(this_indent)
                    all_line_tokens.append([(fst_token[0], '_INDENT_')] + token_list[1:])
                elif indents[-1] > this_indent:
                    dedents = 0
                    while indents[-1] > this_indent:
                        dedents += 1
                        indents.pop()
                        if indents == []:
                            break
                    all_line_tokens.append([(fst_token[0], '_DEDENT_')] * dedents + token_list[1:])
            elif not(fst_token[0] in Text and fst_token[1].replace('\n', '') == '') and \
                    len(indents) > 0:
                all_line_tokens.append([(Text, '_DEDENT_')] * len(indents) + token_list)
                indents = []
            else:
                all_line_tokens.append(token_list)
            if len(token_list) > 1:
                lst_token = token_list[-2]
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '(':
                        paren_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == ')':
                        paren_so_far -= 1
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '{':
                        curly_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == '}':
                        curly_so_far -= 1
                for tok in token_list:
                    if tok[0] in Punctuation and tok[1] == '[':
                        square_so_far += 1
                    elif tok[0] in Punctuation and tok[1] == ']':
                        square_so_far -= 1
                lst_token_prev_line = lst_token[0] in Punctuation and lst_token[1] in ['\\', '{', '(', '[']
                lst_token_prev_line |= lst_token[0] in Punctuation and lst_token[1] == ','  and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
                lst_token_prev_line |= token_list[-1][0] in Text and token_list[-1][1] == '\\\n'
                lst_token_prev_line |= lst_token[0] in Punctuation and lst_token[1] == ':' and curly_so_far > 0
                lst_token_prev_line |= lst_token[0] in Operator and lst_token[1] in ['+', '-', '*', '/', '//', '%', '==', '!=', 'in', 'or', 'and'] and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
                lst_token_prev_line |= lst_token[0] in String and (paren_so_far > 0 or curly_so_far > 0 or square_so_far > 0)
        if len(indents) > 0:
            all_line_tokens.append([(Text, '_DEDENT_')] * len(indents))
        return all_line_tokens


    def update_spaces_and_nls(self, all_lines):
        def is_space(token):
            return token[0] in Text and token[1].replace(' ', '') == ''

        all_line_tokens = []
        for token_list in all_lines:
            token_list_no_spaces = list(filter(lambda tok: not is_space(tok), token_list))
            last_token = token_list_no_spaces[-1]
            if last_token[0] in Text and '\n' in last_token[1]:
                all_line_tokens.append(token_list_no_spaces[:-1] + [(last_token[0], '_NEWLINE_')])
            else:
                all_line_tokens.append(token_list_no_spaces)
        return all_line_tokens

    def update_tokens(self, all_lines):
        all_line_tokens = []
        for token_list in all_lines:
            new_token_list = []
            prev_num = False
            for tok in token_list:
                if tok[0] in Number:
                    prev_num = True
                else:
                    if prev_num and tok[0] in Name and tok[1] == 'j':
                        prev_tok = new_token_list.pop()
                        tok = (prev_tok[0], prev_tok[1] + 'j')
                    prev_num = False
                new_token_list.append(tok)
            new_token_list = list(map(self.choose_token_represent, new_token_list))
            all_line_tokens.append(new_token_list)
        return all_line_tokens

    def choose_token_represent(self, token):
        if token[0] in Name and token[1] != '.':
            return '_NAME_'
        elif token[0] in Number:
            return '_NUMBER_'
        elif token[0] in String:
            return '_STRING_'
        return token[1]

    def update_tokens_with_actual(self, all_lines):
        all_line_tokens = []
        for token_list in all_lines:
            new_token_list = []
            prev_num = False
            for tok in token_list:
                if tok[0] in Number:
                    prev_num = True
                else:
                    if prev_num and tok[0] in Name and tok[1] == 'j':
                        prev_tok = new_token_list.pop()
                        tok = (prev_tok[0], prev_tok[1] + 'j')
                    prev_num = False
                new_token_list.append(tok)
            # Abstract String tokens for now. Will insert them back later
            new_token_list = list(map(lambda x: '_STRING_' if x[0] in String else x[1], new_token_list))
            all_line_tokens.append(new_token_list)
        return all_line_tokens

    def final_cleaning(self, tokens, is_actual):
        tokens.append('_ENDMARKER_')
        tokens = " ".join(tokens)
        tokens = tokens.replace('* *', "**")
        tokens = tokens.replace('= =', "==")
        tokens = tokens.replace('< =', "<=")
        tokens = tokens.replace('> =', ">=")
        tokens = tokens.replace('! =', "!=")
        tokens = tokens.replace('< <', "<<")
        tokens = tokens.replace("> >", ">>")
        tokens = tokens.replace('& &', "&&")
        tokens = tokens.replace('| |', "||")
        tokens = tokens.replace('/ /', "//")
        tokens = tokens.replace('+ =', "+=")
        tokens = tokens.replace('- =', "-=")
        tokens = tokens.replace('/ =', "/=")
        tokens = tokens.replace('* =', "*=")
        tokens = tokens.replace('>> =', ">>=")
        tokens = tokens.replace('<< =', "<<=")
        tokens = tokens.replace('&& =', "&&=")
        tokens = tokens.replace('!! =', "!!=")
        tokens = tokens.replace('// =', "//=")
        tokens = tokens.replace('% =', "%=")
        tokens = tokens.replace('@', "@ ")
        tokens = tokens.replace('@ =', "@=")
        tokens = tokens.replace('| =', "|=")
        tokens = tokens.replace('& =', "&=")
        tokens = tokens.replace('^ =', "^=")
        # tokens = tokens.replace(", ;", ";")
        tokens = tokens.replace(". . .", "...")
        if not is_actual:
            tokens = tokens.replace("not in", "not_in")
            tokens = tokens.replace("is not", "is_not")
            tokens = tokens.replace("- >", "_arrow_")
        else:
            tokens = tokens.replace("- >", "->")
        while tokens.find('_STRING_ _STRING_') >= 0:
            tokens = tokens.replace('_STRING_ _STRING_', '_STRING_')
        while tokens.find(' : _NEWLINE_ _NEWLINE_ ') >= 0:
            tokens = tokens.replace(' : _NEWLINE_ _NEWLINE_ ', ' : _NEWLINE_ ')
        while tokens.find('. _NUMBER_') >= 0:
            tokens = tokens.replace('. _NUMBER_', '_NUMBER_')
        while tokens.find('_NEWLINE_ )') >= 0:
            tokens = tokens.replace('_NEWLINE_ )', ')')
        while tokens.find('_NEWLINE_ ]') >= 0:
            tokens = tokens.replace('_NEWLINE_ ]', ']')
        while tokens.find('_NEWLINE_ }') >= 0:
            tokens = tokens.replace('_NEWLINE_ }', '}')
        # print(tokens.replace('_NEWLINE_ ', '\n'))
        if not is_actual:
            tokens = " ".join(map(lambda t: t if t in self.terminals else '_UNKNOWN_', tokens.split()))
        # print(tokens.replace('_NEWLINE_ ', '\n'))
        return tokens


def read_grammar(grammar_file):
    grammar = ErrorGrammar.load_grammar(grammar_file)
    return grammar


def prog_has_parse(prog, grammar, terminals):
    def run_parse(sentence):
        parser = ErrorEarleyParse(sentence, grammar)
        parser.parse()
        return parser.has_parse(), parser.chart

    lexer = Lexer(terminals)
    # tokenized_prog = lexer.lex(prog)
    # print('-----------------')
    # print(tokenized_prog.replace('_NEWLINE_ ', '\n'))
    # print('-----------------')
    lexed_funs = filter(lambda f: f != '', map(lambda ff: lexer.lex('def ' + ff[1]) if ff[0] > 0 else lexer.lex(ff[1]),  enumerate(prog.split('\ndef '))))
    for lexed in lexed_funs:
        parsed, _ = run_parse(lexed)
        if parsed is None:
            return False
        elif not parsed:
            return False
    return True


def prog_error_rules(prog, grammar, terminals):
    def run_parse(sentence):
        parser = ErrorEarleyParse(sentence, grammar, max_cost=20)
        parser.parse()
        return parser.get_rules(), parser.chart

    lexer = Lexer(terminals)
    # alphabet = grammar.get_alphabet()
    # print(alphabet)
    # tokenized_prog = lexer.lex(prog)
    # print('-----------------')
    # print(tokenized_prog.replace('_NEWLINE_ ', '\n'))
    # print('-----------------')
    # lexed_funs = filter(lambda f: f != '', map(lambda ff: lexer.lex('def ' + ff[1]) if ff[0] > 0 else lexer.lex(ff[1]),  enumerate(prog.split('\ndef '))))
    # error_rules = []
    # for lexed in lexed_funs:
    #     parsed, _ = run_parse(lexed)
    #     if parsed is None:
    #         return []
    #     elif not parsed:
    #         return []
    #     # parse = ErrorEarleyParse(lexed, grammar)
    #     # parse.parse()
    #     # parse.get().pretty_print()
    #     error_rules.extend(list(filter(lambda er: er.lhs.startswith('Err_') or er.lhs == 'InsertErr', parsed)))

    lexed_prog = lexer.lex(prog)
    error_rules = []
    parsed, _ = run_parse(lexed_prog)
    if parsed is None:
        return []
    elif not parsed:
        return []
    # parse = ErrorEarleyParse(lexed_prog, grammar)
    # parse.parse()
    # parse.get().pretty_print()
    # print(parsed)
    error_rules = list(filter(lambda er: er.lhs.startswith('Err_') or er.lhs == 'InsertErr', parsed))
    # for rule in list(filter(lambda er: er.lhs.startswith('Err_') or er.lhs == 'InsertErr' or any(map(lambda r: r.startswith('Err_') or r == 'InsertErr', er.rhs)), parsed)):
    #     print(rule)
    return error_rules


def lexed_prog_has_parse(lexed_prog, grammar):
    def run_parse(sentence):
        parser = ErrorEarleyParse(sentence, grammar)
        parser.parse()
        return parser.get_rules(), parser.chart

    error_rules = []
    parsed, _ = run_parse(lexed_prog)
    if parsed is None:
        return []
    elif not parsed:
        return []
    # parse = ErrorEarleyParse(lexed_prog, grammar)
    # parse.parse()
    # parse.get().pretty_print()
    # print(parsed)
    error_rules = list(filter(lambda er: er.lhs.startswith('Err_') or er.lhs == 'InsertErr', parsed))
    return error_rules


def rule_updates(st):
    if st.rule.lhs.startswith('Err_'):
        return st.rule.lhs.replace('Err_', '')
    elif st.rule.lhs == 'InsertErr':
        return None
    else:
        return st.rule.lhs


def rule_updates_rhs(grammar, st):
    if st.rule.lhs.startswith('Err_'):
        return grammar[st.rule.lhs.replace('Err_', '')][0].rhs[0]
    elif st.rule.lhs == 'InsertErr':
        return None
    else:
        return st.rule.rhs[0]


def rule_updates_repair_operations(grammar, st):
    if st.rule.lhs.startswith('Err_'):
        if st.rule.rhs:
            return '<<$' + grammar[st.rule.lhs.replace('Err_', '')][0].rhs[0]  + '$>>'
        else:
            return '<<+' + grammar[st.rule.lhs.replace('Err_', '')][0].rhs[0] + '+>>'
    elif st.rule.lhs == 'InsertErr':
        return '<<-' + st.rule.rhs[0] + '->>'
    else:
        return st.rule.rhs[0]


def get_repaired_seq_for_1(rule_sequences, grammar):
    rule_seq = sorted(rule_sequences[0], key=lambda st: st.sent_pos)
    abstract_fixed_seq = list(filter(lambda x: x, map(rule_updates, rule_seq)))
    rul_upd = partial(rule_updates_rhs, grammar)
    fixed_seq = list(filter(lambda x: x, map(rul_upd, rule_seq)))
    rul_upd_repair_ops = partial(rule_updates_repair_operations, grammar)
    fixed_seq_ops = list(filter(lambda x: x, map(rul_upd_repair_ops, rule_seq)))
    return (' '.join(abstract_fixed_seq), ' '.join(fixed_seq), ' '.join(fixed_seq_ops), rule_sequences[1], rule_sequences[2])


def fixed_lexed_prog(lexed_prog, grammar, max_cost):
    def run_parse(sentence):
        parser = ErrorEarleyParse(sentence, grammar, max_cost)
        parser.parse()
        return parser.get_fixed_seq(), parser.chart

    parsed, _ = run_parse(lexed_prog)
    if parsed is None:
        return (None, None, None, None, None)
    elif not parsed:
        return (None, None, None, None, None)
    return get_repaired_seq_for_1(parsed, grammar)


def repair_prog(actual_tokens, fix_seq_operations):
    # Reverse program `actual_tokens` for efficient popping
    actual_tkns = actual_tokens
    actual_tkns = list(reversed(actual_tkns.split()))
    repaired = ""
    indents = ''
    if " not_in " in fix_seq_operations:
        fix_seq_operations = fix_seq_operations.replace(" not_in ", " not in ")
    if " is_not " in fix_seq_operations:
        fix_seq_operations = fix_seq_operations.replace(" is_not ", " is not ")
    for tk in fix_seq_operations.split():
        if tk.startswith('<<+'):
            if tk[3:-3] == '_INDENT_':
                indents += ' ' * 4
                repaired += ' ' * 4
            elif tk[3:-3] == '_DEDENT_':
                indents = indents[4:]
                repaired = repaired[:-4]
            elif tk[3:-3] == '_NEWLINE_':
                repaired += '\n'
                repaired += indents
            elif tk[3:-3] in ['pass', 'break', 'continue', 'return', 'yield']:
                flag = (repaired[-2] == ':') if repaired else False
                repaired += '\n'
                repaired += indents
                repaired += (' ' * 4 if flag else '') + tk[3:-3] + " "
            elif tk[3:-3] == '_NAME_':
                repaired += 'simple_name '
            else:
                repaired += tk[3:-3] + " "
        elif tk.startswith('<<$'):
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            if tk[3:-3] == '_INDENT_':
                indents += ' ' * 4
                repaired += ' ' * 4
            elif tk[3:-3] == '_DEDENT_':
                indents = indents[4:]
                repaired = repaired[:-4]
            elif tk[3:-3] == '_NEWLINE_':
                repaired += '\n'
                repaired += indents
            elif tk[3:-3] == '_NAME_':
                repaired += 'simple_name '
            elif tk[3:-3] in ['pass', 'break', 'continue', 'return', 'yield']:
                flag = (repaired[-2] == ':') if repaired else False
                repaired += '\n'
                repaired += indents
                repaired += (' ' * 4 if flag else '') + tk[3:-3] + " "
            else:
                repaired += tk[3:-3] + " "
        elif tk.startswith('<<-'):
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
        elif tk == '_INDENT_':
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            indents += ' ' * 4
            repaired += ' ' * 4
        elif tk == '_DEDENT_':
            # Added checks because of problem with 6100+ in test set
            # It was popping from the empty list
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            indents = indents[4:]
            repaired = repaired[:-4]
        elif tk == '_NEWLINE_':
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            repaired += '\n'
            repaired += indents
        elif tk == '_ENDMARKER_':
            repaired += '\n'
        elif tk == '_arrow_':
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            repaired += "-> "
        elif tk in ['_NAME_', '_STRING_', '_NUMBER_']:
            if actual_tkns:
                repaired += actual_tkns.pop(-1) + " "
            else:
                return repaired
        else:
            if actual_tkns:
                actual_tkns.pop(-1)
            else:
                return repaired
            if tk == '[':
                repaired += tk
            elif tk in ['(', ')', ']', '.', ',', ':'] and len(repaired) > 0:
                if tk == ',' or tk == ')' or tk == ':':
                    repaired = (repaired[:-1] if repaired[-1] == ' ' else repaired) + tk + " "
                else:
                    repaired = (repaired[:-1] if repaired[-1] == ' ' else repaired) + tk
            else:
                repaired += tk + " "
    return repaired


def concretize_seq(seq, grammar):
    tags = seq.split()
    tokens = [grammar[t][0].rhs[0] for t in tags]
    return ' '.join(tokens)


def get_token_list(prog, terminals):
    lexer = Lexer(terminals)
    return lexer.lex(prog)


def get_actual_token_list(prog, terminals):
    lexer = Lexer(terminals)
    return lexer.clean_with_lex(prog)


def main():
    """
    Main.
    """

    parser_description = ("Runs the Earley parser according to a given "
        "grammar.")

    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument('draw', nargs='?', default=False)
    parser.add_argument('grammar_file', help="Filepath to grammer file")
    parser.add_argument('input_program', help="The input program to parse")
    parser.add_argument('--show-chart', action="store_true")

    args = parser.parse_args()

    grammar = ErrorGrammar.load_grammar(args.grammar_file)
    terminals = grammar.get_alphabet()

    def run_parse(sentence):
        parse = ErrorEarleyParse(sentence, grammar)
        parse.parse()
        return parse.get_fixed_seq(), parse.chart

    program_path = Path(args.input_program)
    # # Strip the sentence of any puncutation.
    # stripped_sentence = sentence
    # for p in string.punctuation:
    #     stripped_sentence = stripped_sentence.replace(p, '')

    input_program = program_path.read_text()
    # print(input_program)
    lexer = Lexer(terminals)
    tokenized_prog = lexer.lex(input_program)
    # print(parse(input_program))
    # print(asttokens.ASTTokens(input_program, parse=True).tokens)
    print(tokenized_prog.replace('_NEWLINE_ ', '\n'))
    print('-----------------')
    # lexed_funs = filter(lambda f: f != '', map(lambda ff: lexer.lex('def ' + ff[1]) if ff[0] > 0 else lexer.lex(ff[1]),  enumerate(input_program.split('\ndef '))))
    # for lexed in lexed_funs:
    #     parsed, _ = run_parse(lexed)
    #     if parsed is None:
    #         print(False)
    #         pass
    #     elif not parsed:
    #         print(False)
    #         pass
    #     # print(parsed)
    #     # parsed.pretty_print()
    # print(True)
    parsed, chart = run_parse(tokenized_prog)
    if args.show_chart:
        print(chart)
        print('\n')
    if parsed is None:
        print(input_program + '\n')
        print('Failed parsing...')
    elif not parsed:
        print(input_program + '\n')
        print('Failed parsing...')
    else:
        if args.draw:
            parsed.draw()
        else:
            fixed_seq_ops = get_repaired_seq_for_1(parsed, grammar)[2]
            actual_tokns = get_actual_token_list(input_program, terminals)
            print(repair_prog(actual_tokns, fixed_seq_ops) + '\n')
            print("Successful parse!")
            # parsed.pretty_print()

if __name__ == '__main__':
    main()
