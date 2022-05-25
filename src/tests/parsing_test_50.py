"""Tools for word trees.

Authors:
    TODO: ADD YOUR NAME HERE
    Stefan Woelfl <woelfl@cs.uni-freiburg.de>
    Thorsten Engesser <engesser@cs.uni-freiburg.de>
    Tim Schulte <schultet@cs.uni-freiburg.de>

"""

LETTERS = "abcdefghijklmnopqrstuvwxyz" + \
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
          "Ã„ÃœÃ–Ã¤Ã¼Ã¶ÃŸ"


def next_word(s):
    """Return the first word (or None) of an input string s and the rest of it.

    Args:
        s (str): The input string.

    Returns:
        (str or None, str): The first word and the rest of s.

    Examples:
    >>> next_word('asdf asdf xyqr')
    ('asdf', ' asdf xyqr')
    >>> next_word('echatsteinschalosch')
    ('echatsteinschalosch', '')
    >>> next_word('spam&&ham  ham...')
    ('spam', '&&ham  ham...')

    Leading whitespaces and other invalid characters are removed:
    >>> next_word(' &foo @bar')
    ('foo', ' @bar')

    If the input string contains no valid word, None is returned instead:
    >>> next_word('$$$')
    (None, '')
    >>> next_word('')
    (None, '')

    """
    skip = 0
    for c in s:
        if c in LETTERS:
            break
        else:
            skip += 1
    if skip == len(s):
        return None, ''

    take = 0
    for i in range(skip, len(s)):
        if s[i] in LETTERS:
            take += 1
        else:
            break

    return s[skip:skip+take], s[skip+take:]


# 5.3 (a)
def word_tree(s):
    """Return a word tree for an input string s.

    Args:
        s (str): The input string.

    Returns:
        None or [tree, str, int, tree]: A list representation of a tree.

    """
    result=[]
    if s == "":
        return(None)
    else:
        # Wörter zählen
        s_split = s.split()
        n=len(s_split)
        y = 0
        z = 0
        while y < n:
            if s_split[y] > result[z]:
                result.append(s_split[y])
                y+1
            elif s_split[y] < result[z]:
                result.insert(0,s_split[y])
                y=y+1
                z=z+1
        return(result)
# 5.3 (b)
def word_freq(tree, word):
    """Return the occurence count of a word in a word tree.

    Args:
        tree (tree): The word tree.
        word (string): The word to search for.

    Returns:
        int: Number of word occurences.

    """
    pass  # TODO: implement


# 5.3 (c)
def print_tree(tree):
    """Print word tree in in-order.

    Args:
        tree (tree): The word tree.

    Returns:
        None

    """
    pass  # TODO: implement
word_tree("Hallo du Depp wie geht es dir")
