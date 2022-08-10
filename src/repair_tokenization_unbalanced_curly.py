import re
from tokenizer import lex_file


#we added 6 tokens to keyword list to avoid an extra list
###########################################################################################################

keyWord = [
    'False',
    'await',
    'else',
    'import',
    'pass',
    'None',
    'break',
    'except',
    'in',
    'raise',
    'True',
    'class',
    'finally',
    'is',
    'return',
    'and',
    'continue',
    'for',
    'lambda',
    'try',
    'as',
    'def',
    'from',
    'nonlocal',
    'while',
    'assert',
    'del',
    'global',
    'not',
    'with',
    'async',
    'elif',
    'if',
    'or',
    'yield',
    'expression',
    '__paren_expression__',
    '__condition__',
    # '__indent__',
    # '__broken_indent__',
    # '__new_line__',
    ":",
    "{",
    "}"
    ]

def reconstruct_and_tokenize(final_text, expression, paren, simple_name, conds, indent):
    ww = final_text.split()
    jj = 0
    for j in range(len(ww)):
        while ww[j] == "simple_name":
            ww[j] = ww[j].replace("simple_name", simple_name[jj])
            jj = jj + 1
    # print(ww)
    jj = 0
    for j in range(len(ww)):
        while ww[j] == "expression":
            ww[j] = ww[j].replace("expression", expression[jj] + " \n")
            jj = jj + 1
    jj = 0
    for j in range(len(ww)):
        if ww[j].find("__paren_expression__") != -1:
           r = ww[j].split("__paren_expression__")
           #print(r)
           m = 0
           p = ""
           while m<len(r)-1:
               p = p + r[m] + "( " + paren[jj] + " )" + " "
               m = m + 1
               jj = jj + 1
           p = p[0:len(p)-1]
           p = p+r[len(r)-1]
           ww[j] = p
    jj = 0
    for j in range(len(ww)):
        while ww[j] == "__condition__":
            ww[j] = ww[j].replace("__condition__", conds[jj])
            jj = jj + 1
    # for j in range(len(ww)):
    #     # if ww[j] == "__indent__" or ww[j] == "__broken_indent__":
    #     #     ww[j] = ww[j].replace("__indent__", indent)
    #     if ww[j] == "__new_line__":
    #         ww[j] = ww[j].replace("__new_line__", "\n")
    # print(ww)
    # print("===========HERE==========")
    final_text = ww
    j = 0
    p = ""
    while j < len(final_text):
        if final_text[j].endswith('\n') or final_text[j] == ':' or final_text[j] == "{" or final_text[j] == "}":
            if len(p) > 0:
                final_text.insert(j, p[0:len(p) - 1])
                p = ""
            j = j + 1
            continue
        else:
            p = p + final_text[j] + " "
            del final_text[j]
    if len(p) > 0:
        final_text.insert(j, p[0:len(p) - 1])
    # print(final_text)
    # print("===========HERE==========")
    j = 0
    while j < len(final_text)-1:
        if final_text[j].endswith('\n') == False and final_text[j] != ':' and final_text[j] != "{" and final_text[j] != "}":
            if final_text[j+1] != "{" and final_text[j+1] != "}":
                final_text[j] = final_text[j] + " " + final_text[j + 1]
                del final_text[j + 1]
            else:
                j = j + 1
        else:
            j = j + 1

    all_lines_temp = []
    # print(final_text)
    # print("===========HERE==========")
    for i in range(0, len(final_text)):
        ln = final_text[i]
        #print(ln)
        if ln.find("+") == -1:
            if ln.find("\"\\\"") != -1:
                ln=ln.replace("\"\\\"", "\" \" ")
            if ln.find("\\\"\"") != -1:
                ln = ln.replace("\\\"\"", " \" \"")
            else:
                ln = ln.replace("\\\"", "")
        else:
            ln = ln.replace("\\\"", "")
        # print(ln)
        ln = re.sub(r'\"(.+?)\"', "\" double_quote \"", ln)
        ln = re.sub(r'\'(.+?)\'', "\' single_quote \'", ln)
        # print("1.", ln)
        if ln.startswith("import") == False:
            # print("WHAT")
            ln = lex_file(ln)
        else:
            ln = ln.replace(".", " . ")
            ln = ln.replace("*", "* ")
            ln = ln.replace(";", " ;")
            ln = ln.replace("  ", " ")
            ln = ln + " "
        # print("2.", ln)
        ln = re.sub('[ ][0-9][a-zA-Z0-9]*', " 0", ln)
        # print(ln)
        ln = re.sub(r'\"(.+?)\"', "\" double_quote \"", ln)
        ln = re.sub(r'\'(.+?)\'', "\' single_quote \'", ln)
        # print(ln)
        ln = ln.replace("  ", " ")
        if ln.find("<") != -1 :
            ln = ln.replace('>>>', "> > >")
        else:
            ln = ln.replace('> > >', ">>>")
        ln = ln.replace('< < <', "<<<")
        ln = ln.replace("+ +", "++")
        ln = ln.replace('- -', "--")
        ln = ln.replace('= =', "==")
        ln = ln.replace('< =', "<=")
        ln = ln.replace('> =', ">=")
        ln = ln.replace('! =', "!=")
        ln = ln.replace('< <', "<<")
        if ln.find("<") !=-1 :
            ln = ln.replace('>>', "> >")
        else:
            ln = ln.replace("> >", ">>")
        ln = ln.replace('& &', "&&")
        ln = ln.replace('| |', "||")
        ln = ln.replace('+ =', "+=")
        ln = ln.replace('- =', "-=")
        ln = ln.replace('/ =', "/=")
        ln = ln.replace('* =', "*=")
        ln = ln.replace('>> =', ">>=")
        ln = ln.replace('<< =', "<<=")
        ln = ln.replace('&& =', "&&=")
        ln = ln.replace('!! =', "!!=")
        ln = ln.replace('% =', "%=")
        ln = ln.replace('@', "@ ")
        ln = ln.replace(", ;", ";")
        ln = ln.replace("\"\"", "\" double_quote \"")
        ln = ln.strip()
        all_lines_temp.append(ln)
    return all_lines_temp





#separate curly braces to new lines if not
#note that, the pygments tokenizer does not work perfectly and the code
#also has very incostent formating. That's why we have a list of replace
#instructions to repair the tokenization process.


def repair_unbalanced(lines, expression):
    conds = list(map(lambda m: m[2], re.findall(r'(if|elif|else|for|while|with|async with)( +?)(.+?)( *?):', lines)))
    # conds += list(map(lambda m: m[2], re.findall(r'(try|finally)( *?):', lines)))
    conds += list(map(lambda m: m[2], re.findall(r'(except)( *?)(.*?)( *?):', lines)))
    # lines = re.sub(r'\s*?(\\)\s*?\n', " ", lines)
    lines = re.sub(r'if( +?)(.+?)( *?):', "if __condition__ :", lines)
    lines = re.sub(r'elif( +?)(.+?)( *?):', "elif __condition__ :", lines)
    lines = re.sub(r'else( *?):', "else :", lines)
    lines = re.sub(r'for( +?)(.+?)( *?):', "for __condition__ :", lines)
    lines = re.sub(r'while( +?)(.+?)( *?):', "while __condition__ :", lines)
    lines = re.sub(r'try( *?):', "try :", lines)
    lines = re.sub(r'except( *?)(.*?)( *?):', "except __condition__ :", lines)
    lines = re.sub(r'finally( *?):', "finally __condition__ :", lines)
    lines = re.sub(r'with( +?)(.+?)( *?):', "with __condition__ :", lines)
    lines = re.sub(r'async with( +?)(.+?)( *?):', "async with __condition__ :", lines)
    # while "(" in lines:
    paren = re.findall(r'\(([^\(]+?)\)', lines)
    lines = re.sub(r'\(([^\(]+?)\)', " __paren_expression__ ", lines)
    # print(lines)
    # lines = lines.replace("\n", "\n __new_line__ \n")
    #remove expression and store... and also remove the non token lines
    #abtract the input program here
    # lines = lines.split("\n")
    # for j in range(0, len(lines)):
    #     lines[j] = lines[j].strip()
    #     if lines[j].startswith("__indent__"):
    #         if lines[j].endswith("__indent__") or lines[j].endswith("__broken_indent__") or lines[j].endswith(" "):
    #             lines[j] = ""
    # lines = "\n __new_line__ \n".join(lines)
    # lines = lines.replace("__indent__", " __indent__ \n")
    # lines = lines.replace("__broken_indent__", " __broken_indent__ \n")
    lines = lines.split("\n")
    for j in range(0, len(lines)):
        lines[j] = lines[j].strip()
    j = 0
    while j < len(lines):
        x = re.search('[a-zA-Z{}]', lines[j])
        if x == None:
            del lines[j]
        else:
            j = j + 1
    # block_kwords = ["if", "elif", "else", "for", "while", "try", "except", "finally", "with", \
    #                 "async", "def", "class", "__indent__", "__new_line__", "__broken_indent__"]
    block_kwords = ["if", "elif", "else", "for", "while", "try", "except", \
                    "finally", "with", "async", "def", "class", "{", "}"]
    j = 0
    while j < len(lines):
        if any(map(lines[j].startswith, block_kwords)):
            j = j + 1
            continue
        expression.append(lines[j])
        lines[j] = "expression"
        j = j + 1
    text = ""
    for j in range(len(lines)):
        text = text + lines[j] + " "
    text = text[0:len(text) - 1]
    ln = lex_file(text)
    ln = re.sub('[ ][0-9][a-zA-Z0-9]*', " 0", ln)
    ln = re.sub(r'\"(.+?)\"', "\" double_quote \"", ln)
    ln = re.sub(r'\'(.+?)\'', "\' single_quote \'", ln)
    ln = ln.replace("  ", " ")
    if ln.find("<") != -1:
        ln = ln.replace('>>>', "> > >")
    else:
        ln = ln.replace('> > >', ">>>")
    ln = ln.replace('< < <', "<<<")
    ln = ln.replace("+ +", "++")
    ln = ln.replace('- -', "--")
    ln = ln.replace('= =', "==")
    ln = ln.replace('< =', "<=")
    ln = ln.replace('> =', ">=")
    ln = ln.replace('! =', "!=")
    ln = ln.replace('< <', "<<")
    if ln.find("<") != -1:
        ln = ln.replace('>>', "> >")
    else:
        ln = ln.replace("> >", ">>")
    ln = ln.replace('& &', "&&")
    ln = ln.replace('| |', "||")
    ln = ln.replace('+ =', "+=")
    ln = ln.replace('- =', "-=")
    ln = ln.replace('/ =', "/=")
    ln = ln.replace('* =', "*=")
    ln = ln.replace('>> =', ">>=")
    ln = ln.replace('<< =', "<<=")
    ln = ln.replace('&& =', "&&=")
    ln = ln.replace('!! =', "!!=")
    ln = ln.replace('% =', "%=")
    ln = ln.replace('@', "@ ")
    ln = ln.replace(", ;", ";")
    ln = ln.replace("\"\"", "\" double_quote \"")
    ln = ln.strip()
    simple_name = []
    text = text.split()
    final_text = ""
    for j in range(0, len(text)):
        flag = 1
        for k in range(0, len(keyWord)):
            if text[j] == keyWord[k]:
                flag = 0
                break
        if flag == 1:
            final_text = final_text + "simple_name" + " "
            simple_name.append(text[j])
        else:
            final_text = final_text + text[j] + " "

    final_text = final_text[0:len(final_text) - 1]
    return final_text, expression, simple_name, paren, conds
