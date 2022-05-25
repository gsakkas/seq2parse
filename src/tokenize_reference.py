import re
from comment_remover import removeComments
from tokenizer import lex_file

# This part of the code tokenizes the reference program using pygments and repairs the tokenization. Since the code has syntactical error,
# we could not use Eclipse JDT for all instances
###########################################################################################################

def tokenize_ref(lines):
    all_lines = []
    lines = lines.replace("{", " { ")
    lines = lines.replace("}", " } ")
    lines = removeComments(lines)
    lines = re.sub(r'\"(.+?)\"', "\" double_quote \"",lines)
    lines = re.sub(r'\'(.+?)\'', "\' single_quote \'",lines)
    lines = lines.replace(" if", "\nif")
    lines = lines.replace(" elif", "\nelif")
    lines = lines.replace(" else", "\nelse")
    lines = lines.replace(" for", "\nfor")
    lines = lines.replace(" while", "\nwhile")
    lines = lines.replace(" try", "\ntry")
    lines = lines.replace(" except", "\nexcept")
    lines = lines.replace(" finally", "\nfinally")
    lines = lines.replace(" with", "\nwith")
    lines = lines.replace(" async with", "\nasync with")
    lines = lines.replace("{\n", "\n{\n")
    lines = lines.replace("\n{", "\n{\n")
    lines = lines.replace("\n}", "\n}\n")
    lines = lines.replace("}\n", "\n}\n")
    w = lines.split("\n")
    for ln in w:
        ln = ln.replace("\n", "")
        ln = ln.replace("\t", " ")
        ln = ln.strip()
        if ln != "":
            all_lines.append(ln)

        else:
            continue
    all_lines_temp = []
    for i in range(0,len(all_lines)):
        ln = all_lines[i]

        if ln.find("+") != -1:
            ln = ln.replace("\\\"", "")
        else:
            ln = ln.replace("\\\"", "")
        ln = re.sub(r'\"(.+?)\"', "\" double_quote \"", ln)
        ln = re.sub(r'\'(.+?)\'', "\' single_quote \'", ln)

        if ln.startswith("import") == False:
            ln = lex_file(ln)
        else:
            ln = ln.replace(".", " . ")
            ln = ln.replace("*", "* ")
            ln = ln.replace(";", " ;")
            ln = ln.replace("  ", " ")
            ln = ln+" "
        ln = re.sub('[ ][0-9][a-zA-Z0-9]*', " 0", ln)

        if ln.find("+") != -1:
            ln = ln.replace("\\\"", "")
        else:
            ln = ln.replace("\\\"", "")
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
        ln = ln.replace("\"\"", "\" double_quote \"")
        ln = ln.strip()
        all_lines_temp.append(ln)
    all_lines = all_lines_temp
    after_token = []
    for i in range(0, len(all_lines)):
        p = all_lines[i]
        words = p.split()
        for w in words:
            after_token.append(w)
    return after_token
