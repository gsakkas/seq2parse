import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Text, String, Name


# This part of the code tokenizes the program using pygments. Since the code has syntactical error,
# we could not use Eclipse JDT for all instances
###########################################################################################################

def lex_file(content):
    temp = ""
    donothing = 1
    lexer = get_lexer_by_name("python")
    tokens = pygments.lex(content, lexer)
#     print(list(pygments.lex(content, lexer)))
    inString = False
    for ttype, value in tokens:
        # print(ttype, value, inString)
        if ttype in String.Doc:
                continue
        if str(ttype).startswith("Token.Comment"):
                for i in range(value.count("\n")):
                        donothing=1
        # if ttype in String:
        #         # print(ttype in String)
        #         if inString: continue
        #         inString = True
        # else:
        #         inString = False
        if value == ' ':
                continue
        elif value == '\n':
                continue
        if ttype in Name.Namespace:
                parts = value.split(".")
                if len(parts) > 1:
                    for ix, p in enumerate(parts):
                        if ix < len(parts):
                                donothing = 1
                        else:
                                print(p.replace(":", ""))
                        if ":" in p:
                                donothing = 1
                        else:
                                donothing = 1
                    continue
        if ttype in Comment.Preproc or ttype in Comment.PreprocFile:
                parts = value.split(" ")
                if len(parts) > 1:
                        for p in parts:
                                p = p.strip()
                if len(p) > 0:
                        print("%s\t" % p)
                continue
        elif ttype in Comment or ttype in Text:
                continue
        value = value.replace("\t", "\\t").strip()
        if len(value) > 0:
                temp = temp + " " + value + " "
    temp = temp[1:len(temp)]
#     print(list(pygments.lex(content, lexer)))
    return temp



