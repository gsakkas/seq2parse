import re

#This part of the code removes comments from code
def removeComments(string):
    string = re.sub(re.compile("\"\"\".*?\"\"\"", flags=re.DOTALL) , "\n" , string)
    string = re.sub(re.compile("\'\'\'.*?\'\'\'", flags=re.DOTALL) , "\n" , string)
    string = re.sub(re.compile("#.*?\n" ) , "\n" , string)
    return string
