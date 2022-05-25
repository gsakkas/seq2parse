import sys
from os.path import join
from pathlib import Path
from ast import parse
import json
import tqdm
from earleyparser import read_grammar, prog_has_parse

# pr = {"bad": "def check_rush_hour(day, hour, minute:\n    if day== 'Monday' or day=='Tuesday' or day=='Wednesday' or day=='Thursday' or day=='Friday':\n        if hour>= 5:\n            if hour<= 9:\n                if minute= 0 or minute=1 or minute=2 or minute=3 or minute=4 or minute=5 or minute=6:\n                    return True\n                else:\n                    return False\n        if hour>= 15:\n            if hour<= 19:\n                elif minute= 0:\n                    return True\n                else:\n                    return False\n    else:\n        return False\n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "fix": "def check_rush_hour(day, hour, minute):\n    if day== \"Monday\" or day==\"Tuesday\" or day==\"Wednesday\" or day==\"Thursday\" or day==\"Friday\":\n        if hour>= 5:\n            if hour<= 9:\n                if minute >=0:\n                    if minute<=30:\n                        return True\n                    else:\n                        return False\n        elif hour>= 15:\n            if hour<= 19:\n               return True\n            else:\n                return False\n    else:\n        return False\n        \n    \n        \n        \n        \nprint(check_rush_hour(\"Monday\", 6, 0)) ", "index": 4818, "fixIndex": 4830, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 1205}

# pr = {"bad": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print i\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "fix": "def sumtu(tu):\n    for n in range(len(tu)):\n        i = 0\n        i = tu[n]+i\n        print (i)\n        return \ntu = (1,2,5,3,6)\nsumtu(tu)", "index": 6, "fixIndex": 7, "errMsg": "SyntaxError", "isConsecutive": True, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 15}

# pr = {"bad": "#PROMEDIO FINAL\n>>> #PROMEDIO FINAL\n>>> print(\"\u00bfCuantas unidades son?:\")\n\u00bfCuantas unidades son?:\n>>> n=int(input())\nsuma=0\n>>> for i in range (n):\n    print(\"Dame la calificacion\",i+1)\n    cal=int(input())\n#Next i", "fix": "#PROMEDIO FINAL\n#PROMEDIO FINAL\nprint(\"\u00bfCuantas unidades son?:\")\n\nn=int(input())\nsuma=0\nfor i in range (n):\n    print(\"Dame la calificacion\",i+1)\n    cal=int(input())\n#Next i", "index": 575, "fixIndex": 586, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": ["7", "10", "10", "8", "8", "9", "8", "8"], "mergedInput": [], "duration": 137}

# pr = {"bad": "names = [\"Bob\", \"Mark\", \"Jack\", \"Luise\", \"John\",\\\n    \"Amily\", \"Mary\", \"Rose\"]\n    \ndo_not_come =\"Luise\"\nnames.remove(do_not_come)\nfor name in names:\n\tprint(name\\\n\t+ \" I would like to invite you to have dinner with me.\\n\")\n\nprint(do_not_come + \" can't show up.\\n\")\n\nnames[2] = \"Harbor\"\nnames.insert(0, \"Bill\")\nnames.insert(4, \"Bean\")\nnames.append(\"Edwerd\")\nname_count = len(names)\nname_count = len(names)\nwhile name_count > 2:\n\tname_remove = names.pop()\n\tprint(\"Sorry, \" + name_remove + \"there's no more place. \"\\\n\t\t\"I'll invite you next time.\")\n\tname_count -= 1\nfor name in names:\n\tprint(name + \" I would like to invite you to have dinner with me.\\n\")\n\t\n\tnames.del[:]\nprint(names)", "fix": "names = [\"Bob\", \"Mark\", \"Jack\", \"Luise\", \"John\",\\\n    \"Amily\", \"Mary\", \"Rose\"]\n    \ndo_not_come =\"Luise\"\nnames.remove(do_not_come)\nfor name in names:\n\tprint(name\\\n\t+ \" I would like to invite you to have dinner with me.\\n\")\n\nprint(do_not_come + \" can't show up.\\n\")\n\nnames[2] = \"Harbor\"\nnames.insert(0, \"Bill\")\nnames.insert(4, \"Bean\")\nnames.append(\"Edwerd\")\nname_count = len(names)\nname_count = len(names)\nwhile name_count > 2:\n\tname_remove = names.pop()\n\tprint(\"Sorry, \" + name_remove + \"there's no more place. \"\\\n\t\t\"I'll invite you next time.\")\n\tname_count -= 1\nfor name in names:\n\tprint(name + \" I would like to invite you to have dinner with me.\\n\")\n\t\nprint(names)", "index": 17334, "fixIndex": 17337, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 270}

#### Some bad-fix error examples for error correcting parser

# pr = {"bad": "class Singer :\n    msg = 'Shut up and '\n    count = 1\n    lst = []\ndef __init__ ( self , name , x ):\n    self .x = x\ndef sing ( self ):\n    Singer . count = Singer . count + 1\n    print ( self . msg + self . x)\n    self . lst . append ( self . count )\nclass Group ( Singer ):\ndef __init__ ( self , s1 , s2 ):\n    self . s1 = s1\n    self . s2 = s2\ndef sing ( self ):\n    self . count = self . count + 1\n    self . s1 . sing ()\n    self . s2 . sing ()\n    self . s1 .x , self . s2 .x = self . s2 .x , self . s1 .x\n    self . lst . append ( self . count )\n    w_moon = Singer ( \u2019 WALK THE MOON \u2019, \u2019 dance \u2019)\n    rihanna = Singer ( 'Rihanna' , 'drive')\n    w_ri = Group ( w_moon , rihanna )\n    w_ri . lst = []", "fix": "class Singer :\n    msg = 'Shut up and '\n    count = 1\n    lst = []\ndef __init__ ( self , name , x ):\n    self .x = x\ndef sing ( self ):\n    Singer . count = Singer . count + 1\n    print ( self . msg + self . x)\n    self . lst . append ( self . count )\nclass Group ( Singer ):\n    def __init__ ( self , s1 , s2 ):\n        self . s1 = s1\n    self . s2 = s2\n    def sing ( self ):\n        self . count = self . count + 1\n        self . s1 . sing ()\n        self . s2 . sing ()\n        self . s1 .x , self . s2 .x = self . s2 .x , self . s1 .x\n        self . lst . append ( self . count )\n        w_moon = Singer ( 'WALK THE MOON' , 'dance')\n    rihanna = Singer ( 'Rihanna' , 'drive')\n    w_ri = Group ( w_moon , rihanna )\n    w_ri . lst = []", "index": 1096, "fixIndex": 1098, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 36}

# pr = {"bad": "import math\ndef main():\n    x = int(input())\n    factorize(x)\n\ndef factorize(x):\n    print(x, 'is ', end=' ')\n    ie = math.sqrt(x)\n    ie = int(ie)\n    for i in range(2, ie):\n        if (x % i == 0):\n            while (x > 0):\n                if (x % i == 0):\n                    print(i, end=' ')\n                    x /= i\n                    t = 1            else:\n                    i += 1\n                    \n                if i > iex\n<                       br                    \n                \n                           t = 0\n     if t ==       ', end=' ')\n                    \n                t = 0\n    else:\n        print('pirme')\n\nif __name__ == '__main__':\n    main()\n            \n", "fix": "import math\ndef main():\n    x = int(input())\n    factorize(x)\n\ndef factorize(x):\n    print(x, 'is ', end=' ')\n    ie = math.sqrt(x)\n    ie = int(ie)\n    for i in range(2, ie):\n        if (x % i == 0):\n            while (x > 0):\n                if (x % i == 0):\n                    print(i, end=' ')\n                    x /= i\n                if x == 1:\n                    i += 1\n                    \n                \n                    \n                t = 0\n    else:\n        print('pirme')\n\nif __name__ == '__main__':\n    main()\n            \n", "index": 3065, "fixIndex": 3077, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": ["19877"], "mergedInput": [], "duration": 830}

# pr = {"bad": "temperature = None\n\ndef addTemperature():\n    temperature = float(input(insert a temperature:'))\n    return temperature + 30\n\naddTemperature()", "fix": "temperature = None\n\ndef addTemperature():\n    temperature = float(input('insert a temperature:'))\n    return temperature + 30\n\naddTemperature()", "index": 3794, "fixIndex": 3798, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": ["30"], "mergedInput": [], "duration": 208}

# pr = {"bad": "message=\"hello world\"\n\nfor x in range(len(message)):\n        if (ord(message[x]))==32 or (ord(message[x])>=48 and ord(message[x])<=57):\n            continue\n        elif (ord(message[x])==39 or ord(message[x])==34):\n            message=message.replace(message[x],\"\")\n        message=message.replace(message[x],chr((ord(message[x]))+key))\n    print(message)", "fix": "message=\"hello world\"\n\nfor x in range(len(message)):\n        if (ord(message[x]))==32 or (ord(message[x])>=48 and ord(message[x])<=57):\n            continue\n        elif (ord(message[x])==39 or ord(message[x])==34):\n            message=message.replace(message[x],\"\")\n        message=message.replace(message[x],chr((ord(message[x]))+key))\n        print(message)", "index": 12508, "fixIndex": 12511, "errMsg": "SyntaxError", "isConsecutive": False, "isFinal": False, "badInput": [], "fixInput": [], "mergedInput": [], "duration": 60}


not_fixed_index = [466, 467, 468, 28432, 28433, 28434, 234049]

def has_parse(l):
    dct = json.loads(l)
    # print('# -----------------')
    # print('Bad')
    # print('# -----------------')
    # print(dct['bad'])
    # print('# -----------------')
    # print('Fix')
    # print('# -----------------')
    # print(dct['fix'])
    # print('# -----------------')
    if dct["index"] in not_fixed_index and dct["fixIndex"] in [470, 28437, 234051]:
        pass
    elif dct["errMsg"] == "SyntaxError":
        parse_fix = prog_has_parse(dct['fix'], grammar)
        parse_bad = prog_has_parse(dct['bad'], grammar)
        return (parse_bad, parse_fix)


if __name__ == "__main__":
    grammar_file = sys.argv[1]
    dataDir = Path(sys.argv[2])
    outDir = Path(sys.argv[3])

    grammar = read_grammar(grammar_file)

    parses_bad = []
    parses_fix = []
    bads_allowed = 200
    bads_found = 0
    for partPath in tqdm.tqdm(list(dataDir.glob('part_15'))):
        goodPath = partPath / "goodPairs.jsonl"
        for line in tqdm.tqdm(goodPath.read_text().strip().split('\n')):
        # for idx, line in tqdm.tqdm(enumerate(goodPath.read_text().strip().split('\n'))):
            # if idx < 36000:
            #     continue
            bruh = has_parse(line)
            if bruh:
                parse_bad, parse_fix = bruh
                if not parse_fix:
                    dct = json.loads(line)
                    print('Fix')
                    # print('Index', idx)
                    print('# -----------------')
                    print(dct['fix'])
                    print('# -----------------')
                    print(line)
                    print('# -----------------')
                    # sys.exit(1)
                if parse_bad and bads_found < bads_allowed:
                    dct = json.loads(line)
                    try:
                        parse(dct['bad'])
                    except SyntaxError as err:
                        print('# Bad')
                        # print('Index', idx)
                        print('# Error Message =', err.msg)
                        print('# -----------------')
                        print(dct['bad'])
                        print('# -----------------')
                        # print(line)
                        # print('# -----------------')
                        # if bads_found > bads_allowed:
                        #     sys.exit(1)
                        bads_found += 1
                parses_bad.append(parse_bad)
                parses_fix.append(parse_fix)
        failPath = partPath / "failPairs.jsonl"
        for line in tqdm.tqdm(failPath.read_text().strip().split('\n')):
            bruh = has_parse(line)
            if bruh:
                parse_bad, parse_fix = bruh
                if not parse_fix:
                    dct = json.loads(line)
                    print('Fix')
                    # print('Index', idx)
                    print('# -----------------')
                    print(dct['fix'])
                    print('# -----------------')
                    # print(line)
                    # print('# -----------------')
                    # sys.exit(1)
                if parse_bad and bads_found < bads_allowed:
                    dct = json.loads(line)
                    try:
                        parse(dct['bad'])
                    except SyntaxError as err:
                        print('# Bad')
                        # print('Index', idx)
                        print('# Error Message =', err.msg)
                        print('# -----------------')
                        print(dct['bad'])
                        print('# -----------------')
                        # print(line)
                        # print('# -----------------')
                        # if bads_found > bads_allowed:
                        #     sys.exit(1)
                        bads_found += 1
    total_parses = len(parses_fix)
    bads = len(list(filter(lambda x: x, parses_bad)))
    fixs = len(list(filter(lambda x: x, parses_fix)))
    with open(join(outDir, "EarleyParses.txt"), "w") as dataset_file:
        print("Dataset size:", total_parses)
        print("Bad Dataset Parsed (%):", bads * 100.0 / total_parses)
        print("Fixed Dataset Parsed (%):", fixs * 100.0 / total_parses)
        dataset_file.write("Dataset size:" + str(total_parses) + "\n")
        dataset_file.write("Bad Dataset Parsed (%):" + str(bads * 100.0 / total_parses) + "\n")
        dataset_file.write("Fixed Dataset Parsed (%):" + str(fixs * 100.0 / total_parses) + "\n")
    # l = json.dumps(pr)
    # dct = json.loads(l)
    # print(prog_has_parse(dct['bad'], grammar))
    # print(prog_has_parse(dct['fix'], grammar))

# 2/8/2021 - 18:49 => Bad: 3.70 %, Fix: 45.65 %
# 2/8/2021 - 19:41 => Bad: 5.66 %, Fix: 63.06 %
# 2/8/2021 - 20:45 => Bad: 6.09 %, Fix: 71.40 %
# 2/9/2021 - 0:46 => Bad: 8.89 %, Fix: 89.30 %
# 2/9/2021 - 1:36 => Bad: 9.23 %, Fix: 91.83 %
# 2/9/2021 - 20:56 => Bad: 10.56 %, Fix: 95.51 %
# 2/11/2021 - 20:26 => Bad: 10.36 %, Fix: 97.01 %
# 2/11/2021 - 22:18 => Bad: 10.67 %, Fix: 97.78 %
# 2/12/2021 - 1:58 => Bad: 10.70 %, Fix: 100.00 %
# 2/27/2021 - 21:59 => Bad: 10.82 %, Fix: 100.00 %
