lines = 'AIRS|100|001|Leadership Laboratory|Colleen Delawder|Laboratory|0|false|true|false|false|false|1530|1730|Physics Bldg 203|46|100\
AIRS|1200|001|The Foundations of the U.S. Air Force|Christopher Ulman|Lecture|1|false|true|false|false|false|1300|1350|Maury Hall 115|7|30\
AIRS|1200|002|The Foundations of the U.S. Air Force|Christopher Ulman|Lecture|1|false|false|false|true|false|1300|1350|Shannon House 107|8|35\
AIRS|2200|001|The Evolution of Air and Space Power|Daniel Brown|Lecture|1|false|true|false|false|false|1400|1450|Physics Bldg 205|8|40\
AIRS|2200|002|The Evolution of Air and Space Power|Daniel Brown|Lecture|1|false|false|false|true|false|1400|1450|Shannon House 109|4|20\
AIRS|3200|001|Concepts of Air Force Leadership and Management|Colleen Delawder|Lecture|3|false|true|false|false|false|1230|1345|Shannon House 107|6|25\
AIRS|3200|002|Concepts of Air Force Leadership and Management|Colleen Delawder|Lecture|3|false|false|false|false|true|1230|1500|Astronomy Bldg 139|3|25\
AIRS|4200|001|National Security Affairs/Preparation for Active Duty|Patrick Donley|Lecture|3|false|true|false|false|false|1230|1345|Gibson Hall 242|11|30\
AIRS|4200|002|National Security Affairs/Preparation for Active Duty|Patrick Donley|Lecture|3|false|false|false|true|false|1630|1900|Astronomy Bldg 139|9|30'

for p in lines:
    n1 = lines[y]
    n = n1.split('|')
    n3 = n[4]
    cells.append(n3)
    cells2 = [s.strip('0123456789+') for s in cells]
    y += 1
print(cells)
