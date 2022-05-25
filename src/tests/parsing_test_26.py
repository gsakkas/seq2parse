marks = []

def start_marking(answers_list):
    student = [["D","D","A","A","C","D","B","B","C","B","D","A","C","C","A"]
/["B","D","A","C","C","D","B","A","C","B","D","C","C","A","A"]]
    total1 = 0
    total2 = 0
    position = 0
    correct = answers_list[position]
    global marks
    for person in student:
        for letter in person:
            print(letter, correct)
            if letter == correct:
                total1 = total1 + 1
            position = position + 1
        marks.append(total1)
    for value in marks:
        total2 = total2 + value
    print(total1)
    return total1

def main():
    answers_list = ["B","D","D","B","D","C","D","B","A","B","D","C","A","B","D"]
#B,D,A,C,C,A,B,A,C,B,D,A,C,A,C
#B,D,A,C,C,D,B,A,C,B,D,C,C,A,D
#B,B,C,B,D,A,C,C,A,D,C,A,C,D,B
    start_marking(answers_list)

main()
