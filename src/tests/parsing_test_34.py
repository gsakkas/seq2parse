#Produce 20 random die tosses in a list
import random    
values=[]    
for i in range (20):
    values.append(random.randint(1,6))
     
inRun = False # Set a boolean variable inRun to false.
for i in range (20): #For each valid index i in the list
    if inRun: #If inRun
        if values[i] != values[i - 1]: #If values[i] is different from the preceeding value
            print(")"), #Print ).
            inRun = False #inRun = false.
        else: print(values[i]),
    if not inRun: #If not inRun
        if (values[i] == values[i + 1]): #If values[i] is the same as the following value
            print("("), #Print (.
            inRun = True #inRun = true.
            print(values[i]), #Print values[i].
        else: print(values[i]),
    if i == 18:
        print(values[i + 1]),
        if inRun:
            print(")")  #If inRun, print ).
