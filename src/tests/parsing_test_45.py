#Project_1.py
#Angel Cowen
#Python 3.6

#Import Programs
import math



#This Program will estimate the sine of an angle using the Taylor Series.

#Factorial Function
def fact(f):
    for i in range (f-1, 1, -1):
        f=f*i
    return(f)



def main():
    #Handshake
    print("This Program will estimate the sine of an angle using the Taylor Series")

    #Input
    angle = float(input("Please enter the angle, in degrees: "))
    times = int(input("How many iterations of the"
                      "Taylor Series would you like executed"))

    #Convert to Radians
    radians = math.radians(angle)

    #Calculate
    #sin = (((radians**3)/fact(3))+((radians**5)/fact(5))-((radians**7)/fact(7))+((radians**9)/fact(9)))...

    exp_den = 0
    for i in range(1,times*2+1,2 ):
        exp_den = i

    Sin = 0
    for i in range(times):
        (radians**ext_dem)/fact(extdem)
    sin = sin + i

    #sign
    sign = 1
    sign = -sign


    #Output
    print("The estimated sine of your angle is", sin)
    print("The actual sine of the angle is" (angle))
    print("The Error was", abs(sin - angle))

main()
