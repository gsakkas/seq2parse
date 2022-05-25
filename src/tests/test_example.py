def check_rush_hour(day, hour, minute):
    if day== 'Monday' or day=='Tuesday' or day=='Wednesday' or day=='Thursday' or day=='Friday':
        if hour>= 5:
            if hour<= 9:
                if minute= 0 or minute=1 or minute=2 or minute=3 or minute=4 or minute=5 or minute=6:
                    return True
                else:
                    return False
        if hour>= 15:
            if hour<= 19:
                elif minute= 0:
                    return True
                else:
                    return False
    else:
        return False



print(check_rush_hour("Monday", 6, 0))
