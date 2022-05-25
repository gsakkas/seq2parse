def check_rush_hour(day, hour, minute):
    if day== "Monday" or day=="Tuesday" or day=="Wednesday" or day=="Thursday" or day=="Friday":
        if hour>= 5:
            if hour<= 9:
                if minute >=0:
                    if minute<=30:
                        return True
                    else:
                        return False
        elif hour>= 15:
            if hour<= 19:
               return True
            else:
                return False
    else:
        return False





print(check_rush_hour("Monday", 6, 0))
