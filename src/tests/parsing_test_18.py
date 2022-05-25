def simple_fare(num_stops):
    if num_stops<=5:
        return num_stops
    elif num_stops >5:
        if num_stops < 11:
            return 5 + (num_stops - 5) * 0.50
    elif num_stops>11:
        if num_stops < 19:
            return (5+(num_stops - 5) *.50) *.25


print(simple_fare(1))
print(simple_fare(5))
print(simple_fare(6))
print(simple_fare(11))
print(simple_fare(12))
