P = 1
A = 2
S = 3
C = 4
Q = 5

def print_nation(names, continents, populations, areas):
    """Info of the nation."""
    index = 0
    while index < len(names):


        # print("name:%s" %names[index] \"Continent:%d" %continents[index] \"Population:%d" %populations[index] \"Area:%s" %areas[index])
        # index += 1

        print("name: {}, contintent: {}, population {}".format(names[index], continents[index], populations[index]))
        index += 1
def function_search(names, continents, populations, areas):
    """search nation."""
    country = str(input("What nation do you wish to search for?"))
    while country == false:
        print("invalid nation")
        if country == 'Algeria':
             print("Name:", names[0], "Continent:", continents[0], "Population:", populations[0], "Area:", areas[0])
        elif country == 'Angola':
             print("Name:", names[1], "Continent:", continents[1], "Population:", populations[1], "Area:", areas[1])
        elif country == 'Benin':
             print("Name:", names[2], "Continent:", continents[2], "Population:", populations[2], "Area:", areas[2])
        elif country == 'Afghanistan':
             print("Name:", names[3], "Continent:", continents[3], "Population:", populations[3], "Area:", areas[3])
        elif country == 'Armenia':
             print("Name:", names[4], "Continent:", continents[4], "Population:", populations[4], "Area:", areas[4])
        elif country == 'Bangladesh':
             print("Name:", names[5], "Continent:", continents[5], "Population:", populations[5], "Area:", areas[5])
        elif country == 'Finland':
             print("Name:", names[6], "Continent:", continents[6], "Population:", populations[6], "Area:", areas[6])
        elif country == 'Mexico':
             print("Name:", names[7], "Continent:", continents[7], "Population:", populations[7], "Area:", areas[7])

def pop_density(names, continents, populations, areas):
    people = str(input("What nation do you wish to search for?"))
    if people == false:
        print("Invalid Nation")
    elif people == 'Algeria':
        print("The population density is:", populations[0] / areas[0])
    elif people == 'Angola':
        print("The population density is:", populations[1] / areas[1])
    elif people == 'Benin':
        print("The population density is:", populations[2] /  areas[2])
    elif people == 'Afghanistan':
        print("The population density is:", populations[3] /  areas[3])
    elif people == 'Armenia':
        print("The population density is:", populations[4] /  areas[4])
    elif people == 'Bangladesh':
        print("The population density is:", populations[5] /  areas[5])
    elif people == 'Finland':
        print("The population density is:", populations[6] /  areas[6])
    elif people == 'Mexico':
        print("The population density is:", populations[7] /  areas[7])



def menu():
    print("Please choose from the following menu.")
    print("(P)rint list of nations")
    print("(A)dd a nation")
    print("(S)earch for a nation")
    print("(C)ompute population density of a nation")
    print("(Q)uit")
    option = input("Please enter the first letter of your choice:")
    return option
def main():
    names = ["Algeria", "Angola", "Benin", "Afghanistan", \
         "Armenia", "Bangladesh", "Finland", "Mexico"]

    continents = ["Africa", "Africa", "Africa", "Asia", \
              "Asia", "Asia", "Europe", "North America"]

    populations = [33333216, 12263596, 8078314, 31889923, 2971650, \
               150448339, 5238460, 109955400]

    areas = [2381740, 481353.6, 43482.83, 647500, 29800, 144000, \
         338145, 1972550]


    choice = menu()
    while choice != 'Q':
        if choice == 'S':
                   function_search(names, continents, populations, areas)
        elif choice == 'C':
                   pop_density(names, continents, populations, areas)
        elif choice == 'P':
                   print_nation(names, continents, populations, areas)
        choice = menu()
    print ("goodbye...")

main()
