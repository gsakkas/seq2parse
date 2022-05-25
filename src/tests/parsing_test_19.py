def get_quantities(table_to_foods: Dict[str, List[str]]) -> Dict[str, int]:
    food_to_quantity = {}

    for table in table_to_foods:
        for meal in table:
            if meal in food_to_quantity:
                food_to_quantity[meal] += 1
            else:
                food_to_quantity[meal] = 1

    return food_to_quantity

get_quantities({'t1': ['Vegetarian stew', 'Poutine', 'Vegetarian stew'], 't3': ['Steak pie', 'Poutine', 'Vegetarian stew'], 't4': ['Steak pie', 'Steak pie']})
