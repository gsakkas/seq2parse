def number_of_customers_per_state(customers_dict):
  final_dict = {}
  count = 0
  for k1, v1 in customers_dict.items():
    for i in v1:

        for k2, v2 in i.items():
            if type(v2) == int:
                if v2 > count:
                    count = v2
                    print (k2)
                    print (v2)
                    final_dict[k1] = {}
        final_dict[k1] = count

  print (final_dict)


customers = {
    'UT': [{
        'name': 'Mary',
        'age': 28
    }, {
        'name': 'John',
        'age': 31
    }, {
        'name': 'Robert',
        'age': 16
    }],
    'NY': [{
        'name': 'Linda',
        'age': 71
    }],
    'CA': [{
        'name': 'Barbara',
        'age': 15
    }, {
        'name': 'Paul',
        'age': 18
    }]
}
number_of_customers_per_state(customers)
