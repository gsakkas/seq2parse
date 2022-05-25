#Ulubione języki programowania
favorite_languages = {
    'janek': ['python', 'ruby'],
    'sara': ['c'],
    'edward': ['ruby', 'go'],
    'paweł': ['python', 'haskel'],
    }
for name,languages in favorite_languages.items():
    if len(languages) == 1:
        print("\nUlubiony język programowania użytkownika " + name.title() +
               " to " + language.title())
    else:
        print("\nUlubione języki programowania użytkownika " + name.title() +
          " to:")
        for language in languages:
            print("\t" + language.title())
