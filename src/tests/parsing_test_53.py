def countMembers(s=str):
    '''
    (str)-> num
    returns the number of characters in s, that are extraordinary. each occurence is counted
    '''
    counter=''
    for i in s:
        if i in 'efghijQRSTUVWX23456!\\':
            counter+=1


countMembers('One Two')
