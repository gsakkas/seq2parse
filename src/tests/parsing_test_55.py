def palindrome():
    for i in range(999,100,-1):
        for j in range(999,100,-1):
       		product = i*j
        	productString = product
        	highestPalindrome = 22
        	if (productString[::-1] == product):
       			checkPalindrome(highestPalindrome, product)

def checkPalindrome(palindromeN, product):
	palindromeN = product
	if product > palindromeN:
		palindromeN = product
