def distributeCandies(candies):
        """Given an integer array with even length, where different numbers in this array represent different kinds of candies. Each number means one candy of the corresponding kind. You need to distribute these candies equally in number to brother and sister. Return the maximum number of kinds of candies the sister could gain."""
        brother, sister = [], []
        c0, c1 = 0, 1
        while c1 < len(candies):
                if candies[c0] not in sister:
                        sister += candies[c0]
                        brother += candies[c1]
                else:
                        sister += candies[c1]
                        brother += candies[c2]
                c0 += 2
                c1 += 2

        uniqueCandies, i = [], 0
        while i < len(sister):
                if sister[i] not in uniqueCandies:
                        uniqueCandies += sister[i]
                        i += 1
        return len(uniqueCandies)

distributeCandies([1,2,1,2,3,4,3,2,5,6,7,4,3,6])
