class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        m = {}
        l = []
        for i in range(len(nums)):
            if( nums[i] in m):
                m[nums[i]] += str(i)
            else:
                m[nums[i]] = str(i)
        for i in range(len(nums)):
            v = target - nums[i]
            if(v in m) and int(m[v]) != i:
                if len(m[v]) == 1:
                    l.append(i)
                    l.append(int(m[v]))
                    return l
                else:
                    l.append(i)
                    t = m[v][len(m[v])-1:]
                    l.append(int(t))
                    return l

s=Solution()
s.twoSum([230,863,916,585,981,404,316,785,88,12,70,435,384,778,887,755,740,337,86,92,325,422,815,650,920,125,277,336,221,847,168,23,677,61,400,136,874,363,394,199,863,997,794,587,124,321,212,957,764,173,314,422,927,783,930,282,306,506,44,926,691,568,68,730,933,737,531,180,414,751,28,546,60,371,493,370,527,387,43,541,13,457,328,227,652,365,430,803,59,858,538,427,583,368,375,173,809,896,370,789]
,542)
