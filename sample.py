"""
Given an array of integers, and a number ‘sum’, find the number of pairs of integers in the array whose sum is equal to ‘sum’.

arr = [1, 5, 7, -1, 5]
n = len(arr)
sum = 6

(1,5) , (5,1)
"""


def solution(arr, t):
    result = []
    seen = {}
    for n in arr:
        diff = t - n
        if diff in seen:
            if (n, diff) not in result:
                result.append((n, diff))
                if (diff, n) not in result:
                    result.append((diff,n))
        else:
            seen[diff] = diff

    return result

if __name__ == '__main__':

    result = solution([1, 5, 7, -1, 5],6)
    print(result)