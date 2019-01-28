import numpy as np

def compute_new_entropy(zeroes, ones):
    first = (zeroes/(zeroes+ones)) * np.log2(zeroes/(zeroes+ones))
    second = (ones/(zeroes+ones)) * np.log2(ones/(zeroes+ones))
    return first * second * -1

def compute_old_entropy(zeroes, ones):
    first = (zeroes/(zeroes+ones)) * np.log2(zeroes/(zeroes+ones))
    second = (ones/(zeroes+ones)) * np.log2(ones/(zeroes+ones))
    return (first + second) * -1

def compute_gini(zeroes,ones):
    first = (zeroes/(zeroes+ones)) ** 2
    second = (ones/(zeroes+ones)) ** 2
    return 1 - first - second


print(compute_new_entropy(3,1))

print(compute_gini(0,3))
print(compute_gini(3,1))