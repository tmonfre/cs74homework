import numpy as np

x = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
y = np.array([[7,8,9],[10,11,12]], dtype=np.float64)

# print(x * 2)

print(np.mean(x, axis=0))

print(np.transpose(x))

print(x[0])
print(x[1])
print(np.dot(x[0], x[1]))




