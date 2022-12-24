import random

x = list(random.randint(0, 30) for i in range(20))
a = x[1::2]
a.sort()
print(a)
print(x)