def foo(x):
    i = 1000
    while i > 0:
        i = i - 1
    return x

def benchmark(x):
    a = 100
    i = 200
    while i > 0:
        j = 200
        while j > 0:
            k = 125
            while k > 0:
                if i % 2 == 0:
                    a = a + x * 3
                else:
                    a = a - x + j
                if j % 7 == 0:
                    a = a // 2 + 3 * i
                if k % 2 == 0:
                    a = a + j * k

                k = k - 1
            j = j - 1
        i = i - 1
    return a - 67

print(benchmark(89))
