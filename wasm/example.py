def f(x):
    a = 1
    while x > 0:
        x =  x - a
        a = a * 2 
        if a % 2 :
            a = a + 1 
    return x + a + 1

print(f(3))