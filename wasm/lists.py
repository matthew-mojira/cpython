def list_stuff():
    a = [1, 2, 3]
    a[1] = 0

    return a

def list_stuff2(b):
    a = []
    while b > 0:
        a.append(b)
        b = b - 1
    return a

def list_stuff3():
    a = [1, 2, 3]
    b = [4, 5, 6]

    c = a + b

    return c

import dis

print()
print("list stuff")
dis.dis(list_stuff)
print()
print("list stuff2")
dis.dis(list_stuff2)
print()
print("list stuff3")
dis.dis(list_stuff3)