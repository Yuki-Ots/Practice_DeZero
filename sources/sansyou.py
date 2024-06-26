import sys

class Obj:
    pass

def f(x):
    print(sys.getrefcount(x))

a = Obj()

print(sys.getrefcount(a))
f(a)
print(sys.getrefcount(a))
a = None
print(sys.getrefcount(a))
f(a)

mylist = []
print(sys.getrefcount(mylist))

mylist.append(mylist)
print(sys.getrefcount(mylist))
print(mylist)

mylist.append(mylist)

print(sys.getrefcount(mylist))
print(mylist)

b = []
b.append(3)
print(b)
