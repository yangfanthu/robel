import numpy as np
class A(object):
    def __init__(self, l):
        self.l = l
    def report(self):
        print(self.l.m)
class B(object):
    def __init__(self,m):
        self.m = m

if __name__ == "__main__":
    b=B(10)
    a = A(b)
    a.report()
    b.m=1
    a.report()