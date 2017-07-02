def a():
    raise ValueError("test")
    raise NotImplementedError

def b():
    a()

def c():
    pass
    raise SyntaxError

try:
    b()
except ValueError as e:
    print(type(e))