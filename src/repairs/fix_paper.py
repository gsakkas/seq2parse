def foo1(a):
  a += 14
  return a + 42

def foo2(a):
  print(a)
  return a + 42

def foo3(a):
  b = foo2(foo1(a))
  return b

def bar(a):
  b = foo3(a) + 17
  return b
