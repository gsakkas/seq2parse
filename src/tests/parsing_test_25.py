def soma_dobro(a, b):
  if a != b:
    return a + b
  else:
    return a * b
  return


def test(obtido, esperado):
  if obtido == esperado:
    prefixo = ' Parabéns!'
  else:
    prefixo = ' Ainda não'
  print ('%s obtido: %s esperado: %s'
         % (prefixo, repr(obtido), repr(esperado)))

def main():
  print ('Soma dobro')
  test(soma_dobro(1, 2), 3)
  test(soma_dobro(3, 2), 5)
  test(soma_dobro(2, 2), 8)
  test(soma_dobro(-1, 0), -1)
  test(soma_dobro(0, 0), 0)
  test(soma_dobro(0, 1), 1)

print()
