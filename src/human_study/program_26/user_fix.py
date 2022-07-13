class tree:
    def __init__(self):
        self.raiz = None

    def insert(self, valor, raiz):
        if self.raiz.getDado() == None:
            self.raiz.setDado(valor)
        else:
            self.insert(valor, self.raiz.getDireita())
            self.insert(valor, self.raiz.getEsquerda())

G = tree()
G.insert(12, G.raiz)
G.insert(2, G.raiz)
G.insert(20, G.raiz)
