import time
from random import randint

class Tamagushi():

    def __init__(self):
        self.nome = "Fofinho"
        self.fome = 50
        self.saude = 50


    def alterar_nome(self,nome):
        self.nome = nome
        print("Meu nome é {0}!".format(self.nome))

    def alimentar(self):
        if self.fome < 100:
            self.fome += 10
            print("nham nham!")
        else:
            print("Parece que comi um elefante! :D")

    def cuidar(self):
        interacao = randint(0,2)
        if self.saude < 100:
            self.saude +=10
        if interacao == 0:
            print("abraço")
        elif interacao == 1:
            print("lambeijo!")
        else:
            print("pula em cima")
            print("melhor mãe do mundo!!!")


    def calcula_humor(self):
        media = 0
        if self.saude < 50:
            media += 1
        elif self.saude == 100:
            media += 5
        else:
            media += 3

        if self.fome < 50:
            media += 1
        elif self.fome == 100:
            media +=5
        else:
            media +=3

        media = media/2

        if media == 1:
            print("que dia pessimo!estou me sentindo muito mal >:( ")
        elif media == 2:
            print("tô meio mal :/")
        elif media == 3:
            print("tô melhorando...")
        elif media == 5:
            print("o dia está incrivel! estou super feliz :D")
        else:
            print("tô confuso... não sei o que tô sentindo (?)")



bichinho="""_____000000000____________________00000000000_____
__0000_______00__________________00________0000___
_00__________000000000____00000000____________00__
_0_________000_____________________000_________00_
_0_______000_________________________000________0_
_00_____00_____________________________00______00_
__00___00______0000___________0000______00___000__
___00000______000000_________000000______00000____
______00_______0000___________0000_______00_______
______00_____________00000000____________00_______
______00_____________00____00____________00_______
_______0______________000000_____________0________
_______00_______________00_______________00_______
________000_____________00_____________000________
__________000___________00___________000__________
____________0000________00________0000____________
_______________0000000_0000_0000000"""


tg = Tamagushi()
#print(bichinho)

def estatisticas():
    print("")
    print("{0}".format(bichinho))
    print("Estatisticas de {0}: ".format(tg.nome))
    print("Fome: {0}%".format(tg.fome))
    print("Saúde: {0}%".format(tg.saude))

def menu():
    print("")
    print("Selecione uma opção:")
    print("1 - alimentar")
    print("2 - cuidar")
    print("3 - mudar nome")
    print("4 - ver humor")
    print("0 - sair")


op = 1

while (op != 0):
    estatisticas()
    menu()
    op = int(input("\n"))
    if op == 1:
        tg.alimentar()
    elif op == 2:
        tg.cuidar()
    elif op == 3:
        nome = input("Qual será meu novo nome: ")
        tg.alterar_nome(nome)
    elif op == 4:
        tg.calcula_humor()
    time.sleep(5)


print("Até mais! :3")
