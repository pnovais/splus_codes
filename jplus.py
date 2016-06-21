"""
Programa teste

"""


#!/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from sys import exit
from functools import reduce
from ff_bug import FriendsOfFriends
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def distancias(df_seg):
    d = 0.0
    for i in df_seg:
        d = df_seg[0][i] - df_seg[0][i+1]
    return sqrt(d)

#a funcao, de fato, utilizada
def distance(x1,y1,x2,y2):
    dist = 0.0
    dist = (x1-x2)**2 + (y1-y2)**2
    d = np.sqrt(dist)
    return d

def dist(row):
    return np.sqrt(np.sum((row[1]-row[2])**2))

ini=time.time()

now = datetime.date.today()

print(bcolors.OKBLUE + '===================================================================================' )
print('===================================S3P PROGRAM=====================================')
print('=========================Stellar Populations Pixel by Pixel========================')
print('==================================== %s ===================================' %now)
print('===================================================================================')
print('==================================================================================='+ bcolors.ENDC)




#ABRINDO OS ARQUIVOS COM AS CONTAGENS (12 BANDAS)
print('')
print('*'*83)
str1 = '  LEITURA DOS ARQUIVOS'
print(bcolors.FAIL + str1.center(80)+ bcolors.ENDC)
print('*'*83)
print('Lendo os arquivos com as contagens, em todas as bandas...')

outpath = '/Dropbox/DOUTORADO/JPLUS_clusters/gal_A2589A2593'
filename1 = 'gal-A2589-A2596_uJAVA_swp.txt'
filename2 = 'gal-A2589-A2596_gSDSS_swp.txt'
filename3 = 'gal-A2589-A2596_rSDSS_swp.txt'
filename4 = 'gal-A2589-A2596_iSDSS_swp.txt'
filename5 = 'gal-A2589-A2596_zSDSS_swp.txt'
filename6 = 'gal-A2589-A2596_J0378_swp.txt'
filename7 = 'gal-A2589-A2596_J0395_swp.txt'
filename8 = 'gal-A2589-A2596_J0410_swp.txt'
filename9 = 'gal-A2589-A2596_J0430_swp.txt'
filename10 = 'gal-A2589-A2596_J0515_swp.txt'
filename11 = 'gal-A2589-A2596_J0660_swp.txt'
filename12 = 'gal-A2589-A2596_J0861_swp.txt'


arq1 = pd.read_table(filename1, delim_whitespace=True, header=None)
arq2 = pd.read_table(filename2, delim_whitespace=True, header=None)
arq3 = pd.read_table(filename3, delim_whitespace=True, header=None)
arq4 = pd.read_table(filename4, delim_whitespace=True, header=None)
arq5 = pd.read_table(filename5, delim_whitespace=True, header=None)
arq6 = pd.read_table(filename6, delim_whitespace=True, header=None)
arq7 = pd.read_table(filename7, delim_whitespace=True, header=None)
arq8 = pd.read_table(filename8, delim_whitespace=True, header=None)
arq9 = pd.read_table(filename9, delim_whitespace=True, header=None)
arq10 = pd.read_table(filename10, delim_whitespace=True, header=None)
arq11 = pd.read_table(filename11, delim_whitespace=True, header=None)
arq12 = pd.read_table(filename12, delim_whitespace=True, header=None)

arq1.columns = ['x','y','flux_u']
arq2.columns = ['x','y','flux_g']
arq3.columns = ['x','y','flux_r']
arq4.columns = ['x','y','flux_i']
arq5.columns = ['x','y','flux_z']
arq6.columns = ['x','y','flux_J0378']
arq7.columns = ['x','y','flux_J0395']
arq8.columns = ['x','y','flux_J0410']
arq9.columns = ['x','y','flux_J0430']
arq10.columns = ['x','y','flux_J0515']
arq11.columns = ['x','y','flux_J0660']
arq12.columns = ['x','y','flux_J0861']


#print('')
print('Gerando um arquivo unico...')


#df_fin eh o arquivo com todos os fluxos em todas as bandas
fluxes = [arq1,arq2,arq3,arq4,arq5,arq6,arq7,arq8,arq9,arq10,arq11,arq12]
df_fin = reduce(lambda left, right: pd.merge(left,right),fluxes)

saida_fluxes = 'all_band_fluxes.txt'

formats=['%d','%d','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f']
#headers=['x','y','flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378','flux_J0395','flux_J0410', 'flux_J0430','flux_J0515','flux_J0660','flux_J0861']
headers2='x\ty\tflux_u\tflux_g\tflux_r\tflux_i\tflux_z\tflux_J0378\tflux_J0395\tflux_J0410\tflux_J0430\tflux_J0515\tflux_J0660\tflux_J0861'

np.savetxt(saida_fluxes,df_fin, fmt=formats, delimiter='\t',header=headers2)
print('Todas as contagens podem ser encontradas no arquivo "%s".' %saida_fluxes)


#COORDENADAS DO CEU E COORDENADAS DO RECORTE
xc1=240
xc2=320
yc1=40
yc2=100

xr1=100
xr2=400
yr1=80
yr2=380

str2='  Estatisticas do ceu, na banda r'
s=str2.upper()

print('')
print('*'*83)
print(bcolors.FAIL + s.center(80) + bcolors.ENDC)
print('*'*83)



#Criando novos frames para calcular as estatisticas do ceu, na banda r
#Selecao apenas das colunas x,y e flux_r
#df_sky_r=df_fin.ix[:,[0,1,4]]


#selecao dos pontos que pertencem a regiao do ceu escolhida
df = df_fin.ix[(df_fin['x'] > xc1) & (df_fin['x'] < xc2) & (df_fin['y'] > yc1) & (df_fin['y'] < yc2) ]

print('Sumario das estatistica do ceu, na banda r:')
print('Media: %5.5f' %df['flux_r'].mean())
print('Std: %5.5f' %df['flux_r'].std())


#binn = int(input('Binagem da imagem: Digite o tamanho do bin:'))
binn=1

npbx=(xr2-xr1+1)/binn
npby=(yr2-yr1+1)/binn

#CRIANDO O DATAFRAME COM OS PIXEIS SOMENTE DA REGIAO RECORTADA
df_recorte = df_fin.ix[(df_fin['x'] > xr1) & (df_fin['x'] < xr2) & (df_fin['y'] > yr1) & (df_fin['y'] < yr2) ]

#CRIANDO O DF COM O CEU SUBTRAIDO DE TODAS AS BANDAS, NA REGIAO DE RECORTE
#rss = Recortado e Sky Subtracted

df_aux=df_recorte.ix[:,2:]
df_aux1=df_recorte.ix[:,:2]

df_aux3 = (df_aux - df_aux.mean())

df_rss=df_aux1.join(df_aux3)

#CALCULANDO A RAZAO ENTRE AS BANDAS g E r:
df_corgr = df_rss['flux_g'] / df_rss['flux_r']


str3="Algumas propriedades da area a ser analisada"

print('')
print('*'*83)
print(bcolors.FAIL + str3.upper().center(80) + bcolors.ENDC)
print('*'*83)


print('tamanho do bin: %d' %binn)
print('Num. de pixeis binados, em x e y: %d, %d' %(npbx, npby))
print('Contagens minimas e maximas, na banda r, com ceu subtraido: %5.4f, %5.4f' %(df_rss['flux_r'].min(), df_rss['flux_r'].max()))
print('Cor* minimas e maximas, na banda r, com ceu subtraido: %5.4f, %5.4f' %(df_corgr.min(), df_corgr.max()))
print('')
print(bcolors.HEADER + '''***a cor foi calculada, de forma simplista e apenas para conferir os resultados,
como a razao entre as bandas 1 e 2 (flux_banda1/flux_banda2), quanto menor a razao,
mais vermelho o pixel''' +bcolors.ENDC)
print('')
#print df_corgr.min(),df_corgr.max()


"""
A segmentacao consiste de usar um limiar para separar o objeto do fundo.
No nosso caso, usamos limiar = alpha*std_ceu
"""

str4 = "segmentacao"
print('')
print('*'*83)
print(bcolors.FAIL + str4.upper().center(80) + bcolors.ENDC)
print('*'*83)

alpha=1.5
limiar=alpha*df['flux_r'].std()
#df eh o dataframe para a regiao do ceu considerada

#SELECAO DOS PIXEIS ACIMA DO LIMIAR
df_seg=df_rss.ix[df_rss['flux_r'] > limiar]

alimiar=len(df_seg)
pix=len(df_rss)
pix_acima=float(alimiar)/pix


print('Alpha: %5.1f' %alpha)
print('Limiar de segmentacao: %5.4f' %limiar)
print('Num. total de pixeis: %d' %pix)
print('Pixeis acima do limiar: %d' %alimiar)
print('Fracao de pixeis acima do limiar: %5.5f' %pix_acima)

plt.scatter(df_seg['x'], df_seg['y'])
#plt.show()

np.savetxt('teste.txt',df_seg, fmt=formats, delimiter='\t')
'''
Friends of friends ira identificar os pixeis, na banda r, que pertencem
a galaxia, separando dos residuos da imagem.
'''

str4="friends of friends"

print('')
print('*'*83)
print(bcolors.FAIL + str4.upper().center(80) + bcolors.ENDC)
print('*'*83)

'''
df_fof=df_seg.ix[:,:2]
print(df_fof.head())
np.savetxt('fof.txt',df_fof, delimiter='\t')
'''

formats2=['%d','%d','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%d']


#if-then-else using numpys where()

df_rss2 = df_rss

df_rss2['logic']= np.where(df_rss['flux_r'] > limiar, 1, 0)
np.savetxt('fof2.txt',df_rss2,fmt=formats2, delimiter='\t')

#df_seg['distancia'] = df_seg.apply(dist, axis=1)

print(df_seg.head())



fim = time.time()
time_proc = fim - ini
print('tempo de processamento: %fs' %time_proc)
