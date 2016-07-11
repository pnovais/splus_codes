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
from mpl_toolkits.mplot3d import Axes3D
import fof_fortran
import matplotlib.mlab as mlab
import scipy, pylab
from pandas import DataFrame
import seaborn as sns
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com
from rpy2.robjects import pandas2ri
import math



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

ini=time.time()

now = datetime.date.today()

print(bcolors.OKBLUE + '===================================================================' )
print('===========================S3P PROGRAM=============================')
print('=================Stellar Populations Pixel by Pixel================')
print('============================ %s ===========================' %now)
print('===================================================================')
print('==================================================================='+ bcolors.ENDC)


'''
================================================================================
Abrindo os arquivos
================================================================================
'''

#ABRINDO OS ARQUIVOS COM AS CONTAGENS (12 BANDAS)
print('')
print('*'*67)
str1 = '  LEITURA DOS ARQUIVOS'
print(bcolors.FAIL + str1.center(64)+ bcolors.ENDC)
print('*'*67)
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

formats=['%d','%d','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f',
         '%5.4f','%5.4f','%5.4f','%5.4f','%5.4f']
#headers=['x','y','flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378',
#'flux_J0395','flux_J0410', 'flux_J0430','flux_J0515','flux_J0660','flux_J0861']
headers2='''x\ty\tflux_u\tflux_g\tflux_r\tflux_i\tflux_z\tflux_J0378\tflux_J0395
           \tflux_J0410\tflux_J0430\tflux_J0515\tflux_J0660\tflux_J0861'''

np.savetxt(saida_fluxes,df_fin, fmt=formats, delimiter='\t',header=headers2)
print('As contagens podem ser encontradas no arquivo "%s".' %saida_fluxes)

'''
================================================================================
Recortando a imagem e subtraindo o ceu
================================================================================
'''

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
print('*'*67)
print(bcolors.FAIL + s.center(64) + bcolors.ENDC)
print('*'*67)


#Criando novos frames para calcular as estatisticas do ceu, na banda r
#Selecao apenas das colunas x,y e flux_r
#df_sky_r=df_fin.ix[:,[0,1,4]]

#selecao dos pontos que pertencem a regiao do ceu escolhida
df = df_fin.ix[(df_fin['x'] > xc1) & (df_fin['x'] < xc2) & (df_fin['y'] > yc1)
    & (df_fin['y'] < yc2) ]

print('Sumario das estatistica do ceu, na banda r:')
print('Media: %5.5f' %df['flux_r'].mean())
print('Std: %5.5f' %df['flux_r'].std())


#binn = int(input('Binagem da imagem: Digite o tamanho do bin:'))
binn=1

npbx=(xr2-xr1+1)/binn
npby=(yr2-yr1+1)/binn

#CRIANDO O DATAFRAME COM OS PIXEIS SOMENTE DA REGIAO RECORTADA
df_recorte = df_fin.ix[(df_fin['x'] > xr1) & (df_fin['x'] < xr2) &
             (df_fin['y'] > yr1) & (df_fin['y'] < yr2) ]

#CRIANDO O DF COM O CEU SUBTRAIDO DE TODAS AS BANDAS, NA REGIAO DE RECORTE
#rss = Recortado e Sky Subtracted

df_aux=df_recorte.ix[:,2:]
df_aux1=df_recorte.ix[:,:2]

df_aux3 = (df_aux - df_aux.mean())

df_rss=df_aux1.join(df_aux3)

#CALCULANDO A RAZAO ENTRE AS BANDAS g E r:
df_corgr = df_rss['flux_g'] / df_rss['flux_r']


'''
================================================================================
Propriedades da area a ser analisada
================================================================================
'''

str3="Algumas propriedades da area a ser analisada"

print('')
print('*'*67)
print(bcolors.FAIL + str3.upper().center(64) + bcolors.ENDC)
print('*'*67)


print('tamanho do bin: %d' %binn)
print('Num. de pixeis binados, em x e y: %d, %d' %(npbx, npby))
print('''Contagens min. e max., na banda r, ceu subtr.:
      %5.4f, %5.4f''' %(df_rss['flux_r'].min(), df_rss['flux_r'].max()))
print('''Cor* min. e max., na banda r, ceu subtr.: %5.4f, %5.4f'''
      %(df_corgr.min(), df_corgr.max()))
print('')
print(bcolors.HEADER + '''***a cor foi calculada, de forma simplista e apenas
      para conferir os resultados,como a razao entre as bandas 1 e 2
      (flux_banda1/flux_banda2), quanto menor a razao,mais
      vermelho o pixel''' +bcolors.ENDC)
print('')
#print df_corgr.min(),df_corgr.max()


"""
A segmentacao consiste de usar um limiar para separar o objeto do fundo.
No nosso caso, usamos limiar = alpha*std_ceu
"""


'''
================================================================================
SEGMENTACAO
================================================================================
'''


str4 = "segmentacao"
print('')
print('*'*67)
print(bcolors.FAIL + str4.upper().center(64) + bcolors.ENDC)
print('*'*67)

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



np.savetxt('teste.txt',df_seg, fmt=formats, delimiter='\t')

'''
================================================================================
Friends of Friends
================================================================================

Friends of friends ira identificar os pixeis, na banda r, que pertencem
a galaxia, separando dos residuos da imagem.
'''

str4="friends of friends"

print('')
print('*'*67)
print(bcolors.FAIL + str4.upper().center(64) + bcolors.ENDC)
print('*'*67)


formats2=['%d','%d','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f',
          '%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%5.4f','%d']

#if-then-else using numpys where()

df_rss2 = df_rss

df_rss2['logic']= np.where(df_rss['flux_r'] > limiar, 1, 0)
np.savetxt('fof2.txt',df_rss2,fmt=formats2, delimiter='\t')

arr1=df_seg['x']
arr2=df_seg['y']
n=len(df_seg)
l=1

a = fof_fortran.fof(arr1, arr2,l,n)

df_ff=pd.DataFrame(a)

df_ff.columns = ['fof']

compr=len(df_seg)
labels = []

#criando os rotulos iniciais
for row in range(0,compr):
    labels.append(row)

df_seg['label'] = labels
df_ff['label']=labels


df_x = reduce(lambda left, right: pd.merge(left,right),[df_seg,df_ff])

#np.savetxt('indexes.txt',a,delimiter='\t')

#MAKING HISTOGRAMS
#plt.hist(a)
#plt.axis([0, 50,0,8000])
#plt.show()

#Making a 3D plot

#ax = fig.gca(projection='3d')

#ax.set_title('3D Scatter Plot')
#ax.set_xlabel('Column a')
#ax.set_ylabel('Column b')
#ax.set_zlabel('Column c')

#ax.set_xlim(0, 500)
#ax.set_ylim(0, 500)
#ax.set_zlim(33, 36)

#ax.view_init(elev=0, azim=20)              # elevation and angle
#ax.dist=12

#ax.scatter(
#           df_x['x'], df_x['y'], df_x['fof'],  # data
#           color='purple',                            # marker colour
#           marker='o',                                # marker shape
#           s=30                                       # marker size
#           )

#plt.show()

dfs = df_x.ix[(df_x['fof'] > 34.5) & (df_x['fof'] < 35.5)]

#salvando os pixeis da galaxia em um arquivo txt
np.savetxt('gal.txt',dfs,delimiter='\t')


fig = plt.figure()

axes = plt.gca()
axes.xaxis.set_tick_params(labelsize=15)
axes.yaxis.set_tick_params(labelsize=15)

labelfont = {
        'family' : 'sans-serif',  # (cursive, fantasy, monospace, serif)
        'color'  : 'black',       # html hex or colour name
        'weight' : 'normal',      # (normal, bold, bolder, lighter)
        'size'   : 14,            # default value:12
        }

titlefont = {
        'family' : 'serif',
        'color'  : 'black',
        'weight' : 'bold',
        'size'   : 16,
        }

plt.scatter(dfs['x'], dfs['y'],
            color='dodgerblue',
            label='Spatial Distribution')

plt.title('Spatial Distribution', fontdict=titlefont)
plt.xlabel('X', fontdict=labelfont, fontweight='bold')
plt.ylabel('Y', fontdict=labelfont)

fig.savefig('distrib_FoF.png')

#plt.show()

m = len(dfs)

print('')
print("Total de pixeis acima do limiar: %d" %n)
print("Total de pixeis da galaxia segmentada: %d" %m)


'''
================================================================================
PCA com o R
================================================================================
'''

str5="Principal components Analysis - PCA (using R)"

print('')
print('*'*67)
print(bcolors.FAIL + str5.upper().center(64) + bcolors.ENDC)
print('*'*67)


#convertendo o dataframe do pandas em um dataframe dentro do R
rdf=com.convert_to_r_dataframe(dfs)

ro.globalenv['data'] = rdf
ro.r('mydata2 <- data[3:14]')
ro.r('mydata_coord <- data[1:2]')
ro.r('my_data_coord <- data[1:2]')

#calculando as "cores"
ro.r('data2 = mydata2 - mydata2$flux_r')
ro.r('data2b = data2[1:2]')
ro.r('data2c = data2[4:12]')
ro.r('mydata = cbind(data2b,data2c)')

# pca com variaveis com media zero e variancia unitaria
ro.r('mydata.pca <- prcomp(mydata, retx=TRUE, center=TRUE, scale=TRUE)')
# raiz quadrada dos autovalores (singular value)
ro.r('sd <- mydata.pca$sdev')
# autovalores
lamb = ro.r('lambda <- sd^2')
print('> Autovalores das Componentes Principais <')
print(ro.r('lambda'))
#autovalo = pandas2ri.ri2py_dataframe(autoval)

# autovetores
ro.r('autovec <- mydata.pca$rotation')
# nomes das colunas
ro.r('rownames(autovec)')
# componentes principais
ro.r('pc <- mydata.pca$x')

# preparacao para impressao
ro.r('nl <- min(length(lambda),10)')
ro.r('na <- min(length(autovec)^0.5, 10)')

# distance biplot
#ro.r('pdf("Rpca.pdf")')
#ro.r('plot(pc[,1], pc[,2], xlab="PCA 1", ylab="PCA 2",type="n", xlim=c(min(pc[,1:2]), max(pc[,1:2])),ylim=c(min(pc[,1:2]), max(pc[,1:2])))')
#ro.r('points(pc[,1], pc[,2])')
#ro.r('dev.off()')

#unindo os dados novamente
pydf = ro.r('teste <- cbind(mydata_coord,mydata,pc)')
pca_ord = ro.r('pc_ord <- teste[order(teste$PC1),]')
pydf2 = pandas2ri.ri2py_dataframe(pca_ord)

ro.r('compr = nrow(pc_ord)')

pca_ord1 = ro.r('data1 <- pc_ord[1:nrow(pc_ord)/4,]')
pca_ord2 = ro.r('data2 <- pc_ord[(compr/4 + 1):(compr/2),]')
pca_ord3 = ro.r('data3 <- pc_ord[(compr/2 + 1):(3*compr/4),]')
pca_ord4 = ro.r('data4 <- pc_ord[(3*compr/4 + 1):compr,]')

pca1 = pandas2ri.ri2py_dataframe(pca_ord1)
pca2 = pandas2ri.ri2py_dataframe(pca_ord2)
pca3 = pandas2ri.ri2py_dataframe(pca_ord3)
pca4 = pandas2ri.ri2py_dataframe(pca_ord4)



f, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
ax1.scatter(pca1['x'],pca1['y'], color="blue")
ax2.scatter(pca2['x'],pca2['y'], color="green")
ax3.scatter(pca3['x'],pca3['y'], color="goldenrod")
ax4.scatter(pca4['x'],pca4['y'], color="red")
plt.savefig('Distribution_4panel.png')


plt.figure()
plt.scatter(pca1['x'],pca1['y'], color="blue")
plt.scatter(pca2['x'],pca2['y'], color="green")
plt.scatter(pca3['x'],pca3['y'], color="goldenrod")
plt.scatter(pca4['x'],pca4['y'], color="red")
plt.savefig('Distribution_all.png')


#plt.show()
pydf2.columns = ['x','y','u-r','g-r','i-r','z-r','0378-r','0395-r','0410-r','0430-r',
                '0515-r','0660-r','0861-r','PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11']

#plt.clf() #Clear the current figure (prevents multiple labels)


plt.figure()

plt.title('Principal Components Analysis', fontdict=titlefont)
plt.xlabel('PC1', fontdict=labelfont)
plt.ylabel('PC2', fontdict=labelfont)

axes = plt.gca()
axes.xaxis.set_tick_params(labelsize=15)
axes.yaxis.set_tick_params(labelsize=15)

plt.scatter(pydf2['PC1'],pydf2['PC2'],
            color='teal')

plt.savefig('PC1_PC2.png')

#plt.show()


df_corr = pydf2.ix[:,2:14]
corr=df_corr.corr(method='pearson')
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,mask=mask)
plt.savefig('Correlation_PC1_cores.png')



'''
================================================================================
Calculando os momentos invariantes de Hu
================================================================================
'''

str6="Hu Invariant Moments"

print('')
print('*'*67)
print(bcolors.FAIL + str6.upper().center(64) + bcolors.ENDC)
print('*'*67)


'''
!======================================================
!================CENTROIDES DA IMAGEM==================
!======================================================
!xc = M10/M00
!yc = M01/M00
!
!onde xc e yc sao os centroides da imagem, M10, M01 e
!M00 os momentos nao centrais, onde
!
!Mpq=Somatorio(x^p.y^q)
!------------------------------------------------------
'''

print('*'*50)

m10=pydf2['x'].sum()
m01=pydf2['y'].sum()

cx = int(m10/len(pydf2))
cy = int(m01/len(pydf2))

re = np.sqrt(len(pydf2)/3.14)

print('Centroides (Cx,Cy) da imagem: (%d,%d)' %(cx,cy))
print('Raio equivalente (m00/pi): %5.3f'%re)

f = open('parametros_hu.txt', 'w')
f.write('#Populacao Rm/re   std I1  I2  I3  I4  I5  I6  I7  a   b   f=a+b/2 tetha   Exc     flong   \n')
f.close()

def Humoments(pca,cx,cy,p):
    pca['raio'] = np.sqrt((pca['x'] - cx)**2 + (pca['y'] - cy)**2)
    Rm = pca['raio'].mean()
    SigR = pca['raio'].std()
    Rm_re = pca['raio'].mean()/re
    SigR_re = pca['raio'].std()/re
    print('')
    print('/'*70)
    print('Hu moments - Populacao %d' %p)
    print('Raio medio Rm: %5.3f +- %5.3f' %(Rm,SigR))
    print('Raio madio ponderado Rm/Re: %5.3f +- %5.3f' %(Rm_re,SigR_re))

    file = open('parametros_hu.txt', 'a')
    '''
    Momentos da imagem
    Ver: http://pt.wikipedia.org/wiki/Momentos_Invariantes_de_uma_Imagem
    '''

    #nao centrais de 2 ordem
    m11=(pca['x']*pca['y']).sum()
    m20=(pca['x']*pca['x']).sum()
    m02=(pca['y']*pca['y']).sum()

    #centrais de 3 ordem
    mu_11=((pca['x']-cx)*(pca['y']-cy)).sum()
    mu_20=((pca['x']-cx)**2).sum()
    mu_02=((pca['y']-cy)**2).sum()

    mu_12=((pca['x']-cx)*((pca['y']-cy)**2)).sum()
    mu_21=((pca['y']-cx)*((pca['x']-cx)**2)).sum()

    mu_30=((pca['x']-cx)**3).sum()
    mu_03=((pca['y']-cy)**3).sum()

    #centrais de 4 ordem
    mu_40=((pca['x']-cx)**4).sum()
    mu_04=((pca['y']-cy)**4).sum()

    #print('Momentos nao centrais m11, m20, m02: %d, %d, %d' %(m11, m20, m02))
    #print('Momentos centrais mu_11, mu_20, mu_02: %d, %d, %d' %(mu_11, mu_20, mu_02))
    #print('Momentos centrais mu_12, mu_21: %d, %d' %(mu_12, mu_21))

    #Momentos invariantes por escala n_ij
    n11=mu_11/(len(pca)**2)
    n12=mu_12/(len(pca)**2.5)
    n21=mu_21/(len(pca)**2.5)
    n02=mu_02/(len(pca)**2)
    n20=mu_20/(len(pca)**2)
    n30=mu_30/(len(pca)**2.5)
    n03=mu_03/(len(pca)**2.5)

    #Momentos invariantes por translacao, escala e rotacao
    #Invariantes de Hu
    I1 = n02 + n20
    I2 = ((n20-n02)**2) + 4*((n11)**2)
    I3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    I4 = (n30 + 3*n12)**2 + (3*n21 + n03)**2
    I5 = (n30 - 3*n12)*(n30 + n12)*(((n30 + n12)**2) - 3*((n21 + n03)**2)) + (3*n21 - n03)*(n12 + n03)*(3*((n30 + n12)**2) - ((n21 + n03)**2))
    I6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03)
    I7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)

    print('Hu Moments: %5.2e, %5.2e, %5.2e, %5.2e, %5.2e, %5.2e, %5.2e' %(I1, I2, I3, I4, I5, I6, I7))

    #Parametros da Elipse
    dd = (mu_20 + mu_02)
    ee = (mu_20 - mu_02)*(mu_20 - mu_02) + 4*(mu_11)*(mu_11)
    a = np.sqrt((2*(dd + np.sqrt(ee)))/len(pca))
    b = np.sqrt((2*(dd - np.sqrt(ee)))/len(pca))
    print('')
    print('Parametros obtidos')
    print('Semi-eixos da elipse (a,b): %5.3f, %5.3f' %(a,b))

    #Razao dos semi-eixos
    f = (a+b)/2
    print('razao dos eixos (a+b/2): %5.3f' %f)

    #Orientacao da Elipse
    tetha = 0.5*np.arctan((2*mu_11)/(mu_20 - mu_02))
    print('Orientacao da elipse, tetha: %5.3f' %tetha)

    #Excentricidade
    exc = 1 - (b/a)
    print('Excentricidade da Elipse, e: %5.3f' %exc)

    #FATOR DE ELONGACAO
    flong = np.sqrt((mu_02/mu_20))
    print('Fator de elongacao, flong: %5.3f' %flong)

    file.write('%s %s %s %s %s %s %s %s %s  %s %s %s %s %s %s %s\n' %(p, Rm_re, SigR_re, I1, I2, I3, I4, I5, I6, I7, a, b, f,
                tetha, exc, flong))
    '''
    Simetria
    '''
    cte = cy - tetha*cx
    sym1 = pca.ix[pca['y'] >= tetha*pca['x'] + cte]
    sym2 = pca.ix[pca['y'] < tetha*pca['x'] + cte]

    SYM = 1 - (math.fabs(len(sym1) - len(sym2))/(len(sym1) + len(sym2)))
    print('')
    print('Parametro de simetria: %5.4f' %SYM)

    arr1=pca['x']
    arr2=pca['y']
    n=len(pca)
    l=1
    fof = fof_fortran.fof(arr1, arr2,l,n)
    df_fof=pd.DataFrame(fof)
    df_fof.columns = ['fof']
    print(df_fof.describe())




    return()


Humoments(pca1,cx,cy,p=1)
Humoments(pca2,cx,cy,p=2)
Humoments(pca3,cx,cy,p=3)
Humoments(pca4,cx,cy,p=4)




#f.close()

print('')
print("-="*34)
fim = time.time()
time_proc = fim - ini
print('tempo de processamento: %fs' %time_proc)
