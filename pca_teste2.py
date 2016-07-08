'''
--------Paty Novais--------
----------02/07/16---------

Teste para pca.

'''

#!/usr/bin/python
# -*- coding: utf-8 -*-


print(__doc__)

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
import matplotlib.mlab as mlab
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
import numpy.linalg as la
import scipy, pylab
from pandas import DataFrame
import seaborn


ini=time.time()

data = pd.read_table("gal_old.txt", delim_whitespace=True, header=None)

#selecting just some columns to apply pca


data.columns = ['x','y','flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378','flux_J0395',
                'flux_J0410','flux_J0430','flux_J0515','flux_J0660','flux_J0861','index','fof']

columns_name=['u-r','g-r','i-r','z-r','0378-r','0395-r','0410-r','0430-r','0515-r','0660-r','0861-r']

data2=data.ix[:,2:15]
data2a=data.ix[:,:2]


data3 = pd.DataFrame(index=range(0,len(data2)),columns=columns_name)

for i in range(0,len(data2)):
    data3['u-r'] = data2['flux_u'] - data2['flux_r']
    data3['g-r'] = data2['flux_g'] - data2['flux_r']
    data3['i-r'] = data2['flux_i'] - data2['flux_r']
    data3['z-r'] = data2['flux_z'] - data2['flux_r']
    data3['0378-r'] = data2['flux_J0378'] - data2['flux_r']
    data3['0395-r'] = data2['flux_J0395'] - data2['flux_r']
    data3['0410-r'] = data2['flux_J0410'] - data2['flux_r']
    data3['0430-r'] = data2['flux_J0430'] - data2['flux_r']
    data3['0515-r'] = data2['flux_J0515'] - data2['flux_r']
    data3['0660-r'] = data2['flux_J0660'] - data2['flux_r']
    data3['0861-r'] = data2['flux_J0861'] - data2['flux_r']



data3a=data3.join(data2a)

data4 = data3a.sort_values(by=['g-r'], ascending=True)


compr=len(data4)
labels = []

#criando os rotulos iniciais
for row in range(0,compr):
    labels.append(row)
data4['label'] = labels

print(data4.head())


data5=data4.ix[(data4['label'] >= 0) & (data4['label'] < int(compr/4))]
data6=data4.ix[(data4['label'] >= int(compr/4) ) & (data4['label'] < int(compr/2))]
data7=data4.ix[(data4['label'] >= int(compr/2)) & (data4['label'] < int(3*compr/4))]
data8=data4.ix[(data4['label'] >= int(3*compr/4))]

print("*"*80)
print(data5.tail())


f, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
ax1.scatter(data5['x'],data5['y'], color="red")
ax2.scatter(data6['x'],data6['y'], color="yellow")
ax3.scatter(data7['x'],data7['y'], color="green")
ax4.scatter(data8['x'],data8['y'], color="blue")



plt.figure()
plt.scatter(data5['x'],data5['y'], color="red")
plt.scatter(data6['x'],data6['y'], color="yellow")
plt.scatter(data7['x'],data7['y'], color="green")
plt.scatter(data8['x'],data8['y'], color="blue")


plt.show()

#print(len(data5))


#Momentos de Hu
'''
!======================================================
!================CENTROIDES DA IMAGEM==================
!======================================================
!xc = M10/M00
!yc = M01/M00
!
!onde xc e yc so os centrides da imagem, M10, M01 e
!M00 os momentos no centrais, onde
!
!Mpq=Somatorio(x^p.y^q)
!------------------------------------------------------
'''

print('*'*50)

m10=data4['x'].sum()
m01=data4['y'].sum()

cx = int(m10/len(data4))
cy = int(m01/len(data4))

re = np.sqrt(len(data4))

print('Centroides (Cx,Cy) da imagem: (%5.3f,%5.3f)' %(cx,cy))
print('Raio equivalente (m00/pi): %5.3f'%re)


data5['raio'] = np.sqrt((data5['x'] - cx)**2 + (data5['y'] - cy)**2)

Rm = data5['raio'].mean()
SigR = data5['raio'].std()
Rm_re = data5['raio'].mean()/re
SigR_re = data5['raio'].std()/re

print('Raio medio Rm: %5.3f' %Rm)
print('Desvio padrao Sigma_R: %5.3f' %SigR)
print('Raio medio ponderado Rm/Re: %5.3f' %Rm_re)
print('Desvio padrao ponderado Sigma_R/Re: %5.3f' %SigR_re)


'''
Momentos da imagem
Ver: http://pt.wikipedia.org/wiki/Momentos_Invariantes_de_uma_Imagem
'''

#nao centrais de 2 ordem
m11=(data5['x']*data5['y']).sum()
m20=(data5['x']*data5['x']).sum()
m02=(data5['y']*data5['y']).sum()

#centrais de 3 ordem
mu_11=((data5['x']-cx)*(data5['y']-cy)).sum()
mu_20=((data5['x']-cx)**2).sum()
mu_02=((data5['y']-cy)**2).sum()

mu_12=((data5['x']-cx)*((data5['y']-cy)**2)).sum()
mu_21=((data5['y']-cx)*((data5['x']-cx)**2)).sum()

mu_30=((data5['x']-cx)**3).sum()
mu_03=((data5['y']-cy)**3).sum()

#centrais de 4 ordem
mu_40=((data5['x']-cx)**4).sum()
mu_04=((data5['y']-cy)**4).sum()


print('Momentos nao centrais m11, m20, m02: %5.3f, %5.3f, %5.3f' %(m11, m20, m02))
print('Momentos centrais mu_11, mu_20, mu_02: %5.3f, %5.3f, %5.3f' %(mu_11, mu_20, mu_02))
print('Momentos centrais mu_12, mu_21: %5.3f, %5.3f' %(mu_12, mu_21))

print(mu_04,mu_40)
print(mu_30,mu_03)

#Momentos invariantes por escala n_ij

n11=mu_11/(len(data5)**2)
n12=mu_12/(len(data5)**2.5)
n21=mu_21/(len(data5)**2.5)
n02=mu_02/(len(data5)**2)
n20=mu_20/(len(data5)**2)
n30=mu_30/(len(data5)**2.5)
n03=mu_03/(len(data5)**2.5)

print(n20,n02)

#Momentos invariantes por translacao, escala e rotacao
#Invariantes de Hu

I1 = n02 + n20
I2 = ((n20-n02)**2) + 4*((n11)**2)
I3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
I4 = (n30 + 3*n12)**2 + (3*n21 + n03)**2
I5 = (n30 - 3*n12)*(n30 + n12)*(((n30 + n12)**2) - 3*((n21 + n03)**2)) + (3*n21 - n03)*(n12 + n03)*(3*((n30 + n12)**2) - ((n21 + n03)**2))
I6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03)
I7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)


print(I1, I2, I3, I4, I5, I6, I7)

#Parametros da Elipse
dd = (mu_20 + mu_02)
ee = (mu_20 - mu_02)*(mu_20 - mu_02) + 4*(mu_11)*(mu_11)

a = np.sqrt((2*(dd + np.sqrt(ee)))/len(data5))
b = np.sqrt((2*(dd - np.sqrt(ee)))/len(data5))

print(a,b)

#Razao dos semi-eixos
f = (a+b)/2

print('razao dos eixos: %5.3f' %f)

#Orientacao da Elipse

tetha = 0.5*np.arctan((2*mu_11)/(mu_20 - mu_02))
print('tetha: %5.3f' %tetha)

#Excentricidade
exc = 1 - (b/a)

#FATOR DE ELONGACAO

flong = np.sqrt((mu_02/mu_20))
print(flong)








print('')
print('*'*70)
fim = time.time()
time_proc = fim - ini
print('tempo de processamento: %fs' %time_proc)
