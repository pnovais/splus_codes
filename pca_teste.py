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


data = pd.read_table("gal_old.txt", delim_whitespace=True, header=None)

#selecting just some columns to apply pca

data2=data.ix[:,2:13]

data2.columns = ['flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378','flux_J0395',
                'flux_J0410','flux_J0430','flux_J0515','flux_J0660','flux_J0861']

columns_name=['u-r','g-r','i-r','z-r','0378-r','0395-r','0410-r','0430-r','0515-r','0660-r','0861-r']
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

'''
nova tentativa de calcular o pca e sua projecao
'''

Z = (data3-data3.mean())/data3.std()
Z.index.name = "Medidas"
Z.T

pca = PCA(n_components=11, copy=True)
X = pca.fit_transform(Z)
df_X = DataFrame(X)

print(pca.explained_variance_ratio_)

line = dict(linewidth=1, linestyle='--', color='k')
marker = dict(linestyle='none', marker='o', markersize=7, color='blue', alpha=0.5)

fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], **marker)
ax.set_xlim([-20, 80])
ax.set_ylim([-15, 15])
ax.axhline(**line)
ax.axvline(**line)
_ = ax.set_title("PCA")

loadings = DataFrame(pca.components_.T)
loadings.index = ['PC %s' % pc for pc in loadings.index + 1]
loadings.columns = ['TS %s' % pc for pc in loadings.columns + 1]



plt.show()



'''
line = dict(linewidth=1, linestyle='--', color='k')
marker = dict(linestyle='none', marker='o', markersize=7, color='blue', alpha=0.5)

cov = np.cov(Z.T)
U, S, V = np.linalg.svd(cov, full_matrices=True, compute_uv=True)
'''

'''
pca = PCA(n_components=11, copy=True)
X = pca.fit_transform(data3)

ax.plot(X[:,0], X[:,1])
'''

#print(data3.head())

'''
cov_data = np.cov(np.transpose(data3))
w,v=la.eig(cov_data)

ind = np.argsort(w)[::-1]
w_dec=w[ind]
v_dec=v[ind]

EVR=w/np.sum(w)
print(EVR)
'''

data4 = data3.sort_values(by=['g-r'], ascending=True)

#print(data4.head())

for i in range(0,2000):
    data5=data4

for i in range(2000,4000):
    data6=data4

for i in range(4000,6000):
    data7=data4

for i in range(6000,7259):
    data8=data4
