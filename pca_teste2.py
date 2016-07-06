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


data.columns = ['x','y','flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378','flux_J0395',
                'flux_J0410','flux_J0430','flux_J0515','flux_J0660','flux_J0861','index','fof']

columns_name=['u-r','g-r','i-r','z-r','0378-r','0395-r','0410-r','0430-r','0515-r','0660-r','0861-r']

data2=data.ix[:,2:13]
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
#    data3['0861-r'] = data2['flux_J0861'] - data2['flux_r']



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

print(len(data5))
