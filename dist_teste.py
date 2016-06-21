"""
Programa teste distancias

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
import math

ini=time.time()

def distance(x1,y1,x2,y2):
    dist = (x1-x2)**2 + (y1-y2)**2
    d = np.sqrt(dist)
    return d


df = pd.read_table('teste.txt', delim_whitespace=True, header=None)
df.columns=['x','y','flux_u','flux_g','flux_r','flux_i','flux_z','flux_J0378','flux_J0395','flux_J0410', 'flux_J0430','flux_J0515','flux_J0660','flux_J0861']

labels = []

#criando os rotulos iniciais
for row in range(0,len(df)):
    labels.append(row)

df['label'] = labels

#print(df.tail())

n=0

for i in range(0,len(df)/10):
    for j in range(i+1,len(df)/10):
        dd = distance(df['x'][i], df['y'][i], df['x'][j], df['y'][j])
        if(dd==1):
            ii=min(df['label'][i],df['label'][j])
            iclio=df['label'][i]
            icljo=df['label'][j]
            #print(ii)
            df['label'][i]=ii
            df['label'][j]=ii
#            print(ii)
            for k in range(0,len(df)/10):
                if(df['label'][k]==iclio | df['label'][k]==icljo):
                    df['label'][k]==ii

#print(df.tail())

#        print(df['x'][i],df['y'][i],df['x'][j], df['y'][j],dd)
#        if(dd==1):


print('------')

#print(n)

fim = time.time()
time_proc = fim - ini
print('tempo de processamento: %fs' %time_proc)
