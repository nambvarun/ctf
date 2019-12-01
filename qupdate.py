# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:59:14 2019

@author: gyshi
"""

import timeit
start_time = timeit.default_timer()

import pandas as pd
import numpy as np
from random import randint
#import matplotlib.pyplot as plt

'''
sdata  = pd.read_csv('large.csv')
sdata  = sdata.to_numpy()

ns,na = 312020,9
N = len(sdata)

qfxn = np.zeros([ns,na],float)
lrate = 0.01
dfact = 0.95

for k in range(1000):
    for i in range(N-1):
        s = sdata[i,0] - 1
        a = sdata[i,1] - 1 
        sp= sdata[i,3] - 1
        r = sdata[i,2]
        
        sn= sdata[i+1,0] - 1
        
        if sp != sn:
            continue
        
        ap = sdata[i+1,1] - 1
        qfxn[s,a] = qfxn[s,a] + lrate*(r + dfact*(qfxn[sp,ap]) - qfxn[s,a])

#plcy = np.zeros([ns,1],int)
#plcy   = np.argmax(qfxn,axis=1) + 1 

'''
plcylg = np.zeros([ns,1],int)

for i in range(ns):
    if np.max(qfxn[i]) > 0.01: 
        plcylg[i] = np.argmax(qfxn[i]) + 1
    else:
        continue

#print('time(s) = ', timeit.default_timer() - start_time)


for i in range(ns-1):
    if plcylg[i] != 0:
        if plcylg[i-1] == 0:
            plcylg[i-1] = plcylg[i]
        if plcylg[i+1] == 0:
            plcylg[i+1] = plcylg[i]            

for i in range(ns):
    if plcylg[i] == 0:
        plcylg[i] = 1#randint(1,9)
 
np.savetxt("large.policy", np.array(plcylg), fmt="%s")          
print('time(s) = ', timeit.default_timer() - start_time)