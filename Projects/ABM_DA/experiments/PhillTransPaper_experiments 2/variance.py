# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:26:43 2020

@author: vijay
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


orig=pd.read_csv('exp4_100p_orig.csv')
temper=pd.read_csv('exp4_100p_temper.csv')
temper5=pd.read_csv('exp4_temper_change.csv')
#origgroup=orig.groupby(np.arange(len(orig))//20).mean()
#tempergroup=temper.groupby(np.arange(len(temper))//20).mean()


plt.plot(temper5['Window'],temper5['Error'],label='tempered (5 max)')
plt.plot(temper['Window'],temper['Error'],label='tempered (2.5 max)')
plt.plot(orig['Window'],orig['Error'],label='original')
plt.xlabel('Resample Window')
plt.ylabel('Error')
plt.legend()
plt.title('exp4 100 Particles')
plt.show()