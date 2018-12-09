# -*- coding: utf-8 -*-
# @Time    : 2018/4/12 8:07
# @Author  : LiYun
# @File    : Figure2Plot.py
'''Description :
第二组图展示Xgboost目标跟踪的参量变化情况
    1.训练样本数量
        [1000,3000,10000,20000,50000,100000]
        [1.012081,0.965389,0.919235,0.911213,0.890231,0.870439]
    2.树的数量
        [50,100,150,200,300,500,1000,2000]
        [1.393898,0.959566,0.926564,0.919235,0.911547,0.907572,0.903101,0.903128]
    3.滑动窗口的长度
        [4,7,10,13,16]
        [1.007379,0.953249,0.918272,0.903778,0.902654]
'''
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot(dataX,dataY,labelX,labelY,limX=None,limY=None):
    F=20
    plt.plot(dataX,dataY,'s-')
    if limX: plt.xlim(limX)
    if limY: plt.ylim(limY)
    plt.grid(linestyle='dotted')
    plt.xticks(fontsize=F)
    plt.yticks(fontsize=F)
    plt.xlabel(labelX,fontsize=F)
    plt.ylabel(labelY,fontsize=F)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.show()

plot(np.array([.02,.05,.10,.20,.40,.80,1.20,1.60])*1e5,[10.535371, 10.134594, 9.978468, 9.855803, 9.713731, 9.606736,9.56,9.55],'Training data','RMSE(m)')
# plot([50,100,150,200,300,500],[2721.344527, 209.790170,19.168995,9.957405,9.586067,9.394958,],'Tree count','RMSE(m)')
# plot([4,7,10,13,16],[10.835946,10.040395,9.833757,9.806652,9.795556],'Slide-window length','RMSE(m)')







































