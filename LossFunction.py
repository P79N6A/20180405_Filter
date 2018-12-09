# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 18:24
# @Author  : LiYun
# @File    : LossFunction.py
'''Description :

'''
import numpy as np

def Compute_RMSE(a,b,axis=None):
    return np.sqrt(np.mean(np.square(a-b),axis=axis))










































