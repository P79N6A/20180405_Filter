# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 19:50
# @Author  : LiYun
# @File    : KalmanFilter.py
'''Description :

'''
import os
import pickle
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import linalg as LA
from Generate_sample import Coor2Polar,Polar2Coor

class CVKalmanFilter:
    '''匀速模型卡尔曼滤波器'''
    def __init__(self,measNoiseD, measNoiseA, deltaT, radar,processNoiseX,processNoiseY):
        '''
        初始化卡尔曼滤波器，设定先验参数
        :param measNoiseD: 量测噪声中距离向的均值和方差，[0,0.5]
        :param measNoiseA: 量测噪声中角度向的均值和方差，[0,0.3]
        :param deltaT: 雷达扫描间隔， 1s
        :param radar: 雷达的位置，[4,0]
        :param processNoiseX: 过程噪声沿X轴的均值和方差，[0,0.1]
        :param processNoiseY: 过程噪声沿Y轴的均值和方差，[0,0.1]
        '''
        self.deltaT=deltaT
        self.radar=radar
        # 状态转移矩阵
        self.G =np.array([[1, deltaT, 0,      0],
                          [0,      1, 0,      0],
                          [0,      0, 1, deltaT],
                          [0,      0, 0,      1]], np.float)
        # 过程噪声转移矩阵
        self.Q =np.array([[deltaT**2/2,           0],
                          [     deltaT,           0],
                          [          0, deltaT**2/2],
                          [          0,      deltaT]], np.float)
        # 观测矩阵
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]], np.float)
        # 极坐标系下量测噪声的协方差矩阵
        self.R_polar = np.array([[measNoiseD[1],    0],
                                 [    0,measNoiseA[1]]],np.float)
        # 过程噪声的协方差矩阵
        self.B = np.array([[processNoiseX[1],    0],
                           [    0,processNoiseY[1]]],np.float)

    def init_track(self, x0, x1):
        '''
        目标跟踪初始化,以头两个观测量作为初始状态
        :param x0: t0时刻的量测，直角坐标，[X,Y]
        :param x1: t1时刻的量测，直角坐标，[X,Y]
        '''
        R2=self.R_polar
        self.S=np.array([x1[0],(x1[0]-x0[0])/self.deltaT,x1[1],(x1[1]-x0[1])/self.deltaT]) #x,vx,y,vy
        R3=self.update_meas_cov(x1)
        self.M=np.array([[R2[0,0],   R2[0,0],    0,          0],
                        [R2[0,0],   2*R2[0,0],  0,          0],
                        [0,         0,          R2[0,0],    R2[0,0]],
                        [0,         0,          R2[0,0],    2*R2[0,0]]],np.float)
        # self.M=np.array([[1,   1,    1,          1],
        #                  [1,   1,  1,          1],
        #                  [1,         1,          1,    1],
        #                  [1,         1,          1,    1]],np.float)

    def update_meas_cov(self,z):
        '''
        输入当前的直角坐标系下的观测值，输出直角坐标系下的量测噪声协方差矩阵
        :param z: 直角坐标系下的观测值 [2]
        :return: R: 直角坐标系下的量测噪声协方差矩阵 [2,2]
        '''
        tmp=Polar2Coor(z,self.radar) # 极坐标下的观测值
        pho, theta=tmp[0],tmp[1]
        pho2=pho**2
        vpho,vtheta=self.R_polar[0,0],self.R_polar[1,1] # 极坐标下观测值的协方差
        Rxx=pho2*np.exp(-vtheta)*(np.cos(theta)**2*(np.cosh(vtheta)-1)+np.sin(theta)**2*np.sinh(vtheta))+\
            vpho*np.exp(-vtheta)*(np.cos(theta)**2*np.cosh(vtheta)+np.sin(theta)**2*np.sinh(vtheta))
        Ryy=pho2*np.exp(-vtheta)*(np.sin(theta)**2*(np.cosh(vtheta)-1)+np.cos(theta)**2*np.sinh(vtheta))+\
            vpho*np.exp(-vtheta)*(np.sin(theta)**2*np.cosh(vtheta)+np.cos(theta)**2*np.sinh(vtheta))
        Rxy=np.sin(theta)*np.cos(theta)*np.exp(-2*vtheta)*(vpho+pho2*(1-np.exp(vtheta)))
        Ryx=Rxy
        R=np.array([[Rxx,Rxy],
                    [Ryx,Ryy]],np.float)
        return R

    def filter(self, z):
        '''
        输入t1时刻的量测值，输出t1的直角坐标下的位置
        :param z: [2]
        :return: x: [2]
        '''
        # 直角坐标系下的量测噪声协方差矩阵
        R = self.update_meas_cov(z)
        error=z-LA.multi_dot([self.H,self.G,self.S])
        M2=LA.multi_dot([self.G,self.M,self.G.T]) + LA.multi_dot([self.Q,self.B,self.Q.T])
        K=LA.multi_dot([M2,self.H.T,LA.inv(LA.multi_dot([self.H,M2,self.H.T])+R)])
        self.M=np.dot(np.eye(4)-np.dot(K,self.H),M2)
        self.S=np.dot(self.G,self.S)+np.dot(K,error)
        return self.S[[0,2]]








































