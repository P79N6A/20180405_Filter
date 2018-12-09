# -*- coding: utf-8 -*-
# @Time    : 2018/4/6 19:58
# @Author  : LiYun
# @File    : KalmanSmoother.py
'''Description :

'''
import os
import pickle
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.neighbors import KernelDensity
from Generate_sample import systemModel,Coor2Polar,Polar2Coor

def plotParticle(sequence,radar):
    '''
    各时刻的量测值和粒子群位置
    :param sequence: [[z0,p0]...[zt,pt]] z0[2] p0[500,2]
    :param radar: [2] 雷达位置
    '''
    for i,item in enumerate(sequence):
        z,p=item
        z=Polar2Coor(z,radar)
        plt.plot(p[:,0],p[:,1],'.',label='particles t%d'%(i))
        plt.plot(z[0],z[1],'*',label='measurement t%d'%(i))
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.show()

class RegularizedParticleFilter:
    '''正则化粒子滤波器'''
    def __init__(self, nums, sigR, sigA, T, radar,processNoiseX,processNoiseY):
        '''
        对滤波器初始化，输入t0,t1的量测，和需要的粒子数量，初始化t1的粒子群
        :param nums: 粒子数量，1000
        :param sigR: 距离向的标准差，0.2m
        :param sigA: 角度向的标准差，0.001rad
        :param T: 雷达扫描周期
        :param radar: 雷达位置，[4,0]
        :param processNoiseX: 过程噪声沿X轴的均值和方差，[0,0.1]
        :param processNoiseY:  过程噪声沿Y轴的均值和方差，[0,0.1]
        '''
        self.nums=nums
        self.sigR=sigR
        self.sigA=sigA
        self.T=T
        self.radar=radar
        self.system=systemModel(T,processNoiseX,processNoiseY)
        self.invR=LA.inv(np.array([[sigR**2,0],
                                   [0,sigA**2]])) # 极坐标下的量测协方差的逆
        self.Preprocess_Epanechnikov_kernel()

    def init_track(self, x0, x1):
        '''
        目标跟踪初始化
        :param x0: 第一帧的量测，极坐标，[距离，角度]
        :param x1: 第二帧的量测，极坐标，[距离，角度]
        '''
        # 得到t0,t1时刻的粒子群
        a=np.random.normal(0,self.sigR,[self.nums,2]) #[500,2]
        b=np.random.normal(0,self.sigA,[self.nums,2]) #[500,2]
        p0=np.concatenate((a[:,0,np.newaxis],b[:,0,np.newaxis]),axis=1)+x0 # t0时刻粒子群极坐标位置,[500,2]
        p1=np.concatenate((a[:,1,np.newaxis],b[:,1,np.newaxis]),axis=1)+x1 # t1时刻粒子群极坐标位置,[500,2]
        # 转换到直角坐标
        p0=Polar2Coor(p0,self.radar) #[500,2],x,y
        p1=Polar2Coor(p1,self.radar) #[500,2],x,y
        # 求出t1时刻粒子群的速度
        v1=(p1-p0)/self.T
        # 得到t1时刻的粒子群及权重
        self.particles=np.concatenate((p1[:,0,np.newaxis],v1[:,0,np.newaxis],p1[:,1,np.newaxis],v1[:,1,np.newaxis]),axis=1) #[500,4]
        self.weights=np.ones(self.nums)/self.nums #[500]
        # 记录各个时刻的量测值和粒子群位置
        self.sequence=[]
        self.sequence.append([x0,p0])
        self.sequence.append([x1,p1])
        # 绘图检验程序
        # plt.plot(p0[:,0],p0[:,1],'.')
        # plt.plot(p1[:,0],p1[:,1],'.')
        # y0=Polar2Coor(x0,self.radar)
        # y1=Polar2Coor(x1,self.radar)
        # self.particles =self.system.transformBatch(self.particles)
        # plt.plot(self.particles[:,0],self.particles[:,2],'.')
        # plt.plot(y0[0],y0[1],'x')
        # plt.plot(y1[0],y1[1],'x')
        # plt.plot(self.radar[0],self.radar[1],'+')
        # plt.show()

    def estimate(self,particles,weights):
        '''
        根据当前的粒子群估算目标的状态及协方差
        :param particles: [500,4]
        :param weights: [500]
        :return: state[4], conv[4,4]
        '''
        state=np.dot(weights,particles) #[4]
        error=particles-state #[500,4]
        conv=np.dot(error.T*weights,error)
        return state,conv

    def Resample(self,particles,weights):
        '''
        标准SIR重采样
        :param particles: [500,4]
        :param weights: [500]
        :return: particles: [500,4] ,weights: [500]
        '''
        nums=self.nums
        edges=np.concatenate((np.zeros(1),np.cumsum(weights)))
        # 抽一个初始随机点
        u1=np.random.rand()/nums
        idx = np.digitize(np.arange(u1,1,1/nums),edges)-1
        # if idx.max()>=1000:
        #     a=1
        return particles[idx],np.ones(nums) / nums

    def Preprocess_Epanechnikov_kernel(self):
        '''
        Epanechnikov_kernel中需要的变量的预处理
        '''
        self.nx=4
        self.Cnx=0.5*np.pi**2 # 四维单位球体积
        self.A=(8/self.Cnx*(self.nx+4)*(2*np.sqrt(np.pi))**self.nx)**(1/(self.nx+4))
        self.h_opt=self.A*self.nums**(-1/(self.nx+4))

        EPC=KernelDensity(kernel='epanechnikov')
        EPC.fit(np.zeros([1,4]))
        mgrid = np.lib.index_tricks.nd_grid()
        self.Y=mgrid[0:51,0:51,0:51,0:51]
        self.Y=np.reshape(self.Y,[4,-1]).T/25-1 # [51*51*51*51,4] [-1,1]
        pdf=np.exp(EPC.score_samples(self.Y)) # [51*51*51*51]
        pdf /= np.sum(pdf)
        X=np.arange(0,51*51*51*51)
        np.random.shuffle(X)
        self.Y=self.Y[X]
        pdf=pdf[X]
        self.edges=np.concatenate((np.zeros(1),np.cumsum(pdf)))

    def Epanechnikov_kernel(self,nums):
        '''
        从Epanechnikov_kernel中随机采样
        :param nums:
        :return: ei [500,4]
        '''
        u1=np.random.rand()/nums
        idx = np.digitize(np.arange(u1,1,1/nums),self.edges)-1
        return self.Y[idx]

    def filter(self, z0,z):
        '''
        输入t0和t1时刻的量测值，输出t1的直角坐标下的位置
        t0时刻的量测值是用于当滤波器发散时，重新初始化粒子群的
        :param z0: [2]
        :param z: [2]
        :return: x: [2]
        '''
        # sample from prior density
        self.particles=self.system.transformBatch(self.particles) # [500,4]
        # self.sequence.append([z,self.particles[:,[0,2]]])
        # convert to measurement coordinates and calculate weights
        ptcPolar=Coor2Polar(self.particles[:,[0,2]],self.radar) # 极坐标下的粒子位置 [500,2]
        self.weights=np.exp(-0.5*np.sum(np.dot(z - ptcPolar,self.invR)*(z - ptcPolar),axis=1)) #[500]
        # 如果滤波器发散，重新初始化粒子群
        if self.weights.max()<1e-4:
            # 画出粒子群发散的场景
            # plotParticle(self.sequence,self.radar)
            self.init_track(z0,z)
            state, conv = self.estimate(self.particles, self.weights)
            return state[[0,2]]
        else: self.weights/=np.sum(self.weights)
        # 计算极坐标下的均值，并转换到直角坐标
        x=Polar2Coor(np.dot(self.weights,ptcPolar),self.radar)
        # 转换到直角坐标，并计算粒子的均值和协方差
        state, conv=self.estimate(self.particles,self.weights)
        # 重采样
        self.particles,self.weights=self.Resample(self.particles,self.weights)
        # # 找到 D 使得 DD'=conv
        # D=LA.cholesky(conv) # [4,4]
        # # 从Epanechnikov kernel中随机采样
        # ei=self.Epanechnikov_kernel(self.nums) # [500,4]
        # # 在粒子中加入采样后的噪声
        # self.particles+=self.h_opt*np.dot(ei,D.T)
        return x

if __name__=='__main__':
    processNoiseX=[0,0.05]
    processNoiseY=[0,0.05]
    x=RegularizedParticleFilter(nums=500, x0=[4,np.pi], x1=[3,5/6*np.pi], sigR=0.02, sigA=0.1,T=1,radar=[4,0],
                                processNoiseX=processNoiseX,processNoiseY=processNoiseY)
    x.Epanechnikov_kernel(500)









































