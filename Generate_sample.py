# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 10:20
# @Author  : LiYun
# @File    : Generate_sample.py
'''Description :
用来产生航迹样本的文件
'''
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def Coor2Polar(x,c):
    '''
    将x的最后一维从直角坐标转换到极坐标
    :param x: [500,2] X,Y
    :param c: [2] 雷达坐标
    :return: y: [500,2] 距离，角度
    '''

    # plt.plot(x[:,0],x[:,1],'.')
    # plt.plot(c[0],c[1],'+')
    # plt.show()

    y=np.empty_like(x)
    if x.ndim==1:
        y[0]=np.hypot(x[0]-c[0],x[1]-c[1])
        y[1]=np.arctan2(x[1]-c[1],x[0]-c[0])
    elif x.ndim==2:
        y[:,0]=np.hypot(x[:,0]-c[0],x[:,1]-c[1])
        y[:,1]=np.arctan2(x[:,1]-c[1],x[:,0]-c[0])
    elif x.ndim==3:
        y[:,:,0]=np.hypot(x[:,:,0]-c[0],x[:,:,1]-c[1])
        y[:,:,1]=np.arctan2(x[:,:,1]-c[1],x[:,:,0]-c[0])
    return y

def Polar2Coor(x,c):
    '''
    将x的最后一维从极坐标转换到直角坐标
    :param x: [500,2] 距离，角度
    :param c: [2] 雷达坐标
    :return: y: [500,2] X,Y
    '''
    y=np.empty_like(x)
    if x.ndim==1:
        y[0]=x[0]*np.cos(x[1])+c[0]
        y[1]=x[0]*np.sin(x[1])+c[1]
    elif x.ndim==2:
        y[:,0]=x[:,0]*np.cos(x[:,1])+c[0]
        y[:,1]=x[:,0]*np.sin(x[:,1])+c[1]
    elif x.ndim==3:
        y[:,:,0]=x[:,:,0]*np.cos(x[:,:,1])+c[0]
        y[:,:,1]=x[:,:,0]*np.sin(x[:,:,1])+c[1]
    return y

def plot_tracks(tracks,measures,radarPos,pltTrack,pltMeas):
    '''
    绘制航迹和量测
    输入变量：
        tracks: [sample_nums,Frame,4]
        measures: [sample_nums,Frame,2]
        radarPos: [4,0]
        pltTrack: 是否绘制航迹
        pltMeas: 是否绘制量测
    '''
    if pltTrack:
        plt.figure(figsize=(6, 6))
        for t in range(tracks.shape[0]):
            plt.plot(tracks[t,:,0],tracks[t,:,2],'o-', markersize=4,mfc='none')
        plt.xlabel("x in meters",fontsize=12)
        plt.ylabel("y in meters",fontsize=12)
        # plt.xlim([0,12])
        # plt.ylim([0,12])
        plt.axis('equal')
        plt.grid(linestyle='dotted')
        plt.plot(radarPos[0],radarPos[1],'*',label='radar', markersize=10)
        plt.legend()

    if pltMeas:
        tracks=Polar2Coor(measures,radarPos)
        plt.figure(figsize=(6, 6))
        for t in range(tracks.shape[0]):
            plt.plot(tracks[t,:,0],tracks[t,:,1],'o-', markersize=4,mfc='none')
        plt.xlabel("x in meters",fontsize=12)
        plt.ylabel("y in meters",fontsize=12)
        # plt.xlim([-2,12])
        # plt.ylim([-2,12])
        plt.axis('equal')
        plt.grid(linestyle='dotted')
        plt.plot(radarPos[0],radarPos[1],'*',label='radar', markersize=10)
        plt.legend()

    plt.show()

def Banana_meas_Lolja(radarPos,measureC,rangeGaussian,angleGaussian):
    '''
    在Lolja论文中，在measure附近加上高斯噪声产生banana量测的场景
    '''
    Range=np.hypot(measureC[0]-radarPos[0],measureC[1]-radarPos[1])#量测的径向距离
    angle=np.arctan2(measureC[1]-radarPos[1],measureC[0]-radarPos[0])#量测的极坐标
    measures=np.empty([1000,2])
    for i in range(1000):
        measures[i,0]=Range+np.random.normal(loc=rangeGaussian[0],scale=np.sqrt(rangeGaussian[1]))
        measures[i,1]=angle+np.random.normal(loc=angleGaussian[0],scale=np.sqrt(angleGaussian[1]))
    measures=Polar2Coor(measures,radarPos)
    plt.plot(radarPos[0],radarPos[1],'*',label='radar')
    plt.plot(measures[:,0],measures[:,1],'.',label='noise measurements')
    plt.plot(measureC[0],measureC[1],'x',label='true target')
    # plt.title('Measurements')
    plt.xlabel("x in meters",fontsize=12)
    plt.ylabel("y in meters",fontsize=12)
    plt.legend(prop={'size':14})
    # plt.xlim([-2,12])
    # plt.ylim([-2,12])
    plt.axis('equal')
    plt.grid(linestyle='dotted')
    # plt.legend(loc='upper left')
    plt.legend()
    plt.show()

class systemModel:
    '''系统模型，这里假定是线性系统'''
    def __init__(self, deltaT,processNoiseX,processNoiseY):
        '''
        :param deltaT: 雷达扫描间隔， 1s
        :param processNoiseX: 过程噪声沿X轴的均值和方差，[0,0.1]
        :param processNoiseY: 过程噪声沿Y轴的均值和方差，[0,0.1]
        '''
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
        self.processNoiseX=processNoiseX
        self.processNoiseY=processNoiseY

    def transform(self,track):
        '''
        输入t0时刻的状态，输出t1时刻的状态
        :param track: [4]
        :return: track: [4]
        '''
        accNoiseVx=np.random.normal(loc=self.processNoiseX[0], scale=np.sqrt(self.processNoiseX[1]))
        accNoiseVy=np.random.normal(loc=self.processNoiseY[0], scale=np.sqrt(self.processNoiseY[1]))
        return np.dot(self.G, track) + np.dot(self.Q, np.array([accNoiseVx, accNoiseVy]))

    def transformBatch(self,track):
        '''
        批转换，输入t0时刻的状态，输出t1时刻的状态
        :param track: [500,4]
        :return: track: [500,4]
        '''
        batchsize=track.shape[0]
        accNoiseVx=np.random.normal(loc=self.processNoiseX[0], scale=np.sqrt(self.processNoiseX[1]),size=[batchsize,1])
        accNoiseVy=np.random.normal(loc=self.processNoiseY[0], scale=np.sqrt(self.processNoiseY[1]),size=[batchsize,1])
        return (np.dot(self.G, track.T) + np.dot(self.Q, np.concatenate((accNoiseVx,accNoiseVy),axis=1).T)).T

class measurementModel:
    '''量测模型，这里是非线性系统'''
    def __init__(self, radarPos,measNoiseD,measNoiseA):
        '''
        :param radarPos: 雷达的位置，[4,0]
        :param measNoiseD: 量测噪声中距离向的均值和方差，[0,0.5]
        :param measNoiseA: 量测噪声中角度向的均值和方差，[0,0.3]
        '''
        self.radarPos=radarPos
        self.measNoiseD=measNoiseD
        self.measNoiseA=measNoiseA

    def transform(self,track):
        '''
        输入t1时刻的状态，输出t1时刻的量测
        :param track: [4]
        :return: measure: [2]
        '''
        a=np.hypot(track[0]-self.radarPos[0],track[2]-self.radarPos[1])+\
                        np.random.normal(loc=self.measNoiseD[0], scale=np.sqrt(self.measNoiseD[1]))
        b=np.arctan2(track[2]-self.radarPos[1],track[0]-self.radarPos[0])+\
                        np.random.normal(loc=self.measNoiseA[0], scale=np.sqrt(self.measNoiseA[1]))
        return np.array([a,b])

    def transformBatch(self,track,addNoise=True):
        '''
        批转换，输入t1时刻的状态，输出t1时刻的量测
        :param track: [500,4]
        :param addNoise: 是否加噪声
        :return: measure: [500,2]
        '''
        batchsize=track.shape[0]
        a=np.hypot(track[:,0]-self.radarPos[0],track[:,2]-self.radarPos[1])
        if addNoise: a+=np.random.normal(loc=self.measNoiseD[0], scale=np.sqrt(self.measNoiseD[1]),size=batchsize)
        b=np.arctan2(track[:,2]-self.radarPos[1],track[:,0]-self.radarPos[0])
        if addNoise: b+=np.random.normal(loc=self.measNoiseA[0], scale=np.sqrt(self.measNoiseA[1]),size=batchsize)
        return np.concatenate((a[:,np.newaxis],b[:,np.newaxis]),axis=1)

def generate_tracks_Lolja(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                          processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame):
    '''
    在Lolja论文中，生成航迹的方法
    输入变量：
        sample_file: 生成航迹存储的文件夹，data/track
        sample_nums: 样本数量，50000
        isLoad: 是否加载原有文件
        radarPos: 雷达的位置，[4,0]
        iniState: 目标的初始状态，[0,1,0,1]
        deltaT: 雷达扫描间隔， 1s
        processNoiseX: 过程噪声沿X轴的均值和方差，[0,0.1]
        processNoiseY: 过程噪声沿Y轴的均值和方差，[0,0.1]
        measNoiseD: 量测噪声中距离向的均值和方差，[0,0.5]
        measNoiseA: 量测噪声中角度向的均值和方差，[0,0.3]
        Frame: 航迹一共有几帧，10
    输出变量：
        tracks: [sample_nums,Frame,4]
        measures: [sample_nums,Frame,2]
    '''
    if isLoad == True:
        temp = np.load(sample_file)
        return temp['tracks'],temp['measures']
    else: #生成航迹
        tracks=np.empty([sample_nums,Frame,4])
        measures=np.empty([sample_nums,Frame,2]) #径向距离和角度
        SM=systemModel(deltaT,processNoiseX,processNoiseY)
        MM=measurementModel(radarPos,measNoiseD,measNoiseA)
        for i in range(sample_nums):
            if i%10000==0: print('tracks generating: '+str(i))
            tracks[i,0]=iniState
            for j in range(1,Frame):
                tracks[i, j]=SM.transform(tracks[i, j-1])
        for j in range(Frame):
            addNoise=True if j else False # 第一帧不加噪声
            measures[:,j]=MM.transformBatch(tracks[:,j],addNoise)
        np.savez(sample_file,tracks=tracks, measures=measures)
        return tracks, measures

def generate_tracks_LeeYun(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                          processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame):
    '''
    生成航迹的方法
    输入变量：
        sample_file: 生成航迹存储的文件夹，data/track
        sample_nums: 样本数量，50000
        isLoad: 是否加载原有文件
        radarPos: 雷达的位置，[0,0]
        iniState: 目标的初始状态，[sample_nums,4]
        deltaT: 雷达扫描间隔， 1s
        processNoiseX: 过程噪声沿X轴的均值和方差，[0,0.1]
        processNoiseY: 过程噪声沿Y轴的均值和方差，[0,0.1]
        measNoiseD: 量测噪声中距离向的均值和方差，[0,0.5]
        measNoiseA: 量测噪声中角度向的均值和方差，[0,0.3]
        Frame: 航迹一共有几帧，20
    输出变量：
        tracks: [sample_nums,Frame,4]
        measures: [sample_nums,Frame,2]
    '''
    if isLoad == True:
        temp = np.load(sample_file)
        return temp['tracks'],temp['measures']
    else: #生成航迹
        tracks=np.empty([sample_nums,Frame,4])
        measures=np.empty([sample_nums,Frame,2]) #径向距离和角度
        SM=systemModel(deltaT,processNoiseX,processNoiseY)
        MM=measurementModel(radarPos,measNoiseD,measNoiseA)
        for i in range(sample_nums):
            if i%10000==0: print('tracks generating: '+str(i))
            tracks[i,0]=iniState[i]
            for j in range(1,Frame):
                tracks[i, j]=SM.transform(tracks[i, j-1])
        for j in range(Frame):
            measures[:,j]=MM.transformBatch(tracks[:,j])
        np.savez(sample_file,tracks=tracks, measures=measures)
        return tracks, measures

def windowSample(data,sample_nums,Frame,window):
    '''
    根据滑窗长度提取样本
    :param data: [sample_nums,Frame,X]
    :param sample_nums: 10000
    :param Frame: 20
    :param window: 10
    :return: newData: [sample_nums*Frame,window,X]
    '''
    X=data.shape[2]
    # 从第 1 帧开始估计，前面需要补上 window-1 个 nan
    # data [sample_nums,Frame+window-1,X]
    data=np.concatenate((np.zeros([sample_nums,window-1,X])*np.nan,data),axis=1)
    # 制作样本集
    # data [sample_nums,Frame,window,X]
    newData=np.empty([sample_nums,Frame,window,X])
    for i in range(Frame): newData[:,i]=data[:,i:i+window]
    # reshape
    # newData [sample_nums*Frame,window,X]
    newData=newData.reshape([sample_nums*Frame,window,X])
    return newData

def centralizeMeas(measures):
    '''
    中心化量测数据
    :param measures: [sample,window,2]
    :return: measures: [sample,window,2]
    '''
    measures[:,:-1]=measures[:,:-1]-measures[:,-1,np.newaxis]
    return measures

if __name__=='__main__':
    path_head='../20180405_data/Lolja/'

    isLoad=False
    pltTrack=True
    pltMeas=True
    radarPos=[4,0]
    sample_nums=100
    iniState=np.array([0,1,0,1])
    deltaT=1
    processNoiseX=[0,0.05]
    processNoiseY=[0,0.05]
    measNoiseD=[0,0.3]
    measNoiseA=[0,0.1]
    Frame=10
    sample_file=path_head+'tracks_sample_%d_%.2f_%.2f_%.2f_%.2f.npz'%\
                          (sample_nums,processNoiseX[1],processNoiseY[1],measNoiseD[1],measNoiseA[1])

    # 产生航迹并绘图检验
    # tracks: [sample_nums,Frame,4]
    # measures: [sample_nums,Frame,2]
    tracks,measures=generate_tracks_Lolja(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                                           processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)
































