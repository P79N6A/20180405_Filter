# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 10:54
# @Author  : LiYun
# @File    : Lolja_method.py
'''Description :
复现Lolja的方法
'''
import os
import pickle
import time
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.ensemble import RandomForestRegressor
from LossFunction import Compute_RMSE
from ParticleFilter import RegularizedParticleFilter
from Generate_sample import Banana_meas_Lolja,generate_tracks_Lolja,plot_tracks,Polar2Coor

def repeatFig1():
    '''
    在Lolja论文中，图1的仿真
    '''
    radarPos=np.array([4,0])
    measureC=np.array([10.3,10.7])
    rangeGaussian=[0,0.2]
    angleGaussian=[0,0.1]
    Banana_meas_Lolja(radarPos,measureC,rangeGaussian,angleGaussian)

def repeatFig2():
    '''
    在Lolja论文中，图2的仿真
    '''
    # 对图2仿真所需要的参数
    isLoad=False
    pltTrack=True
    pltMeas=True
    radarPos=[4,0]
    sample_nums=10
    iniState=np.array([0,1,0,1])
    deltaT=1
    processNoiseX=[0,0.05]
    processNoiseY=[0,0.05]
    measNoiseD=[0,0.03]
    measNoiseA=[0,0.01]
    Frame=10
    sample_file=path_head+'tracks_sample_%d_%.2f_%.2f_%.2f_%.2f.npz'%\
                          (sample_nums,processNoiseX[1],processNoiseY[1],measNoiseD[1],measNoiseA[1])

    # 产生航迹并绘图检验
    # tracks: [sample_nums,Frame,4]
    # measures: [sample_nums,Frame,2]
    tracks,measures=generate_tracks_Lolja(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                                           processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)
    plot_tracks(tracks[:100],measures[:100],radarPos,pltTrack,pltMeas)

    # 训练RF，观察在不同样本数量下面的性能
    # for TD in [1000,3000,10000,20000,50000,90000]: # 不同的训练样本
    #     trainX=np.reshape(measures[:TD],[TD,-1])
    #     trainY=np.reshape(tracks[:TD,:,[0,2]],[TD,-1])
    #     testX=np.reshape(measures[TD:TD+1000],[1000,-1])
    #     testY=np.reshape(tracks[TD:TD+1000,:,[0,2]],[1000,-1])
    #     # 每一帧的X和Y都要训练一个RF
    #     regr = RandomForestRegressor(n_estimators=150,
    #                                  criterion="mse",
    #                                  n_jobs=3,
    #                                  random_state=0,
    #                                  verbose=0,)
    #     regr.fit(trainX,trainY)
    #     predY=regr.predict(testX)
    #     print(predY.shape)
    #     rmse=Compute_RMSE(predY,testY)
    #     print('sample: %d, rmse: %f'%(TD,rmse))

    # 训练RF，观察在不同树的数量下面的性能
    # for trees in [50,100,150,200]: # 不同的训练样本
    #     TD=20000
    #     trainX=np.reshape(measures[:TD],[TD,-1])
    #     trainY=np.reshape(tracks[:TD,:,[0,2]],[TD,-1])
    #     testX=np.reshape(measures[TD:TD+1000],[1000,-1])
    #     testY=np.reshape(tracks[TD:TD+1000,:,[0,2]],[1000,-1])
    #     # 每一帧的X和Y都要训练一个RF
    #     regr = RandomForestRegressor(n_estimators=trees,
    #                                  criterion="mse",
    #                                  n_jobs=3,
    #                                  random_state=0,
    #                                  verbose=0,)
    #     regr.fit(trainX,trainY)
    #     predY=regr.predict(testX)
    #     print(predY.shape)
    #     rmse=Compute_RMSE(predY,testY)
    #     print('trees: %d, rmse: %f'%(trees,rmse))

def repeatFig3():
    '''
    在Lolja论文中，图3的仿真
    '''
    # 对图3仿真所需要的参数
    isLoad=False # 是否加载原有文件
    radarPos=[4,0] # 雷达的位置，[4,0]
    sample_nums=41000 # 样本数量，50000
    TD=40000 # 训练样本数量
    iniState=np.array([0,1,0,1]) # 目标的初始状态，[0,1,0,1]
    deltaT=1 # 雷达扫描间隔， 1s
    processNoiseX=[0,0.01] # 过程噪声沿X轴的均值和方差，[0,0.1]
    processNoiseY=[0,0.01] # 过程噪声沿Y轴的均值和方差，[0,0.1]
    measNoiseD=[0,0.5] # 量测噪声中距离向的均值和方差，[0,0.5]
    measNoiseA=[0,0.3] # 量测噪声中角度向的均值和方差，[0,0.3]
    Frame=10 # 航迹一共有几帧，10
    particleNums=1000 # 粒子数量
    sample_file=path_head+'tracks_sample_%d_%.2f_%.2f_%.2f_%.2f.npz'%\
                          (sample_nums,processNoiseX[1],processNoiseY[1],measNoiseD[1],measNoiseA[1])

    # 产生航迹并绘图检验
    # tracks: [sample_nums,Frame,4]
    # measures: [sample_nums,Frame,2]
    tracks,measures=generate_tracks_Lolja(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                                           processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)
    # plot_tracks(tracks[:100],measures[:100],radarPos,pltTrack,pltMeas)

    trainX=np.reshape(measures[:TD],[TD,-1])
    trainY=np.reshape(tracks[:TD,:,[0,2]],[TD,-1])
    testX=measures[TD:TD+1000]
    testY=tracks[TD:TD+1000,:,[0,2]]
    testYv=tracks[TD:TD+1000,:,[1,3]]

    # 训练RF并预测
    # regr = RandomForestRegressor(n_estimators=150,
    #                              criterion="mse",
    #                              n_jobs=-1,
    #                              random_state=0,
    #                              verbose=0,)
    # regr.fit(trainX,trainY)
    # predY=regr.predict(np.reshape(testX,[1000,-1])).reshape([1000,Frame,2])
    # RF_RMSE=Compute_RMSE(predY,testY,axis=(0,2))
    RF_RMSE=np.zeros(10) # 随机森林的预测误差随时间变化曲线

    # 使用KalmanSmoother预测
    KF_RMSE=np.zeros(10) # 用不着，不仿真了

    # 使用ParticleFilter预测
    RPF=RegularizedParticleFilter(nums=particleNums, sigR=np.sqrt(measNoiseD[1]), sigA=np.sqrt(measNoiseA[1]), T=deltaT,
                                  radar=radarPos,processNoiseX=processNoiseX,processNoiseY=processNoiseY)
    predY=np.zeros_like(testX) # [1000,frame,2]
    for track in range(predY.shape[0]):
        if track%100==0: print('RPF tracks: %d'%(track))
        RPF.init_track(testX[track,0],testX[track,1],testYv[track,1])
        predY[track,0]=Polar2Coor(testX[track,0],radarPos)
        predY[track,1]=Polar2Coor(testX[track,1],radarPos)
        for frame in range(2,Frame):
            predY[track, frame]=RPF.filter(testX[track,frame])
        # 绘制航迹
        plt.plot(testY[track,0,0],testY[track,0,1],'o',label='start')
        plt.plot(predY[track,:,0],predY[track,:,1],'x-',label='pred')
        plt.plot(testY[track,:,0],testY[track,:,1],'x-',label='true')
        xxx=Polar2Coor(testX,radarPos)
        plt.plot(xxx[track,:,0],xxx[track,:,1],'x-',label='measure')
        plt.plot(radarPos[0],radarPos[1],'+',label='radar')
        plt.legend()
        plt.grid()
        plt.show()
    PF_RMSE=Compute_RMSE(predY,testY,axis=(0,2))
    # PF_RMSE=np.zeros(10)

    plt.plot(RF_RMSE,'x-',label='random forest')
    plt.plot(KF_RMSE,'o-',label='kalman smoother')
    plt.plot(PF_RMSE,'*-',label='particle filter')
    plt.xlabel("timestep k",fontsize=12)
    plt.ylabel("error",fontsize=12)
    plt.legend(prop={'size':14})
    plt.xlim([0,9])
    plt.ylim([0,2])
    plt.grid(linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    path_head='../20180405_data/Lolja/'
    repeatFig1()
    # repeatFig2()
    # repeatFig3()




































