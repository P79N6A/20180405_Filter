# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 19:56
# @Author  : LiYun
# @File    : LeeYun_method.py
'''Description :

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
from KalmanFilter import CVKalmanFilter
from ParticleFilter import RegularizedParticleFilter
from Generate_sample import Banana_meas_Lolja,generate_tracks_Lolja,generate_tracks_LeeYun,\
    plot_tracks,Polar2Coor,Coor2Polar,windowSample,centralizeMeas

def showFig1():
    '''
    第一组图，展示非线性的问题
        1.展示直角坐标系下的香蕉噪声
        2.展示直角坐标系下的高斯噪声
    '''
    radarPos=np.array([0,0])
    measureC=np.array([10,10])
    # 香蕉配置
    # rangeGaussian=[0,0.005]
    # angleGaussian=[0,0.02]
    # 高斯配置
    rangeGaussian=[0,0.05]
    angleGaussian=[0,0.005]
    Banana_meas_Lolja(radarPos,measureC,rangeGaussian,angleGaussian)

def showFig2():
    '''
    第二组图展示Xgboost目标跟踪的参量变化情况
        1.训练样本数量
        2.树的数量
        3.滑动窗口的长度
    场景配置：
        1.雷达位于坐标原点[0,0]
        2.起始目标状态随机均匀分布在[±10,±2,±10,±2]范围内
    '''
    # 对图2仿真所需要的参数
    isLoad=True
    pltTrack=True
    pltMeas=True
    radarPos=[0,0]
    sample_nums=101000
    iniState=(np.random.rand(sample_nums,4)-0.5)*np.array([20,4,20,4]) # [sample_nums,4]
    deltaT=1
    processNoiseX=[0,0.005]
    processNoiseY=[0,0.005]
    # 香蕉配置
    measNoiseD=[0,0.005]
    measNoiseA=[0,0.02]
    # 高斯配置
    # measNoiseD=[0,0.05]
    # measNoiseA=[0,0.005]
    Frame=16
    sample_file=path_head+'tracks_sample_%d_%.2f_%.2f_%.2f_%.2f.npz'%\
                          (sample_nums,processNoiseX[1],processNoiseY[1],measNoiseD[1],measNoiseA[1])

    # 产生航迹并绘图检验
    # tracks: [sample_nums,Frame,4]
    # measures: [sample_nums,Frame,2]
    tracks,measures=generate_tracks_LeeYun(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                                           processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)
    # plot_tracks(tracks,measures,radarPos,pltTrack,pltMeas)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'seed':0,
    }

    # 图1，训练XGB，观察在不同样本数量下面的性能
    # num_boost_rounds=200
    # window=10 # 滑窗的长度
    # # 先将量测转换到直角坐标系，再根据滑窗的长度提取样本
    # # tracks: [sample_nums*Frame,4]
    # # measures: [sample_nums*Frame,window,2]
    # measures = Polar2Coor(measures,radarPos)
    # tracks = np.reshape(tracks,[sample_nums*Frame,4])
    # measures = windowSample(measures,sample_nums,Frame,window)
    # # 中心化量测数据，并制作输入特征
    # # tracks: [sample_nums*Frame,4]
    # # measures: [sample_nums*Frame,window*2]
    # measures = centralizeMeas(measures)
    # measures=measures.reshape([sample_nums*Frame,window*2])
    # for TD in [1000,3000,10000,20000,50000,100000]: # 不同的训练样本
    #     predY=np.empty([1000*Frame,2])
    #     # X轴和Y轴各训练一个XGB
    #     trainX=measures[:TD*Frame]
    #     testX=measures[TD*Frame:(TD+1000)*Frame]
    #     for axis in range(2):
    #         print('XGB samples %d. Training %c axis'%(TD,'Y' if axis else 'X'))
    #         trainY=tracks[:TD*Frame,axis*2]
    #         testY=tracks[TD*Frame:(TD+1000)*Frame,axis*2]
    #         dtrain = xgb.DMatrix(trainX, label=trainY)
    #         dval=xgb.DMatrix(testX, label=testY)
    #         watchlist = [(dtrain, 'train'),(dval, 'eval')]
    #         regr = xgb.train(dict(xgb_params), dtrain, num_boost_rounds)
    #         predY[:,axis]=regr.predict(dval)
    #     rmse=Compute_RMSE(predY,tracks[TD*Frame:(TD+1000)*Frame,[0,2]])
    #     print('sample: %d, rmse: %f'%(TD,rmse))
    #     break

    # 图2，训练XGB，观察在不同树的数量下面的性能
    # TD=10000
    # window=10 # 滑窗的长度
    # # 先将量测转换到直角坐标系，再根据滑窗的长度提取样本
    # # tracks: [sample_nums*Frame,4]
    # # measures: [sample_nums*Frame,window,2]
    # measures = Polar2Coor(measures,radarPos)
    # tracks = np.reshape(tracks,[sample_nums*Frame,4])
    # measures = windowSample(measures,sample_nums,Frame,window)
    # # 中心化量测数据，并制作输入特征
    # # tracks: [sample_nums*Frame,4]
    # # measures: [sample_nums*Frame,window*2]
    # measures = centralizeMeas(measures)
    # measures=measures.reshape([sample_nums*Frame,window*2])
    # for num_boost_rounds in [50,100,150,200,300,500]: # 不同的树的数量
    #     predY=np.empty([1000*Frame,2])
    #     # X轴和Y轴各训练一个XGB
    #     trainX=measures[:TD*Frame]
    #     testX=measures[TD*Frame:(TD+1000)*Frame]
    #     for axis in range(2):
    #         print('XGB trees %d. Training %c axis'%(num_boost_rounds,'Y' if axis else 'X'))
    #         trainY=tracks[:TD*Frame,axis*2]
    #         testY=tracks[TD*Frame:(TD+1000)*Frame,axis*2]
    #         dtrain = xgb.DMatrix(trainX, label=trainY)
    #         dval=xgb.DMatrix(testX, label=testY)
    #         watchlist = [(dtrain, 'train'),(dval, 'eval')]
    #         regr = xgb.train(dict(xgb_params), dtrain, num_boost_rounds)
    #         predY[:,axis]=regr.predict(dval)
    #     rmse=Compute_RMSE(predY,tracks[TD*Frame:(TD+1000)*Frame,[0,2]])
    #     print('trees: %d, rmse: %f'%(num_boost_rounds,rmse))
    #     break

    # 图3，训练XGB，观察在不同滑动窗口的长度下面的性能
    TD=10000
    num_boost_rounds=200
    for window in [4,7,10,13,16]: # 滑窗的长度
        # 先将量测转换到直角坐标系，再根据滑窗的长度提取样本
        # tracks2: [sample_nums*Frame,4]
        # measures2: [sample_nums*Frame,window,2]
        measures2 = Polar2Coor(measures,radarPos)
        tracks2 = np.reshape(tracks,[sample_nums*Frame,4])
        measures2 = windowSample(measures2,sample_nums,Frame,window)
        # 中心化量测数据，并制作输入特征
        # tracks2: [sample_nums*Frame,4]
        # measures2: [sample_nums*Frame,window*2]
        measures2 = centralizeMeas(measures2)
        measures2=measures2.reshape([sample_nums*Frame,window*2])
        predY=np.empty([1000*Frame,2])
        # X轴和Y轴各训练一个XGB
        trainX=measures2[:TD*Frame]
        testX=measures2[TD*Frame:(TD+1000)*Frame]
        for axis in range(2):
            print('XGB window %d. Training %c axis'%(window,'Y' if axis else 'X'))
            trainY=tracks2[:TD*Frame,axis*2]
            testY=tracks2[TD*Frame:(TD+1000)*Frame,axis*2]
            dtrain = xgb.DMatrix(trainX, label=trainY)
            dval=xgb.DMatrix(testX, label=testY)
            watchlist = [(dtrain, 'train'),(dval, 'eval')]
            regr = xgb.train(dict(xgb_params), dtrain, num_boost_rounds)
            predY[:,axis]=regr.predict(dval)
        rmse=Compute_RMSE(predY,tracks2[TD*Frame:(TD+1000)*Frame,[0,2]])
        print('window: %d, rmse: %f'%(window,rmse))
        # break

def showFig3():
    '''
    第三组图展示在一定场景下的跟踪的比较，比较的方法有随机森林，XGBoost，卡尔曼，粒子滤波
        1.航迹展示
        2.香蕉配置
            rangeGaussian=[0,0.005]
            angleGaussian=[0,0.02]
        2.高斯配置
            rangeGaussian=[0,0.05]
            angleGaussian=[0,0.005]
    '''
    # 对图3仿真所需要的参数
    isLoad=True # 是否加载原有文件
    pltTrack=True
    pltMeas=False
    showTracks=True
    showXGB=True
    showRF=True
    showPF=True
    showKF=True
    GenerateTracksForFigure4=False # 讨论不同扇区不同量测噪声特性，估计的结果
    radarPos=[4,0] # 雷达的位置，[4,0]
    sample_nums=101000 # 样本数量，50000
    # 起始目标状态随机均匀分布在[±10,±2,±10,±2]范围内
    iniState=(np.random.rand(sample_nums,4)-0.5)*np.array([20,4,20,4]) # 目标的初始状态 [sample_nums,4]
    deltaT=1 # 雷达扫描间隔， 1s
    TD=1000 # 训练样本数量
    processNoiseX=[0,0.005] # 过程噪声沿X轴的均值和方差，[0,0.1]
    processNoiseY=[0,0.005] # 过程噪声沿Y轴的均值和方差，[0,0.1]
    # 香蕉配置
    measNoiseD=[0,0.005] # 量测噪声中距离向的均值和方差，[0,0.5]
    measNoiseA=[0,0.02] # 量测噪声中角度向的均值和方差，[0,0.3]
    # 高斯配置
    # measNoiseD=[0,0.05] # 量测噪声中距离向的均值和方差，[0,0.5]
    # measNoiseA=[0,0.005] # 量测噪声中角度向的均值和方差，[0,0.3]
    Frame=16 # 航迹一共有几帧，10
    sample_file=path_head+'tracks_sample_%d_%.2f_%.2f_%.2f_%.2f.npz'%\
                          (sample_nums,processNoiseX[1],processNoiseY[1],measNoiseD[1],measNoiseA[1])

    # 产生航迹并绘图检验
    # tracks: [sample_nums,Frame,4]
    # measures: [sample_nums,Frame,2]
    tracks,measures=generate_tracks_LeeYun(sample_file,sample_nums,isLoad,radarPos,iniState,deltaT,
                                           processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)
    # tracks,measures=generate_tracks_Lolja(sample_file,sample_nums,isLoad,radarPos,[0,1,0,1],deltaT,
    #                                        processNoiseX,processNoiseY,measNoiseD,measNoiseA,Frame)

    # 图1，航迹展示
    if showTracks: plot_tracks(tracks[:10],measures[:10],radarPos,pltTrack,pltMeas)

    # 图2或图3
    # XGB
    if showXGB:
        print('training XGBoost')
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': 1,
            'seed':0,
        }
        num_boost_rounds=200
        window=10
        # 先将量测转换到直角坐标系，再根据滑窗的长度提取样本
        # tracks2: [sample_nums*Frame,4]
        # measures2: [sample_nums*Frame,window,2]
        measures2 = Polar2Coor(measures,radarPos)
        tracks2 = np.reshape(tracks,[sample_nums*Frame,4])
        measures2 = windowSample(measures2,sample_nums,Frame,window)
        # 中心化量测数据，并制作输入特征
        # tracks2: [sample_nums*Frame,4]
        # measures2: [sample_nums*Frame,window*2]
        measures2 = centralizeMeas(measures2)
        measures2=measures2.reshape([sample_nums*Frame,window*2])
        predY=np.empty([1000*Frame,2])
        # X轴和Y轴各训练一个XGB
        trainX=measures2[:TD*Frame]
        testX=measures2[TD*Frame:(TD+1000)*Frame]
        regr=[]
        for axis in range(2):
            trainY=tracks2[:TD*Frame,axis*2]
            testY=tracks2[TD*Frame:(TD+1000)*Frame,axis*2]
            dtrain = xgb.DMatrix(trainX, label=trainY)
            dval=xgb.DMatrix(testX, label=testY)
            regr.append(xgb.train(dict(xgb_params), dtrain, num_boost_rounds))
            predY[:,axis]=regr[axis].predict(dval)
        XGB_RMSE=Compute_RMSE(predY.reshape([1000,Frame,2]),tracks2[TD*Frame:(TD+1000)*Frame,[0,2]].reshape([1000,Frame,2]),axis=(0,2))

        if GenerateTracksForFigure4:
            # 产生Frame帧的航迹的量测值，平移到监控区域的4个位置，再转换到极坐标，用XGB预测，减去最后一帧平移的影响，评价XGB对不同区域量测噪声的考虑
            # tracks2[Frame,2],noise[Frame,2],startpoint[4,2]
            tracks2=np.concatenate((np.arange(Frame).reshape([Frame,1]),np.zeros([Frame,1])),axis=1)
            startpoint=np.array([[8,2],[-2,8],[-8,-2],[2,-8]])
            noise=np.random.randn(Frame,2)*0.5
            # tracks[4, Frame, 4]
            # measures2[4, Frame, 2]
            tracks2=np.tile(tracks2,(4,1,1))+np.tile(startpoint,(Frame,1,1)).transpose([1,0,2])
            measures2=Coor2Polar(tracks2+np.tile(noise,(4,1,1)),radarPos)
            # tracks2=np.concatenate((tracks2[:,:,0,np.newaxis],tracks2[:,:,1,np.newaxis],tracks2[:,:,1,np.newaxis],tracks2[:,:,1,np.newaxis]),axis=2)
            # plot_tracks(tracks2, measures2, radarPos, True, True)
            # 制作输入样本
            testX = Polar2Coor(measures2, radarPos)
            testX = windowSample(testX, 4, Frame, window)
            # 中心化量测数据，并制作输入特征
            testX = centralizeMeas(testX)
            testX = testX.reshape([4 * Frame, window * 2])
            predY = np.empty([4 * Frame, 2])
            for axis in range(2):
                dval = xgb.DMatrix(testX)
                predY[:, axis] = regr[axis].predict(dval)
            predY=predY.reshape([4,Frame,2])
            # 画图
            tracks = Polar2Coor(measures2, radarPos)
            for t in range(4):
                plt.plot(tracks[t, :, 0], tracks[t, :, 1], 'o-', markersize=4, mfc='none')
                plt.plot(predY[t, :, 0], predY[t, :, 1], 'x-.', markersize=4, mfc='none')
            plt.xlabel("x in meters", fontsize=12)
            plt.ylabel("y in meters", fontsize=12)
            # plt.xlim([-2,12])
            # plt.ylim([-2,12])
            plt.axis('equal')
            plt.grid(linestyle='dotted')
            plt.plot(radarPos[0], radarPos[1], '*', label='radar', markersize=10)
            plt.legend()
            plt.show()

    # 训练RF并预测
    if showRF:
        print('training RF')
        trainX=np.reshape(measures[:TD],[TD,-1])
        trainY=np.reshape(tracks[:TD,:,[0,2]],[TD,-1])
        testX=measures[TD:TD+1000]
        testY=tracks[TD:TD+1000,:,[0,2]]
        regr = RandomForestRegressor(n_estimators=150,
                                     criterion="mse",
                                     n_jobs=-1,
                                     random_state=0,
                                     verbose=0,)
        regr.fit(trainX,trainY)
        predY=regr.predict(np.reshape(testX,[1000,-1])).reshape([1000,Frame,2])
        RF_RMSE=Compute_RMSE(predY,testY,axis=(0,2))

    # 使用ParticleFilter预测
    if showPF:
        print('training PF')
        testX=measures[TD:TD+1000]
        testY=tracks[TD:TD+1000,:,[0,2]]
        particleNums=1000 # 粒子数量
        RPF=RegularizedParticleFilter(nums=particleNums, sigR=np.sqrt(measNoiseD[1]), sigA=np.sqrt(measNoiseA[1]), T=deltaT,
                                      radar=radarPos,processNoiseX=processNoiseX,processNoiseY=processNoiseY)
        predY=np.zeros_like(testX) # [1000,frame,2]
        for track in range(predY.shape[0]):
            if track%100==0: print('RPF tracks: %d'%(track))
            RPF.init_track(testX[track,0],testX[track,1])
            predY[track,0]=Polar2Coor(testX[track,0],radarPos)
            predY[track,1]=Polar2Coor(testX[track,1],radarPos)
            for frame in range(2,Frame):
                predY[track, frame]=RPF.filter(testX[track,frame-1],testX[track,frame])
            # 绘制航迹
            # plt.plot(testY[track,0,0],testY[track,0,1],'o',label='start')
            # plt.plot(predY[track,:,0],predY[track,:,1],'x-',label='pred')
            # plt.plot(testY[track,:,0],testY[track,:,1],'x-',label='true')
            # xxx=Polar2Coor(testX,radarPos)
            # plt.plot(xxx[track,:,0],xxx[track,:,1],'x-',label='measure')
            # plt.plot(radarPos[0],radarPos[1],'+',label='radar')
            # plt.legend()
            # plt.grid()
            # plt.show()
        PF_RMSE=Compute_RMSE(predY,testY,axis=(0,2))

    # 使用KalmanFilter预测
    if showKF:
        testX=measures[TD:TD+1000] # [sample,Frame,2] 极坐标
        testY=tracks[TD:TD+1000,:,[0,2]] # [sample,Frame,2] 直角坐标
        # kalman滤波先将量测值转换到直角坐标，再将量测噪声在极坐标下的协方差矩阵转换为直角坐标系下的协方差矩阵
        testX=Polar2Coor(testX,radarPos)
        # 利用4状态CV模型卡尔曼滤波跟踪目标
        predY=np.empty([1000,Frame,2]) #估计值
        KF=CVKalmanFilter(measNoiseD=measNoiseD, measNoiseA=measNoiseA, deltaT=deltaT,
                          radar=radarPos,processNoiseX=processNoiseX,processNoiseY=processNoiseY)
        for track in range(predY.shape[0]):
            if track%100==0: print('KF tracks: %d'%(track))
            KF.init_track(testX[track,0],testX[track,1])
            predY[track,0]=testX[track,0]
            predY[track,1]=testX[track,1]
            for frame in range(2,Frame):
                predY[track, frame]=KF.filter(testX[track,frame])
        KF_RMSE = Compute_RMSE(predY, testY, axis=(0, 2))

    if showXGB: plt.plot(XGB_RMSE,'+-',label='XGBoost')
    if showRF: plt.plot(RF_RMSE,'x--',label='random forest')
    if showKF: plt.plot(KF_RMSE,'o-.',label='kalman filter')
    if showPF: plt.plot(PF_RMSE,'*:',label='particle filter')
    if showXGB or showRF or showKF or showPF:
        plt.xlabel("timestep k",fontsize=12)
        plt.ylabel("error in meters",fontsize=12)
        plt.legend(prop={'size':14})
        plt.xlim([0,Frame-1])
        # plt.ylim([0,2])
        plt.grid(linestyle='dotted')
        plt.legend(loc='upper left')
        plt.show()

if __name__=='__main__':
    path_head='../20180405_data/LeeYun/'
    # showFig1()
    # showFig2()
    showFig3()










































