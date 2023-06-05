# 事件相机去遮挡

计算摄像学课程大作业，复现并改进了"Image De-Occlusion via Event-Enhanced Multi-Modal Fusion Hybrid Network"的工作。

原工作的github项目见[此处](https://github.com/lisiqi19971013/Event_Enhanced_DeOcc)

## 系统环境

- 本项目测试于Python3.9
- 使用到的库详见[requirements.txt](https://github.com/careless-lu/CT-image-annotation/blob/main/requirements.txt)文件，其中由于使用了SNN的encoder用于对事件模态信息进行编码，所以需要配置SNN的优化器
- 相机使用DAVIS346 Mono，同时输出帧和事件信息（但帧信息是灰度的，不是RGB的）

## 训练数据

使用原工作的数据集进行训练：
https://pan.baidu.com/s/1o2nTtNFgeA-Feri1OaGETw, 提取码 ddqf

数据集结构如下：

<img src="http://rvs4ha6m4.hn-bkt.clouddn.com/%E6%95%B0%E6%8D%AE%E9%9B%86%E7%BB%93%E6%9E%84.png" alt="数据集结构" style="zoom: 25%;" />

由于我们使用的相机仅提供灰度信息，我们将原数据集先处理成灰度，再进行训练，以保证模型有效性。

## 模型结构

我们设计了新的backbone来处理相机输入的数据。

![模型结构](http://rvs4ha6m4.hn-bkt.clouddn.com/%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png)

backbone灵感来自于[metaformer](https://arxiv.org/pdf/2111.11418v3.pdf)，我们参考了metaformer的结构，并设计了新的ffn block。

## 实验环境

- 相机使用DAVIS346 Mono，同时输出帧和事件信息，但帧信息是灰度的，不是RGB的
- 导轨选择FSL40滚珠丝杆滑台模组，搭配步进电机，使用KH-01控制
- 遮挡物我们测试了栅栏（密集），栅栏（稀疏），栅栏（缠绕葡萄叶）
- 注意场景中不要使用频闪灯光！

<img src="http://rvs4ha6m4.hn-bkt.clouddn.com/%E5%AE%9E%E9%AA%8C%E7%8E%AF%E5%A2%83.png" alt="实验环境" style="zoom: 33%;" />

## 实验结果

![模型结果](http://rvs4ha6m4.hn-bkt.clouddn.com/%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%9C.png)

![测试](http://rvs4ha6m4.hn-bkt.clouddn.com/%E6%B5%8B%E8%AF%95.png)
