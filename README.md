# 事件相机去遮挡

计算摄像学课程大作业，复现并改进了"Image De-Occlusion via Event-Enhanced Multi-Modal Fusion Hybrid Network"的工作。

原工作的github项目见[此处](https://github.com/lisiqi19971013/Event_Enhanced_DeOcc)

## 系统环境

- 本项目测试于Python3.9
- 使用到的库详见[requirements.txt]()文件，其中使用了SNN的encoder用于对事件模态信息进行编码，所以需要配置SNN的优化器
- 相机使用DAVIS346 Mono，同时输出帧和事件信息（但帧信息是灰度的，不是RGB的）

## 训练数据

使用原工作的数据集进行训练：
https://pan.baidu.com/s/1o2nTtNFgeA-Feri1OaGETw, 提取码 ddqf

由于我们使用的相机仅提供灰度信息，我们将原数据集先处理成灰度，再进行训练，以保证模型有效性。

## 模型结构

我们设计了新的backbone来处理相机输入的数据。

backbone灵感来自于[metaformer](https://arxiv.org/pdf/2111.11418v3.pdf)，我们参考了metaformer的结构，并设计了新的ffn block。

## 实验环境

- 相机使用DAVIS346 Mono，同时输出帧和事件信息，但帧信息是灰度的，不是RGB的
- 导轨选择FSL40滚珠丝杆滑台模组，搭配步进电机，使用KH-01控制
- 遮挡物我们测试了栅栏（密集），栅栏（稀疏），栅栏（缠绕葡萄叶）
- 注意场景中不要使用频闪灯光！

