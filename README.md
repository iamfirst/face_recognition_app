一个基于PyQt5和InsightFace的简单人脸识别应用。该应用能够实现人脸的注册、识别以及人脸数据库的管理。未来，进一步扩展该应用的功能，例如增加实时摄像头人脸识别、多人脸检测等功能。
## 1. 项目概述

本项目的主要功能包括：
- **人脸注册**：通过上传图片或摄像头捕获图像，提取人脸特征并保存到本地数据库中。
- **人脸识别**：通过上传图片或摄像头捕获图像，与数据库中的人脸特征进行比对，识别出对应的人脸。
- **人脸数据库管理**：显示已注册的人脸图片列表，方便用户查看和管理。

## 2. 技术栈

- **PyQt5**：用于构建图形用户界面（GUI）。
- **InsightFace**：用于人脸检测和特征提取。
- **OpenCV**：用于图像处理和摄像头捕获。
- **NumPy**：用于处理人脸特征向量的存储和计算。
- **scikit-learn**：用于计算余弦相似度，判断人脸是否匹配。

## 3. 运行效果
先上传图片并输入人脸名称，点击注册完成人脸库录入；然后再重新上传一张相关或不相关的图片，点击比对人脸，程序将人脸数据与已有人脸库数据对比，识别完成后返回比对结果及相似度
![人脸注册](static/register.png)
![人脸识别](static/compare.png)

## 4. 代码解析
https://blog.csdn.net/weixin_54862871/article/details/144999853
