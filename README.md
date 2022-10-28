# 比赛链接
  https://aistudio.baidu.com/aistudio/competition/detail/495/0/introduction

# 赛题背景
  汽车产业正在经历巨大变革，新能源汽车市场规模持续扩大，电池安全问题日益引发重视。 电池异常检测面临着汽车数据质量差，检出率低，误报率高，大量无效报警无法直接自动化运维等问题。  
  为了更好的检验电池安全问题，比赛通过募集优秀异常检测方案，使用特征提取、参数优化、异常对比等手段对实车数据进行处理，优化异常检测结果，以便更好的应用于车辆预警、故障模式识别等多种场景。
##  &nbsp;&nbsp;&nbsp;&nbsp;贴一个比赛结果，方案的选择及及其实验结果，可以查看[博客](https://blog.csdn.net/weixin_43509698/article/details/127417008)和[能源AI挑战赛_异常检测赛.pptx](https://github.com/Shybert-AI/Energy_Anomaly_Detection_TOP3/files/9889316/AI._.pptx)
![image](https://user-images.githubusercontent.com/82042336/198660038-d466bb59-74af-4d43-8f41-86edadc9021d.png)

![image](https://user-images.githubusercontent.com/82042336/198649860-826d7b38-0e00-4cfe-ad8f-ef6177f43c7c.png)

## 1.代码结构  
&nbsp;&nbsp;&nbsp;&nbsp;./Energy_Anomaly_Detection  
&nbsp;&nbsp;&nbsp;&nbsp;│  clf1.torch                                                            # 模型文件  
&nbsp;&nbsp;&nbsp;&nbsp;│  data_anls.ipynb                                                       # 数据分析脚本  
&nbsp;&nbsp;&nbsp;&nbsp;│  demo_resnet_epoch532_lr0.001_dropout0.25_auc0.8998.ipynb              # jupyter版本训练代码  
&nbsp;&nbsp;&nbsp;&nbsp;│  loss.png                                                              # loss曲线  
&nbsp;&nbsp;&nbsp;&nbsp;│  myutils.py                                                            # 配置文件  
&nbsp;&nbsp;&nbsp;&nbsp;│  README.md                                                             # 中文用户手册  
&nbsp;&nbsp;&nbsp;&nbsp;│  requirements.txt                                                      # 依赖环境  
&nbsp;&nbsp;&nbsp;&nbsp;│  Revisiting Deep Learning Models for Tabular Data.pdf   
&nbsp;&nbsp;&nbsp;&nbsp;│  submision.csv                                                         # 比赛要求的csv文件  
&nbsp;&nbsp;&nbsp;&nbsp;│  train.py                                                              # 在jupyter版本训练代码进行规整的训练脚本  
&nbsp;&nbsp;&nbsp;&nbsp;│  train_optuna.ipynb                                                    # optuna自动化调参  
&nbsp;&nbsp;&nbsp;&nbsp;│  能源AI挑战赛_异常检测赛.pptx                                      
&nbsp;&nbsp;&nbsp;&nbsp;├─log                                                                    # 日志类文件夹  
&nbsp;&nbsp;&nbsp;&nbsp;|      train.log                                                         # 训练日志文件                                                        
&nbsp;&nbsp;&nbsp;&nbsp;├─Test_A                                                                 # 验证数据（只存在样例数据，训练时需拷贝全部数据）  
&nbsp;&nbsp;&nbsp;&nbsp;│      0.pkl                                                             
&nbsp;&nbsp;&nbsp;&nbsp;│      1.pkl                                                                                                                        
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
&nbsp;&nbsp;&nbsp;&nbsp;│                                                                        
&nbsp;&nbsp;&nbsp;&nbsp;├─Train                                                                  # 训练数据（只存在样例数据，训练时需拷贝全部数据）  
&nbsp;&nbsp;&nbsp;&nbsp;│      0.pkl                                                             
&nbsp;&nbsp;&nbsp;&nbsp;│      1.pkl                                                                                                                        
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
&nbsp;&nbsp;&nbsp;&nbsp;│      ...                                                               
## 2.环境依赖
The experiment code is written in Python 3 and built on a number of Python packages:    
- delu==0.0.13  
- loguru==0.6.0  
- matplotlib==3.6.1  
- numpy==1.23.4  
- pandas==1.3.2  
- rtdl==0.0.13  
- scikit_learn==1.1.2  
- scipy==1.9.2  
- torch==1.11.0  
- tqdm==4.41.0  
- matplotlib==3.0.3  
- scipy==1.2.1  
- numpy==1.21.5  

## 3.两种形式执行训练，生成比赛要求的csv文件  
### &nbsp;&nbsp;&nbsp;&nbsp;3.1 建议采用train.py执行训练  
```python  
python train.py
python train.py  --model_name ResNet --train  Train  --test Test_A  --epoch 532  --batch_size 256 
```
### &nbsp;&nbsp;&nbsp;&nbsp;3.2 通过demo_resnet_epoch532_lr0.001_dropout0.25_auc0.8998.ipynb执行训练  



