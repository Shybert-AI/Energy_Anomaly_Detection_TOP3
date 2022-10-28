## 1.代码结构  
./Energy_Anomaly_Detection
│  clf1.torch                                                            # 模型文件  
│  data_anls.ipynb                                                       # 数据分析脚本  
│  demo_resnet_epoch532_lr0.001_dropout0.25_auc0.8998.ipynb              # jupyter版本训练代码  
│  loss.png                                                              # loss曲线  
│  myutils.py                                                            # 配置文件  
│  README.md                                                             # 中文用户手册  
│  requirements.txt                                                      # 依赖环境  
│  Revisiting Deep Learning Models for Tabular Data.pdf   
│  submision.csv                                                         # 比赛要求的csv文件  
│  train.py                                                              # 在jupyter版本训练代码进行规整的训练脚本  
│  train_optuna.ipynb                                                    # optuna自动化调参  
│  能源AI挑战赛_异常检测赛.pptx                                      
├─log                                                                    # 日志类文件夹  
|      train.log                                                         # 训练日志文件                                                        
├─Test_A                                                                 # 验证数据（只存在样例数据，训练时需拷贝全部数据）  
│      0.pkl                                                             
│      1.pkl                                                                                                                        
│      ...                                                               
│      ...                                                               
│      ...                                                               
│                                                                        
├─Train                                                                  # 训练数据（只存在样例数据，训练时需拷贝全部数据）  
│      0.pkl                                                             
│      1.pkl                                                                                                                        
│      ...                                                               
│      ...                                                               
│      ...                                                               
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
### 3.1 建议采用train.py执行训练  
```python  
python train.py
python train.py  --model_name ResNet --train  Train  --test Test_A  --epoch 532  --batch_size 256 
```
### 3.2 通过demo_resnet_epoch532_lr0.001_dropout0.25_auc0.8998.ipynb执行训练  



