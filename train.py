# -*- coding:utf-8 -*-
import os 
import pickle
import random
import argparse
import platform
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import delu
import rtdl
import scipy
from loguru import logger
from myutils import Utils
utils = Utils() # utils function


def  load_data(pkl_list,label=True):
    '''
    输入pkl的列表，进行文件加载
    label=True用来加载训练集
    label=False用来加载真正的测试集，真正的测试集无标签
    '''
    X = []
    y = []
    

    for  each_pkl in pkl_list:
        pic = open(each_pkl,'rb')
        item= pickle.load(pic)#下载pkl文件
        # 此处选取的是每个滑窗的最后一条数据，仅供参考，可以选择其他的方法，比如均值或者其他处理时序数据的网络
        # 此处选取了前7个特征，可以需求选取特征数量
        feature = item[0][:,0:7][-1]
        #feature = item[0][:,0:8][-1]
        #feature = item[0][:,0:7].mean(axis=0)
        #feature = np.append(item[0][:,0:7][-1],(item[0][:,3][-1] - item[0][:,4][-1])) #加max_single_volt - min_single_volt 一列为特征
        feature=np.append(feature,item[1]["mileage"])
        X.append(feature)
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.vstack(X)
    if label:
        y = np.vstack(y)
    return X, y
    
def normalization(data): 
    """
    归一化数据
    """
    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    data = (data - _mean) / (_std + 1e-4)
    return data


class FTTransformer():
    '''
    The original code: https://yura52.github.io/rtdl/stable/index.html
    The original paper: "Revisiting Deep Learning Models for Tabular Data", NIPS 2019
    '''
    def __init__(self, seed:int, model_name:str, n_epochs=1000, batch_size=256):

        self.seed = seed
        self.model_name = model_name
        self.utils = Utils()

        # device
        if model_name == 'FTTransformer':
            self.device = self.utils.get_device(gpu_specific=True)
        else:
            self.device = self.utils.get_device(gpu_specific=False)

        # Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
        # zero.improve_reproducibility(seed=self.seed)
        delu.improve_reproducibility(base_seed=int(self.seed))

        # hyper-parameter
        self.n_epochs = n_epochs # default is 1000
        self.batch_size = batch_size # default is 256

    def apply_model(self, x_num, x_cat=None):
        if isinstance(self.model, rtdl.FTTransformer):
            return self.model(x_num, x_cat)
        elif isinstance(self.model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return self.model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(self, X, y=None):
        self.model.eval()
        score = []
        # for batch in delu.iter_batches(X[part], 1024):
        for batch in delu.iter_batches(X, self.batch_size):
            score.append(self.apply_model(batch))
        score = torch.cat(score).squeeze(1).cpu().numpy()
        score = scipy.special.expit(score)

        # calculate the metric
        if y is not None:
            target = y.cpu().numpy()
            metric = self.utils.metric(y_true=target, y_score=score)
        else:
            metric = {'aucroc': None, 'aucpr': None}

        return score, metric['aucpr']

    def fit(self, X_train, y_train, ratio=None,X_test=None,y_test=None):
        # set seed
        self.utils.set_seed(self.seed)

        # training set is used as the validation set in the anomaly detection task
        X = {'train': torch.from_numpy(X_train).float().to(self.device),
             'val': torch.from_numpy(X_train).float().to(self.device)}

        y = {'train': torch.from_numpy(y_train).float().to(self.device),
             'val': torch.from_numpy(y_train).float().to(self.device)}
        

        task_type = 'binclass'
        n_classes = None
        d_out = n_classes or 1


        if self.model_name == 'ResNet':
            self.model = rtdl.ResNet.make_baseline(
                d_in=X_train.shape[1],
                d_main=128,
                d_hidden=256,
                dropout_first=0.25,
                dropout_second=0.0,
                n_blocks=2,
                d_out=d_out,
            )
            lr = 0.001
            weight_decay = 0.0
        
        elif self.model_name == 'MLP':
            self.model = rtdl.MLP.make_baseline(
            d_in=X_train.shape[1],
            d_layers= [128, 256, 128],
            dropout=0.25,
            d_out=d_out,
            )
            lr = 0.001
            weight_decay = 0.0

        elif self.model_name == 'FTTransformer':
            self.model = rtdl.FTTransformer.make_default(
                n_num_features=X_train.shape[1],
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=d_out,
            )
            
        elif self.model_name == 'FTTransformer_baseline':
            self.model = rtdl.FTTransformer.make_baseline(
                n_num_features=X_train.shape[1],
                cat_cardinalities=None,
                d_token=X_train.shape[1],
                n_blocks=2,
                attention_dropout=0.2,
                ffn_d_hidden=6,
                ffn_dropout=0.2,
                residual_dropout=0.0,
                d_out=d_out,
            ) 
        else:
            raise NotImplementedError

        self.model.to(self.device)
        optimizer = (
            self.model.make_default_optimizer()
            if isinstance(self.model, rtdl.FTTransformer)
            else torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        )
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )

        # Create a dataloader for batches of indices
        # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
        train_loader = delu.data.IndexLoader(len(X['train']), self.batch_size, device=self.device)

        # Create a progress tracker for early stopping
        # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=100)

        # training
        # report_frequency = len(X['train']) // self.batch_size // 5

        for epoch in range(1, self.n_epochs + 1):
            loss_tmp = []
            for iteration, batch_idx in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()
                x_batch = X['train'][batch_idx]
                y_batch = y['train'][batch_idx]
                loss = loss_fn(self.apply_model(x_batch).squeeze(1), y_batch)
                loss_tmp.append(loss.item())
                loss.backward()
                optimizer.step()
                # if iteration % report_frequency == 0:
                #     print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')
                logger.info(f"Epoch {epoch:03d}/{self.n_epochs} | batch_size:{self.batch_size} | iteration:{iteration} | batch_loss:{loss.item():.4f}")

            loss_.append(sum(loss_tmp)/len(loss_tmp))
            _, val_metric = self.evaluate(X=X['val'], y=y['val'])
            print(f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}', end='')
            logger.info(f'Epoch {epoch:03d}/{self.n_epochs} | Validation metric: {val_metric:.4f} | Train loss:{sum(loss_tmp)/len(loss_tmp):.4f}')

            progress.update((-1 if task_type == 'regression' else 1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            # 验证
            # output predicted anomaly score on testing set
            score = self.predict_score(X_test)
            # evaluation
            result = utils.metric(y_true=y_test, y_score=score)
            aucroc.append(result['aucroc'])
            aucpr.append(result['aucpr'])
            if progress.fail:
                break

        return self

    def predict_score(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        score, _ = self.evaluate(X=X, y=None)
        return score

if __name__ == "__main__":
    # 模型参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='Train')
    parser.add_argument('--test', default='Test_A')
    parser.add_argument('--epoch',default=532)
    parser.add_argument('--batch_size',default=256)
    parser.add_argument('--model_name',default="ResNet", choices=["ResNet", "MLP", "FTTransformer"])
    args = parser.parse_args()
    data_path3 = args.test
    epoch = int(args.epoch)
    batch_size = int(args.batch_size)
    model_name = args.model_name
    #log文件
    log = "log/train.log"
    if not os.path.isdir(os.path.dirname(log)):
        os.makedirs(os.path.dirname(log))
    else:
        # 删除历史log
        if os.path.isfile(log):
            os.remove(log)
    logger.add(log,
                   rotation="500MB",
                   encoding="utf-8",
                   enqueue=True,
                   retention="10 days")
    
    # 加载训练集的pkl文件，划分训练集与验证集
    ind_pkl_files = []#存放标签为0的文件
    ood_pkl_files = []#存放标签为1的文件
    data_path=args.train#存放数据的路径
    pkl_files = glob(data_path+'/*.pkl')
    for each_path in tqdm(pkl_files):
        pic = open(each_path,'rb')
        this_pkl_file= pickle.load(pic)#下载pkl文件
        if this_pkl_file[1]['label'] == '00':
            ind_pkl_files.append(each_path)
        else:
            ood_pkl_files.append(each_path)

    random.seed(0)
    # 排序并打乱存放车辆序号的集合
    random.shuffle(ind_pkl_files)
    random.shuffle(ood_pkl_files)
    # 3/4的正样本和全部的负样本作为训练集，1/4的正样本和1/4的负样本作为训练集
    train_pkl_files = [ ind_pkl_files[j] for j in range(len(ind_pkl_files)//4,len(ind_pkl_files))] + [ ood_pkl_files[i] for i in range(len(ood_pkl_files))]
    test_pkl_files=[ind_pkl_files[i] for i in range(len(ind_pkl_files)//4)] + [ood_pkl_files[i] for i in range(len(ood_pkl_files)//4)]

    print(len(train_pkl_files))
    print(len(test_pkl_files))
    
    # 加载并归一化训练数据和验证数据
    X_train,y_train=load_data(train_pkl_files)
    # 进行随机打乱，这里random_state指定为固定值，则打乱结果相同
    X_train,y_train = shuffle(X_train,y_train,random_state=40)
    X_test,y_test=load_data(test_pkl_files)
    X_train = normalization(X_train)
    X_test = normalization(X_test)

    test1_files = glob(data_path3+'/*.pkl')
    X_val,_=load_data(test1_files,label=False)
    X_val = normalization(X_val)
    
    # 初始化模型。并进行训练
    seed = 42
    clf=FTTransformer(seed,model_name,n_epochs=epoch,batch_size=batch_size)
    aucroc = []
    aucpr = []
    loss_ = []
    clf = clf.fit(X_train=X_train, y_train=y_train.squeeze(1),X_test=X_test,y_test=y_test)
    import platform
    y_val_scores = clf.predict_score(X_val)   #返回未知数据上的异常值 (分值越大越异常) # outlier scores
    #记录文件名和对应的异常得分
    predict_result={}
    for i in tqdm(range(len(test1_files))):
        file=test1_files[i]
        #如果是window系统：
        if platform.system().lower() == 'windows':
            name=file.split('\\')[-1]
        #如果是linux系统
        elif platform.system().lower() == 'linux':
            name=file.split('/')[-1]
        predict_result[name]=y_val_scores[i]
    predict_score=pd.DataFrame(list(predict_result.items()),columns=['file_name','score'])#列名必须为这俩个
    predict_score.to_csv(f'submision.csv',index = False) #保存为比赛要求的csv文件
    # 保存模型
    torch.save(clf,"clf1.torch")
    # 绘制loss曲线
    plt.subplot(121)
    plt.plot(aucroc,label="aucroc")
    plt.plot(aucpr,label="aucpr")
    plt.legend()
    plt.subplot(122)
    plt.plot(loss_,label="loss")
    plt.legend()
    plt.savefig("loss.png")
