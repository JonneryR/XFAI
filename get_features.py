#encoding=utf8
#author:JonneryR
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import math,gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from com_util import *
from util import *
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import time,datetime
warnings.filterwarnings('ignore')

train1 = pd.read_csv("./data/round1_iflyad_train.txt",delimiter="\t")
train2 = pd.read_csv("./data/round2_iflyad_train.txt",delimiter="\t")
train = pd.concat([train1,train2],axis = 0)
test = pd.read_csv("./data/round2_iflyad_test_feature.txt",delimiter="\t")

print('Data read successfully!', train.shape,test.shape)

data,cate_feature,num_feature,bool_feature = data_pre_process(train,test)
data.to_csv("./features/original_data.csv", index=False)
print(data.shape)

adid_unicou_data = adid_unique_count_features(data)
size_data = get_size_features(data)

adid_unicou_data.to_csv("./features/adid_unicou_data.csv", index=False)
size_data.to_csv("./features/size_data.csv", index=False)
print(adid_unicou_data.shape,size_data.shape)

label_encoder_data = get_labelcounter(data)
label_encoder_data.to_csv("./features/label_encoder_data.csv", index=False)
print(label_encoder_data.shape)


bool_features = get_bool_features(data)
bool_features.to_csv("./features/bool_features.csv", index=False)
print(bool_features.shape)

time_quantum = get_time_quantum(data)
time_quantum.to_csv("./features/time_quantum.csv", index=False)
print(time_quantum.shape)

data_csr = baseline_data(train,test)
sparse.save_npz('./features/data_csr.npz', data_csr)
print(data_csr.shape)

cross_A_1 = ['advert_industry_inner','campaign_id','creative_id','advert_id']
cross_A_2 = ['app_id','inner_slot_id','city','make','model','hour']
cross_feature = get_cross_feature(data,cross_A_1,cross_A_2)
cross_feature.to_csv("./features/cross_features.csv", index=False)