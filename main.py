#encoding=utf8
#author:JonneryR
###这个文件主要用来做单模cross_validation加特征的。
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

bool_feature = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first','advert_industry_inner_second',
                   'advert_industry_inner', 'advert_name','campaign_id', 
                   'creative_id', 'creative_type', 'creative_tp_dnf']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'os_name','make', 'model']
origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
cate_feature = origin_cate_list

num_feature = ['creative_width', 'creative_height','day','hour','area','user_tags_length']

###data 包括all features and area,hour,day
data = pd.read_csv('./features/original_data.csv')
print(data.shape)

adid_unicou_data = pd.read_csv('./features/adid_unicou_data.csv')
print(adid_unicou_data.shape)

size_data = pd.read_csv('./features/size_data.csv')
print(size_data.shape)

label_encoder_data = pd.read_csv('./features/label_encoder_data.csv')
print(label_encoder_data.shape)

#bool_features = pd.read_csv('./features/bool_features.csv')
#print(bool_features.shape)

time_quantum = pd.read_csv('./features/time_quantum.csv')
print(time_quantum.shape)

###不太好用
#cross_features = pd.read_csv('./features/cross_features.csv')
#print(cross_features.shape)

train_label = data[data.click!=-1].click.values
train = data[data.click!=-1]
test = data[data.click==-1]
num_bool_feature = num_feature + bool_feature##+ ['click','instance_id']
data_new = data[num_bool_feature]
train_data = pd.concat([data_new,label_encoder_data,adid_unicou_data,size_data,time_quantum],axis = 1)
print(train_data.shape)
#result = lgb_test(train_data)
#print(result.shape)

#train_data_new = train_data.drop()
data_csr = sparse.load_npz('./features/data_csr.npz').tocsr().astype('bool')
final_csr = sparse.hstack((data_csr,sparse.csr_matrix(train_data)),'csr').astype('float32')
print(final_csr.shape)
train_csr = final_csr[:train.shape[0]]
predict_csr = final_csr[train.shape[0]:]

gc.collect()
#result = lgb_csr_test(train_csr,train_label,predict_csr,test)

result = xgb_cross_validation(train_csr,train_label,predict_csr,test)
#seeds = [1000,2018,3036,4096,5000,6666,7096,8888]
#result = lgb_many_seeds_cross_validation(train_csr,train_label,predict_csr,test,seeds,10)
print(result.shape)