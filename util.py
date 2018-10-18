#encoding=utf8
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
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import time,datetime
warnings.filterwarnings('ignore')

def data_pre_process(train,test):
    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = data.fillna(-1)

    ##from eda WE can find this bool var has only one value,so it can be reduced
    data.drop('creative_is_voicead',axis=1,inplace = True)
    data.drop('app_paid',axis=1,inplace=True)  
    data.drop('creative_is_js',axis=1,inplace=True)  

    ##process the date features
    data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
    data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
    data['day'][data['day']<27] = data['day'][data['day']<27] + 31
    ##base features
    data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[0])
    data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[1])
    data['area'] = data['creative_height'] * data['creative_width']
    data['user_tags'] = data['user_tags'].astype(str)
    data['user_tags_length'] = data['user_tags'].apply(lambda x:len(x.split(',')))

    bool_feature = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
    for i in bool_feature:
        data[i] = data[i].astype(int)
    
    ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_first','advert_industry_inner_second',
                       'advert_industry_inner', 'advert_name','campaign_id', 
                       'creative_id', 'creative_type', 'creative_tp_dnf']

    media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

    content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'os_name','make', 'model']

    origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
    
    cate_feature = origin_cate_list

    num_feature = ['creative_width', 'creative_height', 'hour','day', 'area','user_tags_length']

    feature = cate_feature + num_feature
    print(len(feature), feature)
    gc.collect()
    return data,cate_feature,num_feature,bool_feature

def countvector_user_tags(data,length_min,chi2_length):
    
    texts=data['user_tags'].astype(str)
    cv = CountVectorizer(min_df =length_min) ##min_df来选择最小出现次数
    cv.fit(texts)
    train_x = data[data.click != -1]
    train_y = train_x['click'].values
    predict_x = data[data.click == -1]

    train_csr = cv.transform(train_x['user_tags'].astype(str))
    predict_csr = cv.transform(predict_x['user_tags'].astype(str))
    feature_select = SelectKBest(chi2,k = chi2_length)
    feature_select.fit(train_csr,train_y)
    
    train_csr = feature_select.transform(train_csr)
    predict_csr = feature_select.transform(predict_csr)
    
    print(train_csr.shape,predict_csr.shape)
    gc.collect()
    return train_csr,predict_csr

##6个10w分点
def get_bool_features(data):
    cate_gory = ['adid', 'advert_id', 'orderid','advert_industry_inner_second',
               'advert_industry_inner', 'campaign_id', 'creative_id','advert_industry_inner_first',
                'advert_name', 'creative_tp_dnf']
    
               
    bool_feature = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
    data = data[cate_gory + bool_feature]
    for fea in cate_gory:
        for bool_fea in bool_feature:
            data = feat_mean(data,data,[fea],bool_fea)
            data = feat_var(data,data,[fea],bool_fea)
            data = feat_sum(data,data,[fea],bool_fea)
        print(fea+bool_fea,' finished!')
    data = data.drop(cate_gory + bool_feature,axis = 1)
    print(data.shape)
    gc.collect()
    return data

def get_time_quantum(data):
    data = data[['hour']]
    data['hour_isin_03'] = data['hour'].apply(lambda x: 1 if (x<=3) &(x>=0) else 0)
    data['hour_isin_46'] = data['hour'].apply(lambda x: 1 if (x<=6) &(x>=4) else 0)
    data['hour_isin_712'] = data['hour'].apply(lambda x: 1 if (x<=12) &(x>=7) else 0)
    data['hour_isin_1219'] = data['hour'].apply(lambda x: 1 if (x<=19) &(x>=12) else 0)
    data['hour_isin_2023'] = data['hour'].apply(lambda x: 1 if (x<=23) &(x>=20) else 0)
    data = data.drop(['hour'],axis=1)
    print('Time features get!')
    return data

def get_labelcounter(data):

    # labelencoder 转化
    encoder = ['city', 'province', 'make', 'model', 'osv', 'adid', 'advert_id', 'orderid','advert_industry_inner_second',
               'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id','advert_industry_inner_first',
               'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf','carrier','user_tags']
    col_encoder = LabelEncoder()
    for feat in encoder:
        col_encoder.fit(data[feat].astype(str))
        data[feat] = col_encoder.transform(data[feat].astype(str))
    gc.collect()
    print('labelencoder finished')
    return data[encoder]

def get_onehot_feature(data,cate_feature,num_feature):
    ##get train and test data

    for i in cate_feature:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

    train = data[data.click!=-1]
    test = data[data.click==-1]
       
    train_x = train
    train_y = train.click.values

    predict = test
    predict_x = predict.drop('click', axis=1)

    base_train_csr = sparse.csr_matrix((len(train_x), 0))

    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    count_dict={}
    for col in cate_feature:
        count_dict[col]=len(data[col].unique())
    count_df=pd.DataFrame()
    count_df['col']=count_dict.keys()
    count_df['count']=count_dict.values()
    feature_sort=count_df.sort_values('count')['col']
    #feature_sort=cate_feature

    enc = OneHotEncoder()
    for feature in feature_sort:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr','bool')

        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),'csr','bool')
        print('enc :',feature)
    print('one-hot prepared !')
    print('Before feature_select:',base_train_csr.shape,base_predict_csr.shape)
    
    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(base_train_csr, train_y)
    train_csr = feature_select.transform(base_train_csr)

    predict_csr = feature_select.transform(base_predict_csr)
    
    print('After feature_select:',train_csr.shape,predict_csr.shape)
    
    train_csr = sparse.hstack((sparse.csr_matrix(train_x[num_feature]), train_csr), 'csr').astype('float32')

    predict_csr = sparse.hstack((sparse.csr_matrix(predict_x[num_feature]), predict_csr), 'csr').astype('float32')
    
    print('Final shape:',train_csr.shape,predict_csr.shape)
    gc.collect()
    return train_csr,predict_csr


def get_xgb_features(train,train_csr,predict_csr,tree_number):
    
    train_y = train.click.values
    params={
    'eta': 0.3,
    'max_depth':7,   
    'min_child_weight':1,
    'gamma':0.3, 
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'nthread':12,
    'scale_pos_weight': 1,
    'lambda':1,  
    'seed':1000,
    'silent':0 ,
    'eval_metric': 'auc'
    }
    
    d_train = xgb.DMatrix(train_csr, label=train_y)
    d_test = xgb.DMatrix(predict_csr)
    watchlist = [(d_train, 'train')]

    #sklearn接口


    model_bst = xgb.train(params, d_train, tree_number, watchlist, early_stopping_rounds=100, verbose_eval=0)
    #model_sklearn=clf.fit(X_train, y_train)

    y_bst= model_bst.predict(d_test)
    #y_sklearn= clf.predict_proba(X_test)[:,1]
    train_new_feature= model_bst.predict(d_train, pred_leaf=True)
    test_new_feature= model_bst.predict(d_test, pred_leaf=True)
    print("新的特征集(自带接口)：",train_new_feature.shape)
    print("新的测试集(自带接口)：",test_new_feature.shape)
    return train_new_feature,test_new_feature


'''
get adid features:曝光率加级数特征
'''
def adid_unique_count_features(data):
    adid_nuq=['model','make','os','city','province','user_tags','f_channel','app_id','carrier','nnt', 'devtype',
             'app_cate_id','inner_slot_id']
    data = data[adid_nuq + ['adid']]
    #广告的曝光率 提升5个w
    for fea in adid_nuq:
        data = feat_nunique(data,data,['adid'],fea)
        data = feat_count(data,data,['adid'],fea)
        data = feat_nunique(data,data,[fea],'adid')
        data = feat_count(data,data,[fea],'adid')
        #gp1 = data.groupby('adid')[fea].nunique().reset_index().rename(columns={fea:"adid_%s_nuq_num"%fea})
        #gp2 = data.groupby(fea)['adid'].nunique().reset_index().rename(columns={'adid':"%s_adid_nuq_num"%fea})
        #data=pd.merge(data,gp1,how='left',on=['adid'])
        #data=pd.merge(data,gp2,how='left',on=[fea])   
        gc.collect()
    data = data.drop(adid_nuq + ['adid'],axis = 1)
    print('uni_count get!',data.shape)
    return data

'''
get size features
'''
def get_size_features(data):
    count_fea=['city','adid','advert_id','app_id','advert_industry_inner','make','model','os','f_channel']
    data = data[count_fea]
    for fea in count_fea:
        gp=data.groupby(fea).agg('size').reset_index().rename(columns={0:"%s_count_num"%fea})
        data=pd.merge(data,gp,how='left',on=fea)
        gc.collect()
    data = data.drop(count_fea,axis = 1)
    print('size get!',data.shape)
    return data


###baseline coding
def baseline_data(train,test):
    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = data.fillna(-1)
    data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
    data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
    data['label'] = data.click.astype(int)
    del data['click']
    data.drop('creative_is_voicead',axis=1,inplace = True)
    data.drop('app_paid',axis=1,inplace=True)  
    data.drop('creative_is_js',axis=1,inplace=True)  
    bool_feature = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
    for i in bool_feature:
        data[i] = data[i].astype(int)

    data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])

    ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                       'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                       'creative_is_jump', 'creative_is_download']

    media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

    content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

    origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

    for i in origin_cate_list:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

    cate_feature = origin_cate_list

    num_feature = ['creative_width', 'creative_height', 'hour']

    feature = cate_feature + num_feature
    print(len(feature), feature)

    predict = data[data.label == -1]
    predict_result = predict[['instance_id']]
    predict_result['predicted_score'] = 0
    predict_x = predict.drop('label', axis=1)

    train_x = data[data.label != -1]
    train_y = data[data.label != -1].label.values


    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')
    '''
    train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
        'float32')
    predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
    '''
    print(base_train_csr.shape)
    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(base_train_csr, train_y)
    train_csr = feature_select.transform(base_train_csr)
    predict_csr = feature_select.transform(base_predict_csr)
    print('feature select')
    data_csr = sparse.vstack((train_csr, predict_csr))
    print(data_csr.shape)
    return data_csr


#15%划分验证集，单折
def lgb_test(data):
    train_data = data[data.click!=-1]
    test_data = data[data.click==-1]    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test_data[['instance_id']]
    
    train_x = train_data.drop(['instance_id','click'],axis = 1)
    train_y = train_data['click'].values
    final_predict = test_data.drop(['instance_id','click'],axis = 1)
    train_csr,valid_csr,train_label,valid_label = train_test_split(train_x, train_y, test_size=0.15, random_state=2018)
    print('训练集和验证集划分完成：',train_csr.shape,valid_csr.shape)
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    loss = 0

    
    lgb_model = model.fit(train_csr, train_label,
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(train_csr, train_label),(valid_csr, valid_label)],
                          early_stopping_rounds=100,verbose = 10)

    loss = lgb_model.best_score_['valid']['binary_logloss']
    test_pred = lgb_model.predict_proba(final_predict, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['predicted_score'] = test_pred
    print('logloss:',loss)

    res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res


#15%划分验证集，单折
def lgb_csr_test(train_csr,label,predict_csr,test):
    res = test[['instance_id']]
    
    train_csr,valid_csr,train_label,valid_label = train_test_split(train_csr, label, test_size=0.15, random_state=2018)
    print('训练集和验证集划分完成：',train_csr.shape,valid_csr.shape)
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    loss = 0

    
    lgb_model = model.fit(train_csr, train_label,
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(train_csr, train_label),(valid_csr, valid_label)],
                          early_stopping_rounds=100,verbose = 10)

    loss = lgb_model.best_score_['valid']['binary_logloss']
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['predicted_score'] = test_pred
    print('logloss:',loss)

    res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res




def lgb_cross_validation(train_csr,label,predict_csr,test,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test[['instance_id']]

    print('训练集和测试集划分完成：',train_csr.shape,predict_csr.shape)	
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    # model = lgb.LGBMClassifier(objective='binary',
    #                                 max_depth=8,
    #                                 num_leaves=64,
    #                                 learning_rate=0.05,
    #                                 n_estimators=5000,
    #                                 colsample_bytree=0.3,#0.3
    #                                 subsample = 0.75,#0.75
    #                                 n_jobs=4,
    #                                 lambda_l2=10,
    #                                 seed=2018,
    #                                 )

    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    baseloss = []
    loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_csr, label)):
        print("Fold", i)
        lgb_model = model.fit((train_csr[train_index]), label[train_index],
                              eval_names =['train','valid'],
                              eval_metric='logloss',
                              eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                              ,early_stopping_rounds=100,verbose = 0)

        baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
        loss += lgb_model.best_score_['valid']['binary_logloss']
        test_pred= lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
        print(baseloss)
        print('test mean:', test_pred.mean())
        res['prob_%s' % str(i)] = test_pred
    print('logloss:', baseloss, loss/n_folds)

    # 加权平均
    res['predicted_score'] = 0
    for i in range(n_folds):
        res['predicted_score'] += res['prob_%s' % str(i)]
    res['predicted_score'] = res['predicted_score']/n_folds
    mean = res['predicted_score'].mean()
    print('mean:',mean)
    res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res


def lgb_many_seeds_cross_validation(train_csr,label,predict_csr,test,seeds,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test[['instance_id']]

    print('训练集和测试集划分完成：',train_csr.shape,predict_csr.shape) 
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    # model = lgb.LGBMClassifier(objective='binary',
    #                                 max_depth=8,
    #                                 num_leaves=64,
    #                                 learning_rate=0.05,
    #                                 n_estimators=5000,
    #                                 colsample_bytree=0.3,#0.3
    #                                 subsample = 0.75,#0.75
    #                                 n_jobs=4,
    #                                 lambda_l2=10,
    #                                 seed=2018,
    #                                 )
    res['predicted_score'] = 0
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        baseloss = []
        loss = 0
        for i, (train_index, test_index) in enumerate(skf.split(train_csr, label)):
            print("Fold", i)
            lgb_model = model.fit((train_csr[train_index]), label[train_index],
                                  eval_names =['train','valid'],
                                  eval_metric='logloss',
                                  eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                                  ,early_stopping_rounds=100,verbose = 0)

            baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
            loss += lgb_model.best_score_['valid']['binary_logloss']
            test_pred= lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
            print(baseloss)
            print('test mean:', test_pred.mean())
            res['prob_%s' % str(i)] = test_pred
        print('logloss:', baseloss, loss/n_folds)

        # 加权平均
        res['predicted_score_%s' %str(seed)] = 0
        for i in range(n_folds):
            res['predicted_score_%s'%str(seed)] += res['prob_%s' % str(i)]
        res['predicted_score_%s'%str(seed)] = res['predicted_score_%s'%str(seed)]/n_folds
        mean = res['predicted_score_%s'%str(seed)].mean()
        res['predicted_score'] += res['predicted_score_%s'%str(seed)]
        res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline_%s.csv"%str(seed), index=False)
        print('mean:',mean)

    res['predicted_score'] = res['predicted_score']/len(seeds)
    print('All_result_mean:',res['predicted_score'].mean())
    res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res


def xgb_cross_validation(train_csr,label,predict_csr,test,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test[['instance_id']]

    print('训练集和测试集划分完成：',train_csr.shape,predict_csr.shape) 
    model = xgb.XGBClassifier(boosting_type='gbdt', num_leaves=48, max_depth=8, learning_rate=0.05, n_estimators=3000,
                                subsample=0.8,
                                colsample_bytree=0.6, reg_alpha=3, reg_lambda=5, seed=1000, nthread=10,verbose=50)

    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    baseloss = []
    loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_csr, label)):
        print("Fold", i)
        xgb_model = model.fit((train_csr[train_index]), label[train_index],
                              eval_metric='logloss',
                              eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                              ,early_stopping_rounds=100,verbose = 100)
        valid_pred = xgb_model.predict_proba(train_csr[test_index], ntree_limit=xgb_model.best_iteration)[:, 1]
        valid_loss = log_loss(label[test_index],valid_pred)
        baseloss.append(valid_loss)
        loss += valid_loss
        test_pred= xgb_model.predict_proba(predict_csr, ntree_limit=xgb_model.best_iteration)[:, 1]
        print(baseloss)
        print('test mean:', test_pred.mean())
        res['prob_%s' % str(i)] = test_pred
    print('logloss:', baseloss, loss/n_folds)

    # 加权平均
    res['predicted_score'] = 0
    for i in range(n_folds):
        res['predicted_score'] += res['prob_%s' % str(i)]
    res['predicted_score'] = res['predicted_score']/n_folds
    mean = res['predicted_score'].mean()
    print('mean:',mean)
    res[['instance_id', 'predicted_score']].to_csv("./submit/xgb_baseline.csv", index=False)
    return res


def lgb_stacking(train_csr,label,predict_csr,test,n_folds = 5):
    
    # X_loc_test = lgb.Dataset(base_predict_csr)
    res = test[['instance_id']]

    print('训练集和测试集划分完成：',train_csr.shape,predict_csr.shape) 
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=3000,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    # model = lgb.LGBMClassifier(objective='binary',
    #                                 max_depth=8,
    #                                 num_leaves=64,
    #                                 learning_rate=0.05,
    #                                 n_estimators=5000,
    #                                 colsample_bytree=0.3,#0.3
    #                                 subsample = 0.75,#0.75
    #                                 n_jobs=4,
    #                                 lambda_l2=10,
    #                                 seed=2018,
    #                                 )

    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    baseloss = []
    loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_csr, label)):
        print("Fold", i)
        lgb_model = model.fit((train_csr[train_index]), label[train_index],
                              eval_names =['train','valid'],
                              eval_metric='logloss',
                              eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                              ,early_stopping_rounds=100,verbose = 0)

        val_pred= lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
        baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
        loss += lgb_model.best_score_['valid']['binary_logloss']
        test_pred= lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
        print(baseloss)
        print('test mean:', test_pred.mean())
        res['prob_%s' % str(i)] = test_pred
    print('logloss:', baseloss, loss/n_folds)

    # 加权平均
    res['predicted_score'] = 0
    for i in range(n_folds):
        res['predicted_score'] += res['prob_%s' % str(i)]
    res['predicted_score'] = res['predicted_score']/n_folds
    mean = res['predicted_score'].mean()
    print('mean:',mean)
    res[['instance_id', 'predicted_score']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res


def stacking(train_csr,label,predict_csr,test,n_folds = 5):
    test_stack = test[['instance_id']]
        # # 模型部分
    model_1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2,
                                max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True,verbose=50)

    model_2 = xgb.XGBClassifier(boosting_type='gbdt', num_leaves=48, max_depth=8, learning_rate=0.05, n_estimators=2,
                                subsample=0.8,
                                colsample_bytree=0.6, reg_alpha=3, reg_lambda=5, seed=1000, nthread=10,verbose=50)

    model_3 = cb.CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, loss_function='Logloss',
                                   logging_level='Verbose')

    skf=list(StratifiedKFold(label, n_folds=5, shuffle=True, random_state=1024))

    valid_stack = pd.DataFrame(columns=['valid_index','pred_1','pred_2','pred_3','pred_4','pred_5'])

    for i, (train_index, test_index) in enumerate(skf):
        print("Fold", i)
        valid_pred_fold = pd.DataFrame()

        model_1 = model_1.fit((train_csr[train_index]), label[train_index],
                               eval_names =['train','valid'],
                               eval_metric='logloss',
                               eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                               ,early_stopping_rounds=100)
        model_2 = model_2.fit((train_csr[train_index]), label[train_index],
                               eval_metric='logloss',
                               eval_set=[((train_csr[train_index]), label[train_index]),((train_csr[test_index]), label[test_index])]
                               ,early_stopping_rounds=100,verbose = 100)

        model_3 = model_3.fit((train_csr[train_index].todense()), label[train_index],
                               eval_set=[((train_csr[train_index].todense()), label[train_index]),
                                         ((train_csr[test_index].todense()), label[test_index])]
                               ,early_stopping_rounds=100,verbose = 100)
        
        valid_pred_1 = model_1.predict_proba(train_csr[test_index], num_iteration=model_1.best_iteration_)[:, 1]
        valid_pred_2 = model_2.predict_proba(train_csr[test_index],ntree_limit=model_2.best_iteration)[:,1]
        valid_pred_3 = model_3.predict_proba(train_csr[test_index].todense())[:,1]

        valid_pred_fold['valid_index'] = list(test_index)
        valid_pred_fold['pred_1'] = list(valid_pred_1)
        valid_pred_fold['pred_2'] = list(valid_pred_2)
        valid_pred_fold['pred_3'] = list(valid_pred_3)
        valid_pred_fold['label'] = list(label[test_index])
        
        valid_stack = valid_stack.append(valid_pred_fold)
        
        test_pred_1 = model_1.predict_proba(predict_csr, num_iteration=model_1.best_iteration_)[:, 1]
        test_pred_2 = model_2.predict_proba(predict_csr,ntree_limit=model_2.best_iteration)[:,1]
        test_pred_3 = model_3.predict_proba(predict_csr_cat.todense())[:,1]

        test_stack['prob1_%s' % str(i)] = test_pred_1
        test_stack['prob2_%s' % str(i)] = test_pred_2
        test_stack['prob3_%s' % str(i)] = test_pred_3
    #     test_stack['prob4_%s' % str(i)] = test_pred_4
    #     test_stack['prob5_%s' % str(i)] = test_pred_5

    test_stack['pred_1'] = 0
    test_stack['pred_2'] = 0
    test_stack['pred_3'] = 0

    for i in range(5):
        test_stack['pred_1'] += test_stack['prob1_%s' % str(i)]
        test_stack['pred_2'] += test_stack['prob2_%s' % str(i)]
        test_stack['pred_3'] += test_stack['prob3_%s' % str(i)]
#     test_stack['pred_4'] += test_stack['prob4_%s' % str(i)]
#     test_stack['pred_5'] += test_stack['prob5_%s' % str(i)]
    test_stack['pred_1'] = test_stack['pred_1']/5
    test_stack['pred_2'] = test_stack['pred_2']/5
    test_stack['pred_3'] = test_stack['pred_3']/5

    test_stack = test_stack[['pred_1','pred_2','pred_3']]
    test_stack.to_csv("data/test_stack_cat.csv", index=False)
    valid_stack.to_csv("data/valid_stack_cat.csv", index=False)


def stacking_get_result(valid_stack,test_stack,test):
    res = test[['instance_id']]
    train_csr = valid_stack[['pred_1','pred_2','pred_3']]
    label = valid_stack[['label']]
    predict_csr = test_stack[['pred_1','pred_2','pred_3']]
    lgb_cross_validation(train_csr,label,predict_csr,test)



