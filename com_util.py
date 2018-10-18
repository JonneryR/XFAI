#encoding=utf8
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile


def get_gongxian_feature(id_A,id_B,all_data):
    def get_gongxian_list(data):
        data = data.split('_')
        data2 = data[1].split(';')
        data2.remove(data[0])
        return ','.join(data2)

    gongxian = all_data[[id_A,id_B]]
    gongxian = feat_nunique(gongxian,gongxian,[id_A],id_B)
    gongxian_duplicate = gongxian.drop_duplicates()
    gongxian_join = gongxian_duplicate.groupby([id_A])[id_B].agg(lambda x:';'.join(x.astype(str))).reset_index().rename(columns={id_B: id_A+'_'+id_B})
    gongxian = gongxian.merge(gongxian_join,on=[id_A],how='left')
    gongxian[id_B+'-'+id_A+'_'+id_B] = gongxian[id_B].astype(str) + '_' + gongxian[id_A+'_'+id_B]
    gongxian['gongxian-'+id_A+'_'+id_B] = gongxian[id_B+'-'+id_A+'_'+id_B].apply(get_gongxian_list)
    gongxian = gongxian[[id_A,'gongxian-'+id_A+'_'+id_B]]
    base_gongxian_csr = pd.DataFrame() 
    cv = CountVectorizer(min_df=1000)
    y_csr = all_data['click'].values
    for feature in ['gongxian-'+id_A+'_'+id_B]:
        gongxian[feature] = gongxian[feature].astype(str)
        cv.fit(gongxian[feature])
        base_gongxian_csr = sparse.hstack((base_gongxian_csr, cv.transform(gongxian[feature].astype(str))), 'csr', 'float32')

    print('cv prepared !',base_gongxian_csr.shape)
    if base_gongxian_csr.shape[1]>200:
        base_gongxian_csr = SelectKBest(chi2,k = 100).fit_transform(base_gongxian_csr,y_csr)

    print('cv selected !',base_gongxian_csr.shape)
    return base_gongxian_csr

# ad_data
def get_ad_feature(ad_data):
    
    ad_data['creative_area'] = ad_data.creative_height*ad_data.creative_width
    ad_data['ad_industry_one'] = ad_data['advert_industry_inner'].apply(lambda x:(x.split('_'))[0])
    ad_data['ad_industry_two'] = ad_data['advert_industry_inner'].apply(lambda x:(x.split('_'))[1])
    replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink']
    for feat in replace:
        ad_data[feat] = ad_data[feat].replace([False, True], [0, 1])
    for feat in ['adid','advert_id','orderid','advert_industry_inner','advert_name','campaign_id',
                 'creative_id','creative_tp_dnf','ad_industry_one','ad_industry_two']:
        ad_data = encode_count(ad_data,feat)
    
    
    ad_features = ['adid','advert_id','orderid','advert_industry_inner','advert_name','campaign_id',
                 'creative_id','creative_tp_dnf','ad_industry_one','ad_industry_two','creative_type',
                  'creative_is_jump', 'creative_is_download', 'creative_has_deeplink',
                  ]+['creative_height','creative_area','creative_width']
    
    cate_features = ['adid','advert_id','orderid','advert_industry_inner','advert_name','campaign_id',
                 'creative_id','creative_tp_dnf','ad_industry_one','ad_industry_two','creative_type',
                  'creative_is_jump', 'creative_is_download', 'creative_has_deeplink',
                  ]
    
    return ad_data,ad_features,cate_features



# app_data
def get_app_feature(app_data):
    app_data = app_data.fillna(0)
    app_data['inner_slot'] = app_data['inner_slot_id'].apply(lambda x:(x.split('_'))[0])
    
    for feat in ['app_cate_id','app_id','f_channel','app_paid','inner_slot_id','inner_slot']:
        app_data = encode_count(app_data,feat)

    app_features = ['app_cate_id','app_id','f_channel','inner_slot_id','inner_slot','time_hour']
    cate_features = ['app_cate_id','app_id','f_channel','inner_slot_id','inner_slot','time_hour']
    return app_data,app_features,cate_features

# context_data
def get_context_feature(context_data):
    
    for feat in ['city', 'province', 'make', 'model', 'osv', 'os_name']:
        context_data = encode_count(context_data,feat)
        
    context_features = ['city','province','make','carrier','nnt','devtype', 'osv', 'os_name','model','os']
    cate_features = ['city','province','make','carrier','nnt','devtype', 'osv', 'os_name','model','os']
    return context_data,context_features,cate_features

# zhuan huan lv
def get_rate_feature(data):
    data['period'] = data['time_day']
    data['period'][data['period']<27] = data['period'][data['period']<27] + 31
    rate_features = []
    for feat_1 in  ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
#    ['adid','advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
#               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
        rate_features.append(feat_1+'_rate')
        res=pd.DataFrame()
        temp=data[[feat_1,'period','click']]
        for period in range(27,35):
            if period == 27:
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
            else: 
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']==period-1).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']==period-1).values].sum()).reset_index(name=feat_1+'_1')
            count[feat_1+'_1']=count1[feat_1+'_1'].astype(int)
            count.fillna(value=0, inplace=True)
            count[feat_1+'_rate'] = count[feat_1+'_1'] / count[feat_1+'_all']
            count['period']=period
            count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        print(feat_1,' over')
        data = pd.merge(data,res, how='left', on=[feat_1,'period'])
        data = data[data.period!=27]
    return data,rate_features

# zhuan huan lv
def get_rate_yu_feature(data):
    data['period'] = data['time_day']
    data['period'][data['period']<27] = data['period'][data['period']<27] + 31
    rate_features = []
    for feat_1 in  ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
                 # add more ]:
#    ['adid','advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
#               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
        rate_features.append(feat_1+'_rate')
        res=pd.DataFrame()
        temp=data[[feat_1,'period','click']]
        for period in range(27,35):
            if period == 27:
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
            else: 
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
            count[feat_1+'_1']=count1[feat_1+'_1'].astype(int)
            count.fillna(value=0, inplace=True)
            count[feat_1+'_rate'] = count[feat_1+'_1'] / count[feat_1+'_all']
            count['period']=period
            count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        print(feat_1,' over')
        data = pd.merge(data,res, how='left', on=[feat_1,'period'])
#         data = data[data.period!=27]
    return data,rate_features

def get_rate_more_feature(data):
    data['period'] = data['time_day']
    data['period'][data['period']<27] = data['period'][data['period']<27] + 31
    rate_features = []
    for feat_1 in  ['app_id','inner_slot_id','model']:
#    ['adid','advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
#               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
        rate_features.append(feat_1+'_rate')
        res=pd.DataFrame()
        temp=data[[feat_1,'period','click']]
        for period in range(27,35):
            if period == 27:
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
            else: 
                count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
                count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
            count[feat_1+'_1']=count1[feat_1+'_1'].astype(int)
            count.fillna(value=0, inplace=True)
            count[feat_1+'_rate'] = count[feat_1+'_1'] / count[feat_1+'_all']
            count['period']=period
            count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
            count.fillna(value=0, inplace=True)
            res=res.append(count,ignore_index=True)
        print(feat_1,' over')
        data = pd.merge(data,res, how='left', on=[feat_1,'period'])
#         data = data[data.period!=27]
    return data,rate_features

## nuique feature
def get_adid_feature(all_data):
    fe_nuique = ['advert_id','orderid','advert_industry_inner','advert_name','campaign_id','creative_id',
                 'app_cate_id','app_id','inner_slot_id',
                 'city','province','nnt','make','osv']
    n_features = []
    for fe in fe_nuique:
        all_data = feat_nunique(all_data,all_data,[fe],'adid')
        n_features.append('adid'+"_%s_nunique" % ("_".join([fe])))
    for fe in fe_nuique:
        all_data = feat_count(all_data,all_data,[fe],'adid')
        n_features.append('adid'+"_%s_count" % ("_".join([fe])))    
    return all_data,n_features
def get_cross_feature(all_data,context_f,app_f):
    cross_f = []
#    context_f = ['province','carrier','os','make','nnt']
#    app_f = ['app_id','inner_slot_id']
#    ad_f =  ['adid','advert_industry_inner','advert_id','campaign_id','creative_id','creative_type','creative_area']
    all_data = all_data[context_f+app_f]
    for c_f in context_f:
        for ap_f in app_f:
            f_name = c_f+'_'+ap_f
            cross_f.append(f_name)
            all_data[f_name] = all_data[c_f].astype(str)+'_'+all_data[ap_f].astype(str)
            all_data = encode_count(all_data,f_name)
            print(ap_f+c_f,' finished')
    all_data = all_data.drop(context_f+app_f,axis = 1)
    return all_data

#overfit
def encode_count_rank(all_data,column_name):
    rate_str = [column_name]
    all_data_train = all_data[all_data.click!=-1]
    rate_feat = all_data_train[rate_str+['click']]
    rate_feat = merge_count(rate_feat,rate_str,'click','r_count')
    rate_feat = merge_sum(rate_feat,rate_str,'click','r_sum')
    rate_feat = rate_feat[rate_str+['r_count','r_sum']].drop_duplicates()
    rate_feat['rate'] = rate_feat['r_sum']/(rate_feat['r_count']+1000)
    train_type = rate_feat[rate_str[0]].tolist()[:]
    rate_feat = rate_feat[rate_feat.r_count>0].sort_values([rate_str[0],'r_count','rate'][2],ascending=False).reset_index(drop=True)
    rate_feat[rate_str[0]+'_idrank']=rate_feat.index
    rate_feat = rate_feat[[rate_str[0],rate_str[0]+'_idrank']]
    all_data = all_data.merge(rate_feat,on=rate_str,how='left')
    all_data.fillna(-1)
    all_data.drop(rate_str[0],axis=1,inplace=True)
    all_data.rename(columns={rate_str[0]+'_idrank':rate_str[0]},inplace=True)
    return all_data


def deal_make(data):
    if 'vivo' in data:
        return 'vivo'
    elif 'iphone' in data:
        return 'APPLE'
    elif 'iPhone' in data:
        return 'APPLE'
    elif 'OPPO' in data:
        return 'OPPO'
    elif 'xiaomi' in data:
        return 'Xiaomi'
    elif 'Xiaomi' in data:
        return 'Xiaomi'
    elif 'HUAWEI' in data:
        return 'HUAWEI'
    elif 'Huawei' in data:
        return 'HUAWEI'
    elif 'Apple' in data:
        return 'APPLE'
    elif 'Lenovo' in data:
        return 'Lenovo'
    elif 'iPad' in data:
        return 'APPLE'
    elif 'Meizu' in data:
        return 'Meizu'
    elif 'SAMSUNG' in data:
        return 'SAMSUNG'
    elif 'samsung' in data:
        return 'SAMSUNG'
    elif 'Philips' in data:
        return 'Philips'
    elif 'Hisense' in data:
        return 'Hisense'
    else:
        return data
    
def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all,feature_df.columns.tolist()

def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df
    
#sample=feat_count(sample,sample,["user_id"],"a_date")

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_count" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_nunique" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_mean" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_std" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_median" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_max" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_min" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_sum" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_var" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

