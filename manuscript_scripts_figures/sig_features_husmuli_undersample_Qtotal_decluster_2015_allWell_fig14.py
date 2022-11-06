import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2
from whakaari import datetimeify,save_dataframe, load_dataframe
from tsfresh.transformers import FeatureSelector
from textwrap import wrap
from imblearn.under_sampling import RandomUnderSampler
import os,sys
# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def alt(a, end, window, start=0, step=1):
    # stack overflow copy-paste 
    # https://stackoverflow.com/questions/54237254/count-values-in-overlapping-sliding-windows-in-python
    # bin_starts = pd.to_datetime(np.arange(start, end+step-window, step))
    # bin_ends = bin_starts + window
    bin_ends = pd.to_datetime(np.arange(start, end+step, step))
    bin_starts = bin_ends - window
    last_index = np.searchsorted(a, bin_ends, side='right')
    first_index = np.searchsorted(a, bin_starts, side='left')
    return  last_index - first_index
def cal_lambda(Dti,ti_FM,times):
    
    Dti_df=timedelta(days=1*Dti)###1 week Dti
    EQs=list()
    # rate_ti2=list()
    for i in ti_FM:
        EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
    
    rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
    return rate_ti
def main():
    # percentile_list=[30,40,50,60,70,80]
    percentile_list=50
    Dti_list=20##1 week Dti
    Nbins=30
    well_keyword=['HN09','HN12','HN14','HN16','CO2','HN17']
    # well_keyword=['CO2','CO2','CO2','CO2','CO2','CO2']
    # well_keyword='Hus_Q_total'
    # FM_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_Husmuli_Q_total_norm.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN09_norm.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN12_norm.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN14_norm.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN17_norm.csv']
    FM_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_HN09_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN12_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN14_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN17_norm2015.csv']
    # FM_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv']
    # FM_data_path=r'Z:\MLproject\whakaari-master\features\test_features_Husmuli_Q_total_norm2015.csv'
    # feature_keyword_list=['norm_Q_total(t/h)','norm_HN09_flow(t/h)','norm_HN12_flow(t/h)',
    #     'norm_HN14_flow(t/h)','norm_HN16_flow(t/h)','norm_HN17_flow(t/h)']
    feature_keyword_list=['norm_HN09_flow(t/h)','norm_HN12_flow(t/h)',
        'norm_HN14_flow(t/h)','norm_HN16_flow(t/h)','norm_CO2(t/h)','norm_HN17_flow(t/h)']
    # feature_keyword_list=['norm_CO2(t/h)','norm_CO2(t/h)','norm_CO2(t/h)',
    #     'norm_CO2(t/h)','norm_CO2(t/h)','norm_CO2(t/h)']
    # savefile_name_list=['normQ','diff_normQ','norm_IntQ']
    features_num_plot=1
    start_id_list=[2,2,1,3,6,2]
    # start_id_list=[59,60,61,62,63,64]
    # start_id_list=[12,13,14,15,16,17]
    # number_oder=['(a)','(b)','(c)','(d)','(e)','(f)']
    number_oder=['(a)','(b)','(c)','(d)','(e)','(f)']
    f,axq = plt.subplots(3,2,figsize=(8,12))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.965)
    axq=axq.ravel()
    for i in range(len(FM_data_path)):
        fts_Q_10,pvs_Q_10,FM_sig_Q_LB_1,FM_sig_Q_LB_0=One_well_data(FM_data_path[i],Dti_list,percentile_list,well_keyword[i],feature_keyword_list[i],features_num_plot,start_id_list[i])
        # print('hi')
        bin_edges = np.linspace(np.min(FM_sig_Q_LB_1[fts_Q_10[0][1]]), np.max(FM_sig_Q_LB_1[fts_Q_10[0][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_1[fts_Q_10[0][1]], bin_edges)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        bin_edges = np.linspace(np.min(FM_sig_Q_LB_0[fts_Q_10[0][1]]), np.max(FM_sig_Q_LB_0[fts_Q_10[0][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_0[fts_Q_10[0][1]], bin_edges)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        if 'HN16_flow'in fts_Q_10[0][1]:
            axq[i].set_xlabel('\n'.join(wrap('{}:HN16_water{} ;p value ={:.2e}'.format(number_oder[i],fts_Q_10[0][1][14:],pvs_Q_10[0]),30)),fontsize=12)
        elif 'CO2' in fts_Q_10[0][1]:
            axq[i].set_xlabel('\n'.join(wrap('{}:HN16_$CO_{{2}}${} ;p value ={:.2e}'.format(number_oder[i],fts_Q_10[0][1][8:],pvs_Q_10[0]),30)),fontsize=12)
        else:   
            axq[i].set_xlabel('\n'.join(wrap('{}:{} ;p value ={:.2e}'.format(number_oder[i],fts_Q_10[0][1][5:],pvs_Q_10[0]),30)),fontsize=12)
        axq[i].set_ylabel('Frequency',fontsize=12)
        axq[i].legend(loc='best',fontsize=12)
        plt.setp(axq[i].get_xticklabels(), Fontsize=10)
        plt.setp(axq[i].get_yticklabels(), Fontsize=10)
    plt.suptitle('$\lambda_{{th}}$ = {} percentile $\lambda$, $\Delta t$={} days, $\Delta t_j$={} days'.format(percentile_list,Dti_list,Dti_list+4),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\HN09\Allwells_Q_Lambda_th_{}_dti_{}d_dtj_{}d1224_1026.png'.format(percentile_list,Dti_list,Dti_list+4),dpi=500)
    plt.show()
    
    '''
    
    for i in range(len(Dti_list)):
        for j in range(len(percentile_list)):
            for k in range(len(feature_keyword_list)):
                fts_Q_10,pvs_Q_10,FM_sig_Q_LB_1,FM_sig_Q_LB_0=One_well_data(FM_data_path,Dti_list[i],percentile_list[j],well_keyword,feature_keyword_list[k],features_num_plot)
                f,axq=plt.subplots(2,5,figsize=(19,11))
                f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.97)
                axq=axq.ravel()
                Nbins=25
                for ii in range(len(fts_Q_10)):
                    bin_edges = np.linspace(np.min(FM_sig_Q_LB_1[fts_Q_10[ii][1]]), np.max(FM_sig_Q_LB_1[fts_Q_10[ii][1]]), Nbins+1)
                    h,e = np.histogram(FM_sig_Q_LB_1[fts_Q_10[ii][1]], bin_edges)
                    axq[ii].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
                    bin_edges = np.linspace(np.min(FM_sig_Q_LB_0[fts_Q_10[ii][1]]), np.max(FM_sig_Q_LB_0[fts_Q_10[ii][1]]), Nbins+1)
                    h,e = np.histogram(FM_sig_Q_LB_0[fts_Q_10[ii][1]], bin_edges)
                    axq[ii].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
                    axq[ii].set_xlabel('\n'.join(wrap('{} and p value ={:.2e}'.format(fts_Q_10[ii][1],pvs_Q_10[ii]),30)))
                    axq[ii].set_ylabel('Frequency')
                    axq[ii].legend(loc='best')
                plt.suptitle('Husmuli_$Q_{{total}}$_FM_KW-{}--$\lambda_{{th}}$ = {} percentile $\lambda$, $\Delta t_i$={} days,$\Delta t_j$={} days'.format
                    (feature_keyword_list[k],percentile_list[j],Dti_list[i],Dti_list[i]+4),fontsize=14)
                plt.tight_layout()
                # plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\Q_total\top10_after_decluster2015\Qt_{}_Lam_th_{}_dti_{}d_dtj_{}d_de2015.png'.format
                #     (savefile_name_list[k],percentile_list[j],Dti_list[i],Dti_list[i]+4),dpi=500)
                plt.close('all')
    '''

    # FM_us=One_well_data(FM_data_path[0],Dti_list[0],percentile_list[0],well_keyword[0],feature_keyword_list[0],features_num_plot)[0]
    

def One_well_data(FM_path,Dti,percentile,well_keyword,feature_keyword,features_num_plot,start_id):
    df=pd.read_csv(r'Z:\MLproject\whakaari-master\data\Husmuli_earthquake_events.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    date_all=df.index
    mag_all=df['magnitude']
    date_filter=np.where((mag_all>0.2))## filter data to magnitude completeness
    date=date_all[date_filter[0]]
    mag=mag_all[date_filter[0]]
    t0=datetime.strptime('20120101','%Y%m%d')
    times=list(date)
    ##decluster to ignore the aftershocks effect
    ind_at_25big=np.where(np.array(mag)>=2.5)[0]##the index position of event larger than 2.5
    time_at_25big=np.array(times)[ind_at_25big]
    week_num=1.### decluster 1 weeks events after big events
    weeks_dt=timedelta(days=7*week_num)
    for j in range(len(time_at_25big)):
        times_decluster=[times[i] for i in range(len(times)) if(times[i]<=time_at_25big[j] or times[i]>=time_at_25big[j]+weeks_dt)]
        times=times_decluster

    df_FM=pd.read_csv(FM_path,index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=df_FM.index
    Dti_df=timedelta(days=1*Dti)###1 week Dti
    Dtj=Dti+4
    fl = 'eqrate_{:3.2f}days_{}2021.pkl'.format(Dti,well_keyword)
    if not os.path.isfile(fl):
        rate_ti = alt(np.array(times), end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/(Dti/7.) ## scaled to per week unit
        # rate_ti2=cal_lambda(Dti,ti,times)
        save_dataframe(rate_ti, fl)
    rate_ti = load_dataframe(fl)
    
    Dtj_df=timedelta(days=1*Dtj)###2 week Dti
    ##calculate the label vector
    lambda_th=np.percentile(rate_ti,percentile)##50 percentile  
    Y_1=np.array([1 if rate>=lambda_th else 0 for rate in rate_ti])
    tj = ti[-len(rate_ti):]
    Y_Dtj = np.zeros(tj.shape, dtype=int)
    for i in np.where(Y_1==1)[0]:
        Y_Dtj[np.where((tj<tj[i])&(tj>(tj[i]-Dtj_df)))] = 1

    ##undersample
    rus=RandomUnderSampler(1,random_state=0)
    df_FM_us,Y_Dtj_us=rus.fit_resample(df_FM,Y_Dtj)
    n=df_FM_us.shape[0]//2
    df_FM_us=pd.concat([df_FM_us.iloc[:n],df_FM_us.iloc[n:]])
    Y_Dtj_us=pd.Series(np.concatenate([np.zeros(n), np.ones(n)]),index=range(df_FM_us.shape[0]))
    select=FeatureSelector(n_jobs=0,ml_task='classification')
    df_FM_us.index=Y_Dtj_us.index
    select.fit_transform(df_FM_us,Y_Dtj_us)
    fts=select.features
    pvs=select.p_values
    fts_Q=[(ind,s) for ind,s in enumerate(fts) if feature_keyword in s]
    pvs_Q=pvs[[i[0] for i in fts_Q][:]]
    fts_Q_10=fts_Q[start_id:features_num_plot+start_id]
    pvs_Q_10=pvs[[i[0] for i in fts_Q_10][:]]
    FM_sig_Q=df_FM_us[[i[1] for i in fts_Q_10]]
    FM_sig_Q_LB_1=FM_sig_Q.loc[Y_Dtj_us.index[Y_Dtj_us==1]]##get the feature values when lable vector=1 for Q
    FM_sig_Q_LB_0=FM_sig_Q.loc[Y_Dtj_us.index[Y_Dtj_us==0]]##get the feature values when lable vector=0 for Q

    return fts_Q_10,pvs_Q_10,FM_sig_Q_LB_1,FM_sig_Q_LB_0

if __name__ == "__main__":
    
    main()


'''

def one_well_data(FM_path,column_keyword,Dti,percentile):
    

    

    
    
    
    
    
    
    
    df_FM_list=list()
    
    

    N=df_FM_us.shape[0]
    





    for percentile in percentile_list:
        lambda_th=np.percentile(rate_ti,percentile)##50 percentile  
        Y_1=np.array([1 if rate>=lambda_th else 0 for rate in rate_ti])
        tj = ti[-len(rate_ti):]
        # Dtj=(2*7*24*60*60)/(3600*24*365.25) ## 2 weeks--look-back time gap
        # Y_Dtj=list()##look-back label vector 
        # t_Dtj=list()
        # for i in ti:
        #     min_index=np.where(np.array(ti)>=i-Dtj)[0][0]
        #     Y_Dtj.append(Y_1[min_index])
        Y_Dtj = np.zeros(tj.shape, dtype=int)
        for i in np.where(Y_1==1)[0]:
            Y_Dtj[np.where((tj<tj[i])&(tj>(tj[i]-Dtj)))] = 1
        ## undersample first then downsample
        rus=RandomUnderSampler(1,random_state=0)
        df_FM_us,Y_Dtj_us=rus.fit_resample(df_FM,Y_Dtj)
        df_FM_list.append(df_FM_us)
    N=np.min([i.shape[0] for i in df_FM_list])##N needs to be achieved for all Dti, Dtq, percentile and rsd
    for j in range(len(df_FM_list)):
        n=df_FM_list[j].shape[0]//2
        df_FM_us=pd.concat([df_FM_list[j].iloc[:n].sample(n=N//2),df_FM_list[j].iloc[n:].sample(n=N//2)])
        # df_FM_us_series=df_FM_us.squeeze()## convert to pandas series
        # df_FM_us_series.index=range(df_FM_us.shape[0])
        Y_Dtj_us=pd.Series(np.concatenate([np.zeros(N//2), np.ones(N//2)]),index=range(df_FM_us.shape[0]))
        select=FeatureSelector(n_jobs=0,ml_task='classification')
        # Y_Dtj=pd.Series(Y_Dtj,index=range(len(Y_Dtj)))
        df_FM_us.index=Y_Dtj_us.index
        select.fit_transform(df_FM_us,Y_Dtj_us)
        # Nfts=10
        fts=select.features
        pvs=select.p_values
        ## the first 10 most significant features for injection rate
        pvs_Q_median=pvs[np.where(np.array(fts)==column_keyword)[0][0]]
        FM_Q_median=df_FM_us[column_keyword]
        FM_sig_Q_LB_1=FM_Q_median.loc[Y_Dtj_us.index[Y_Dtj_us==1]]##get the feature values when lable vector=1 for Q
        FM_sig_Q_LB_0=FM_Q_median.loc[Y_Dtj_us.index[Y_Dtj_us==0]]##get the feature values when lable vector=0 for Q
    return FM_sig_Q_LB_1,FM_sig_Q_LB_0,pvs_Q_median

def main():
    f,axq = plt.subplots(1,3,figsize=(12,6))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.965)
    axq=axq.ravel()
    # Nbins=20
    FM_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_Q_total_0726.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_RK23_0726.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_RK24_0726.csv']
    column_keyword=['Q_total(t/h)__median','Flow_rate(t/h)__median','Flow_rate(t/h)__median']
    # column_keyword=['Q_total(t/h)__quantile__q_0.9','Flow_rate(t/h)__median','Flow_rate(t/h)__median']
    well_keyword=['Q_total','RK23','RK24']
    well_keyword=['Q_total','Well #2','Well #3']
    number_oder=['(a)','(b)','(c)']
    for i in range(len(FM_data_path)):
        FM_sig_Q_LB_1,FM_sig_Q_LB_0,pvs_Q_median=one_well_data(FM_data_path[i],column_keyword[i])
        bin_edges_1 = np.linspace(np.min(FM_sig_Q_LB_1), np.max(FM_sig_Q_LB_1), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_1, bin_edges_1)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        bin_edges_0 = np.linspace(np.min(FM_sig_Q_LB_0), np.max(FM_sig_Q_LB_0), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_0, bin_edges_0)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        axq[i].set_xlabel('\n'.join(wrap('{} {}: {} and p value ={:.2e}'.format(number_oder[i],well_keyword[i],column_keyword[i],pvs_Q_median),30)))
        axq[i].set_ylabel('Frequency')
        axq[i].legend(loc='best')
    plt.suptitle('$\lambda_{{th}}$ = {} percentile $\lambda$, $\Delta t_i$={} days,$\Delta t_j$={} days'.format(percentile_list[0],Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\Q_total\T3wells_Q_Lambda_th_{}_dti_{}d_dtj_{}d08021.png'.format(percentile_list[0],Dti,Dtj_df),dpi=500)
    plt.show()

    
''' 










