import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2
from whakaari import datetimeify
from tsfresh.transformers import FeatureSelector
from textwrap import wrap
from imblearn.under_sampling import RandomUnderSampler
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

percentile_list=[50]
Dti=20.##1 week Dti
Dtj_df=Dti+4
Nbins=60
def one_well_data(FM_path,column_keyword):
    df=pd.read_csv('event_information_noid.csv')
    date_all=df['eventTIME']
    lat_all=df['event_latitude']
    lon_all=df['event_longitude']
    mag_all=df['event_magnitude']
    depth_all=df['event_depth']/1000.
    # distance_filter=np.where((lon_all>176.15)&(lon_all<176.25)&(lat_all>=-38.65)&(lat_all<-38.57)&(depth_all<6.0)&(mag_all>0.5))#&(lon_all>=176.15)&(lon_all<=176.25)
    distance_filter=np.where((lon_all>176.19)&(lon_all<176.22)&(lat_all>=-38.625)&(lat_all<-38.60)&(depth_all<6.0)&(mag_all>0.5))#&(lon_all>=176.15)&(lon_all<=176.25)
    date=date_all[distance_filter[0]]
    lat=lat_all[distance_filter[0]]
    lon=lon_all[distance_filter[0]]
    mag=mag_all[distance_filter[0]]
    depth=depth_all[distance_filter[0]]
    t0=datetime.strptime('20120101','%Y%m%d')
    times=[]
    for each_date in date:
        # str_date=str(int(each_date))
        t=datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ")
        times.append(t)
        # dt=t-t0
        # times.append(dt.total_seconds()/(3600*24*365.25)+2012)

    df_FM=pd.read_csv(FM_path,index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=df_FM.index
    # nrows=len(ti)
    
    
    Dti_df=timedelta(days=1*Dti)###1 week Dti
    # rate_ti = alt(t, end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/Dti
    EQs=list()
    rate_ti=list()
    for i in ti:
        EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
    # calculate the lambda(ti, Delta_ti) using the methods of EQs(ti-Delta_ti,ti)/Delta_ti
    # Dti=(1*7*24*60*60)/(3600*24*365.25) ##1 week time gap, with unit concerted to year
    #built N+1 time gaps for the earthquake event between 2012-2015; the left 2015-2015.9will be used for training
    # ti = np.linspace(2012,2015,1000) ##1000 ti are generated between 2012-2015
    rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
    
    Dtj=timedelta(days=1*Dtj_df)###2 week Dti
    
    df_FM_list=list()
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
        fts_WHP=[(ind,s) for ind,s in enumerate(fts) if (column_keyword in s and 'diff_WHP' not in s)]
        pvs_WHP=pvs[[i[0] for i in fts_WHP][:]]
        fts_WHP_1=fts_WHP[:1]
        pvs_WHP_1=pvs[[i[0] for i in fts_WHP_1][:]]

        FM_WHP=df_FM_us[[i[1] for i in fts_WHP_1]]
        FM_sig_Q_LB_1=FM_WHP.loc[Y_Dtj_us.index[Y_Dtj_us==1]]##get the feature values when lable vector=1 for Q
        FM_sig_Q_LB_0=FM_WHP.loc[Y_Dtj_us.index[Y_Dtj_us==0]]##get the feature values when lable vector=0 for Q
    return FM_sig_Q_LB_1,FM_sig_Q_LB_0,pvs_WHP_1,fts_WHP_1

def main():
    f,axq = plt.subplots(1,2,figsize=(8,5))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.965)
    axq=axq.ravel()
    # Nbins=20
    FM_data_path=[r'H:\UOAPC_Zdrive\MLproject\whakaari-master\features\test_features_RK20_0622.csv',
        r'H:\UOAPC_Zdrive\MLproject\whakaari-master\features\test_features_RK24_0726.csv']
    column_keyword=['WHP','WHP']
    # column_keyword=['Q_total(t/h)__quantile__q_0.9','Flow_rate(t/h)__median','Flow_rate(t/h)__median']
    # well_keyword=['Q_total','RK20','RK23','RK24']
    well_keyword=['Well #1','Well #3']
    number_oder=['(a)','(b)']
    for i in range(len(FM_data_path)):
        FM_sig_Q_LB_1,FM_sig_Q_LB_0,pvs_WHP_1,fts_WHP_1=one_well_data(FM_data_path[i],column_keyword[i])
        FM_sig_Q_LB_1=np.array(FM_sig_Q_LB_1)
        bin_edges_1 = np.linspace(np.min(FM_sig_Q_LB_1), np.max(FM_sig_Q_LB_1), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_1, bin_edges_1)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        FM_sig_Q_LB_0=np.array(FM_sig_Q_LB_0)
        bin_edges_0 = np.linspace(np.min(FM_sig_Q_LB_0), np.max(FM_sig_Q_LB_0), Nbins+1)

        h,e = np.histogram(FM_sig_Q_LB_0, bin_edges_0)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        axq[i].set_xlabel('\n'.join(wrap('{} {}: Pressure{}; p_value ={:.2e}'.format(number_oder[i],well_keyword[i],fts_WHP_1[0][1][9:],pvs_WHP_1[0]),30)),fontsize=12)
        axq[i].set_ylabel('Frequency',fontsize=12)
        axq[i].legend(loc='best',fontsize=12)
        plt.setp(axq[i].get_xticklabels(), Fontsize=10)
        plt.setp(axq[i].get_yticklabels(), Fontsize=10)
    axq[1].set_xlim([-6,5])
    plt.suptitle('$\lambda_{{th}}$ = {} percentile $\lambda$, $\Delta t$={} days, $\Delta t_j$={} days'.format(percentile_list[0],Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'H:\UOAPC_Zdrive\MLproject\whakaari-master\Q_total\T3wells_Q_Lambda_th_{}_dti_{}d_dtj_{}d08021_4W_WHP_1026.png'.format(percentile_list[0],Dti,Dtj_df),dpi=500)
    plt.show()

    
    








if __name__ == "__main__":
    
    main()

