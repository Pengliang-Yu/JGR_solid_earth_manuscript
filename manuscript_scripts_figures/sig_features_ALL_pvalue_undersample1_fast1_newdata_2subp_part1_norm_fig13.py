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


def Onewell(well_data_path):
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
    # times=[]
    # for each_date in date:
    #     # str_date=str(int(each_date))
    #     t=datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ")
    #     times.append(t)
    #     # dt=t-t0
    #     # times.append(dt.total_seconds()/(3600*24*365.25)+2012)
    times = np.array([datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ") for each_date in date])
    df_FM=pd.read_csv(well_data_path,index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=[datetimeify(i) for i in df_FM.index]
    # ti = df_FM.index
    # lambda_th_list=np.linspace(5,10,6)
    # lambda_th_list=[5]
    percentile_list=[50,80]
    # percentile_list=[30,50,60,70,80,90]
    # percentile_list=[80]
    
    FM_list=list()
    # Y_Dtj_list=list()
    
    
    for lambda_percentile in percentile_list:
        
        DD=np.linspace(2,20,10)
        # DD=np.linspace(2,4,2)
        for Dti in DD:
            # Dti=2.##1 week Dti
            Dti_df=timedelta(days=1*Dti)###1 week Dti
            rate_ti = alt(times, end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/Dti
            # EQs=list()
            # rate_ti=list()
            # for i in ti:
            #     EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
            # calculate the lambda(ti, Delta_ti) using the methods of EQs(ti-Delta_ti,ti)/Delta_ti
            # Dti=(1*7*24*60*60)/(3600*24*365.25) ##1 week time gap, with unit concerted to year
            #built N+1 time gaps for the earthquake event between 2012-2015; the left 2015-2015.9will be used for training
            # ti = np.linspace(2012,2015,1000) ##1000 ti are generated between 2012-2015
            # rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
            lambda_th=np.percentile(rate_ti,lambda_percentile)
            # tj = ti[-len(rate_ti):]
            # lambda_th=6##unit per week
            Y_1=[1 if rate>=lambda_th else 0 for rate in rate_ti]
            # Dtj_df=5
            Dtj_df=Dti+4
            Dtj=timedelta(days=1*Dtj_df)###2 week Dti
            # Dtj=(2*7*24*60*60)/(3600*24*365.25) ## 2 weeks--look-back time gap
            Y_Dtj=list()##look-back label vector 
            t_Dtj=list()
            for i in ti:
                min_index=np.where(np.array(ti)>=i-Dtj)[0][0]
                Y_Dtj.append(Y_1[min_index])
            # Y_Dtj = np.zeros(tj.shape, dtype=int)
            # for i in np.where(Y_1==1)[0]:
            #     Y_Dtj[np.where((tj<tj[i])&(tj>(tj[i]-Dtj)))] = 1
            #undersample data
            rus=RandomUnderSampler(1,random_state=0)
            df_FM_us,Y_Dtj_us=rus.fit_resample(df_FM,Y_Dtj)
            FM_list.append(df_FM_us)
            # Y_Dtj_list.append(Y_Dtj_us)

    N=np.min([i.shape[0] for i in FM_list])
    DD=np.linspace(2,20,10)
    # pvalue_Q_list=[]
    try_time=10
    pvalue_Q_all=[]
    for i in FM_list:
        # pvalue_ave_one_list=[]
        n=i.shape[0]//2
        for ii in range(try_time):

            df_FM_us = pd.concat([i.iloc[:n].sample(n=N//2),i.iloc[n:].sample(n=N//2)])
            Y_Dtj_us=pd.Series(np.concatenate([np.zeros(N//2), np.ones(N//2)]),index=range(len(df_FM_us)))
            select=FeatureSelector(n_jobs=0,ml_task='classification')
            df_FM_us.index=Y_Dtj_us.index
            select.fit_transform(df_FM_us,Y_Dtj_us)
                
            fts=select.features
            pvs=select.p_values
            fts_Q=[(ind,s) for ind,s in enumerate(fts) if (('Flow_rate' in s and 'diff_Flow_rate' not in s) or ('Integ_qdt' in s))]
            pvs_Q=pvs[[i[0] for i in fts_Q][:]]
            pvs_Q= [x for x in pvs_Q if ~np.isnan(x)]
            pvalue_Q_all.append(pvs_Q)
    min_size_pvalue_Q_all=min([len(i) for i in pvalue_Q_all])
    return min_size_pvalue_Q_all, pvalue_Q_all


  
try_time=10
def main():
    # well_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_RK20_0622.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_RK23_0622.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_RK24_0622.csv',
    #     r'Z:\MLproject\whakaari-master\features\test_features_Q_total_0622.csv']
    
    well_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_RK20_part1_1024_norm.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_RK23_part1_1024_norm.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_RK24_part1_1024_norm.csv']
    # well_name=['RK20 well','RK23 well','Rk24 well','Q_total']
    well_name=['RK20 well','RK23 well','RK24 well']
    # well_name=['Well #1-Part 1','Well #2-Part 1','Well #3-Part 1']
    well_name=['Well #1','Well #2','Well #3']
    # percentile_list=[50,60,70,80,90]
    percentile_list=[50,80]
    # percentile_list=[30,50,60,70,80,90]
    # colors=['r','b','k','c','g']
    colors=['r','b','k']
    DD=np.linspace(2,20,10)
    # f,ax = plt.subplots(2,3,figsize=(18,12))
    f,ax = plt.subplots(1,2,figsize=(12,6))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.95)
    ax=ax.ravel()
    # min_size=100000000
    min_size_pvalue=min([Onewell(well_data_path[i])[0] for i in range(len(well_data_path))])
    # pvalue_Q_list.append(np.mean(np.array(pvalue_ave_one_list)))
    
    # pvalue_list_diffPercentile=np.array(pvalue_list).reshape((len(percentile_list),len(DD)))
    # return pvalue_list_diffPercentile  
    for i in range(len(percentile_list)):
        for j in range(len(well_data_path)):
            onewell_pvalue_list=Onewell(well_data_path[j])[1]
            pvalue_ave_one_list=[]
            pvalue_Q_list=[]
            for ii in range(len(onewell_pvalue_list)):
                pvalue_ave_one_list.append(np.sum(np.log10(onewell_pvalue_list[ii][:min_size_pvalue])))
            pvalue_Q_list=np.array(pvalue_ave_one_list).reshape((len(percentile_list),len(DD),try_time))
            onewell_pvalue_list_diffPercentile=np.mean(np.array(pvalue_Q_list),axis=2)
            # =Onewell(well_data_path[j])

            ax[i].plot(DD,onewell_pvalue_list_diffPercentile[i,:],color=colors[j],marker='s',ms=5,label=well_name[j])
            ax[i].set_xlabel('$\Delta t$ (days)--time scales for averaging microseismicity rates',fontsize=12)
            ax[i].set_ylabel('Sum of $log_{10}$ p-value for all features',fontsize=12)
            ax[i].set_title('Pre Jul. 2013 period, $\lambda_{{th}}$ = {} percentile $\lambda$'.format(percentile_list[i]),fontsize=12)
            ax[i].legend(loc='best',fontsize=12)
            ax[i].set_ylim([-18500,-3000])
            plt.setp(ax[i].get_xticklabels(), Fontsize=10)
            plt.setp(ax[i].get_yticklabels(), Fontsize=10)
    ax[0].text(0.05, 0.05, 'a',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[1].text(0.95, 0.95, 'b',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
    '''
    for i in range(len(well_data_path)):
        onewell_pvalue_list_diffPercentile=Onewell(well_data_path[i])
        for j in range(len(percentile_list)):
            ax[i].plot(DD,onewell_pvalue_list_diffPercentile[j,:],color=colors[j],marker='s',ms=5,label='$\lambda_{{th}}$ = {} percentile $\lambda$'.format(percentile_list[j]))
            ax[i].set_xlabel('$\Delta t_i$ (days)--time scales for averaging seimicity rates;$\Delta t_j = \Delta t_i + 4 $')
            ax[i].set_ylabel('Sum of log10 of all features p values ')
            ax[i].set_title('{}'.format(well_name[i]),fontsize=10)
            ax[i].legend(loc='best')
    '''
    # ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\Threewells_pva_percentile_balanced_reduced_newda_2subp1_part1_norm1026.png',dpi=500)
    plt.show()



if __name__ == "__main__":
    
    main()


