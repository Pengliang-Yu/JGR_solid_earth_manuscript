import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2
from whakaari import datetimeify
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


def Onewell(FM_data_path,DD,well_name,percentile_list):
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
    
    df_FM=pd.read_csv(FM_data_path,index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=df_FM.index
    # percentile_list=[40,80]
    
    FM_list=list()
    # Y_Dtj_list=list()
    
    
    for lambda_percentile in percentile_list:
        
        for Dti in DD:
            
            Dti_df=timedelta(days=1*Dti)###1 week Dti
            fl = 'eqrate_{:3.2f}days_{}2015.pkl'.format(Dti,well_name)
            if not os.path.isfile(fl):
                rate_ti = alt(np.array(times), end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/(Dti) ## no scaled to per week unit
                # rate_ti2=cal_lambda(Dti,ti,times)
                save_dataframe(rate_ti, fl)
            rate_ti = load_dataframe(fl)
            lambda_th=np.percentile(rate_ti,lambda_percentile)
    
            Y_1=[1 if rate>=lambda_th else 0 for rate in rate_ti]
            Dtj=Dti+4
            Dtj_df=timedelta(days=1*Dtj)###2 week Dti
            ##calculate the label vector
            lambda_th=np.percentile(rate_ti,lambda_percentile)##50 percentile  
            Y_1=np.array([1 if rate>lambda_th else 0 for rate in rate_ti])
            tj = ti[-len(rate_ti):]
            Y_Dtj = np.zeros(tj.shape, dtype=int)
            for i in np.where(Y_1==1)[0]:
                Y_Dtj[np.where((tj<tj[i])&(tj>(tj[i]-Dtj_df)))] = 1
            #undersample data
            rus=RandomUnderSampler(1,random_state=0)
            df_FM_us,Y_Dtj_us=rus.fit_resample(df_FM,Y_Dtj)
            FM_list.append(df_FM_us)
            # Y_Dtj_list.append(Y_Dtj_us)

    N=np.min([i.shape[0] for i in FM_list])
    
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
            # fts_Q=[(ind,s) for ind,s in enumerate(fts) 
            #     if (('norm_{}'.format(well_name) in s and 'diff_norm_{}'.format(well_name) not in s) or ('norm_Integ_{}'.format(well_name) in s))]
            fts_Q=[(ind,s) for ind,s in enumerate(fts) 
                if (('norm_{}'.format(well_name) in s and 'diff_norm_{}'.format(well_name) not in s) or ('norm_Integ_{}'.format(well_name) in s))]
            pvs_Q=pvs[[i[0] for i in fts_Q][:]]
            pvs_Q= [x for x in pvs_Q if ~np.isnan(x)]
            pvalue_Q_all.append(pvs_Q)
    min_size_pvalue_Q_all=min([len(i) for i in pvalue_Q_all])
    return min_size_pvalue_Q_all, pvalue_Q_all


  
try_time=10
def main():

    FM_data_path=[r'Z:\MLproject\whakaari-master\features\test_features_HN09_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN12_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN14_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN16_norm2015_CO2.csv',
        r'Z:\MLproject\whakaari-master\features\test_features_HN17_norm2015.csv']
    well_name_list=['HN09','HN12','HN14','HN16','CO2','HN17']
    well_name_legend_list=['HN09','HN12','HN14','HN16_water','HN16_$CO_2$','HN17']
    # well_name=['Well #1','Well #2','Well #3']
    # percentile_list=[50,60,70,80,90]
    percentile_list=[50,80]
    # percentile_list=[30,50,60,70,80,90]
    # colors=['r','b','k','c','g']
    colors=['r','b','k','g','c','y']
    DD=np.linspace(2,20,10)
    # f,ax = plt.subplots(2,3,figsize=(18,12))
    f,ax = plt.subplots(1,2,figsize=(12,6))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.95)
    ax=ax.ravel()
    # min_size=100000000 Onewell(FM_data_path,DD,well_name,percentile_list)
    min_size_pvalue=min([Onewell(FM_data_path[i],DD,well_name_list[i],percentile_list)[0] for i in range(len(FM_data_path))])
    # pvalue_Q_list.append(np.mean(np.array(pvalue_ave_one_list)))
    
    # pvalue_list_diffPercentile=np.array(pvalue_list).reshape((len(percentile_list),len(DD)))
    # return pvalue_list_diffPercentile  
    for i in range(len(percentile_list)):
        for j in range(len(FM_data_path)):
            onewell_pvalue_list=Onewell(FM_data_path[j],DD,well_name_list[j],percentile_list)[1]
            pvalue_ave_one_list=[]
            pvalue_Q_list=[]
            for ii in range(len(onewell_pvalue_list)):
                pvalue_ave_one_list.append(np.sum(np.log10(onewell_pvalue_list[ii][:min_size_pvalue])))
            pvalue_Q_list=np.array(pvalue_ave_one_list).reshape((len(percentile_list),len(DD),try_time))
            onewell_pvalue_list_diffPercentile=np.mean(np.array(pvalue_Q_list),axis=2)
            # =Onewell(well_data_path[j])

            ax[i].plot(DD,onewell_pvalue_list_diffPercentile[i,:],color=colors[j],marker='s',ms=5,label=well_name_legend_list[j])
            ax[i].set_xlabel('$\Delta t$ (days)--time scales for averaging microseismicity rates',fontsize=12)
            ax[i].set_ylabel('Sum of $log_{10}$ p-value for all features',fontsize=12)
            ax[i].set_title('$\lambda_{{th}}$ = {} percentile $\lambda$'.format(percentile_list[i]),fontsize=12)
            plt.setp(ax[i].get_xticklabels(), Fontsize=10)
            plt.setp(ax[i].get_yticklabels(), Fontsize=10)
            
            ax[i].legend(loc='best',fontsize=12)
    ax[0].set_ylim([-20000,-1000])
    ax[1].set_ylim([-20000,-1000])
    ax[0].text(0.95, 0.05, 'a',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[1].text(0.95, 0.05, 'b',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)        
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
    plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\HN09\wellcomparison_decluster2015_2subp45_FQ_FintQ_CO2_10261.png',dpi=500)
    plt.show()



if __name__ == "__main__":
    
    main()


