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



def main():
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
    '''
    df_FM=pd.read_csv(r'Z:\MLproject\whakaari-master\features\test_features_Q_total_0726.csv',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=[datetimeify(i) for i in df_FM.index]
    
    Dti=1.##1 week Dti
    Dti_df=timedelta(days=7*Dti)###1 week Dti
    EQs=list()
    rate_ti=list()
    for i in ti:
        EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
    # calculate the lambda(ti, Delta_ti) using the methods of EQs(ti-Delta_ti,ti)/Delta_ti
    # Dti=(1*7*24*60*60)/(3600*24*365.25) ##1 week time gap, with unit concerted to year
    #built N+1 time gaps for the earthquake event between 2012-2015; the left 2015-2015.9will be used for training
    # ti = np.linspace(2012,2015,1000) ##1000 ti are generated between 2012-2015
    rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
    lambda_th=70.##unit per week
    Y_1=[1 if rate>=lambda_th else 0 for rate in rate_ti]
    Dtj_df=2
    Dtj=timedelta(days=7*Dtj_df)###2 week Dti
    # Dtj=(2*7*24*60*60)/(3600*24*365.25) ## 2 weeks--look-back time gap
    Y_Dtj=list()##look-back label vector 
    t_Dtj=list()
    for i in ti:
        min_index=np.where(np.array(ti)>=i-Dtj)[0][0]
        Y_Dtj.append(Y_1[min_index])
    '''
    
    
    
    df_Q=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_husmuli_hour_unit_CO2.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ind0=np.where(df_Q.index==datetimeify('2015-01-01'))[0][0]
    df_Q=df_Q.iloc[ind0:,:]
    fp,ax=plt.subplots(1,1,figsize=(12,7))
    col_name=['HN09_flow(t/h)','HN12_flow(t/h)','HN14_flow(t/h)','HN16_flow(t/h)','HN16_flow(t/h)','HN17_flow(t/h)']
    cc=['b-','r-','g-','k-','y-','c-']
    # ls=[]
    # for i in range(len(col_name)):
    #     if i==0:
    #         Q_total=df_Q[col_name[i]]
    #         ax.plot(df_Q.index,Q_total,cc[i],linewidth=1.0,label='Total injection rate')
            
    #     else:
    #         Q_well=df_Q[col_name[i]]
    #         ax.plot(df_Q.index,Q_well,cc[i],linewidth=1.5,label='{} well'.format(col_name[i][:4]))
            
    for i in range(len(col_name)):
        if i==3:

            Q_total=df_Q[col_name[i]]
            ax.plot(df_Q.index,Q_total,cc[i],linewidth=1.0,label='HN16 well_water')
        elif i==4:
            
            Q_total=df_Q[col_name[i]]
            ax.plot(df_Q.index,Q_total,cc[i],linewidth=1.0,label=r'HN16 well_$CO_2$')   
        else:
            Q_well=df_Q[col_name[i]]
            ax.plot(df_Q.index,Q_well,cc[i],linewidth=1.5,label='{} well'.format(col_name[i][:4]))
            
    ax.set_xlabel('Time(year)',fontsize=14)
    ax.set_ylabel('Injection rate(t/h)',fontsize=14)
    # lgs=[l.get_label() for l in ls]
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ##calculate the seismicity rate and plot it
    from copy import copy
    ti = copy(df_Q.index)
    def norm_asym(m,s,x):
        return (x>m)*np.exp(-0.5*(x-m)**2/s**2)/s/np.sqrt(2*np.pi)*2
    def KDE_asym(points, bandwidth, x):
        '''
            scale accounts for # eqs, conversion to weekly
        '''
        scale = np.sum((points>x[0])&(points<x[-1]))*7*4*3
        return np.mean([norm_asym(pt,bandwidth,x) for pt in points],axis=0)*scale
    ts = np.array([(t-ti[0]).total_seconds()/86400 for t in times])
    tv = np.array([(t-ti[0]).total_seconds()/86400 for t in ti])
    rate_ti = KDE_asym(ts,3.5*4*3,tv)

    
    # ax[1].plot(ti,rate_ti,'m-',linewidth=1.5,label='Seismicity rate')
    # ax[1].set_xlabel('Time(year)',fontsize=14)
    
    # ax[1].set_ylabel(r'Seismicity rate ($Week^{-1}$)',fontsize=14)
    # ax.plot([],[],'m-',linewidth=1.5,label='Seismicity rate')
    # plt.setp(ax[1].get_xticklabels(), fontsize=12)
    # plt.setp(ax[1].get_yticklabels(), fontsize=12)
    # ax[1].legend(loc='best',fontsize=14)

    leg=ax.legend(bbox_to_anchor=(0.,1.02,1.,.102),loc='lower left',ncol=3,mode='expand',borderaxespad=0.,fontsize=14)
    plt.tight_layout()
    # ax[0].text(0.02, 0.9, 'A',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    # ax[1].text(0.02, 0.9, 'B',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)        
    plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\Qvstime_vslambda1211.png',dpi=500)
    plt.show()








if __name__ == "__main__":
    
    main()


    



