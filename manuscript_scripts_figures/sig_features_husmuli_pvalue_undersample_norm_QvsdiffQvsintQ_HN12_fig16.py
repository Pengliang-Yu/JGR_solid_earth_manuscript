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

    df_FM=pd.read_csv(r'Z:\MLproject\whakaari-master\features\test_features_HN12_norm2015.csv',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ti=df_FM.index
    
    # lambda_th_list=np.linspace(5,10,6)
    # lambda_th_list=[5]
    # percentile_list=[50,60,70,80,90]
    percentile_list=[50]
    # percentile_list=[50]
    # percentile_list=[80]
    f,ax = plt.subplots(1,1,figsize=(7,6))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.95)
    FM_list=list()
    # Y_Dtj_list=list()
    for lambda_percentile in percentile_list:
        
        DD=np.linspace(2,24,8)
        # DD=[2]
        # DD=np.linspace(2,4,2)
        for Dti in DD:
            # Dti=2.##1 week Dti
            Dti_df=timedelta(days=1*Dti)###1 week Dti
            EQs=list()
            rate_ti=list()
            rate_ti = alt(np.array(times), end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/(Dti)
            # for i in ti:
            #     EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
            # # calculate the lambda(ti, Delta_ti) using the methods of EQs(ti-Delta_ti,ti)/Delta_ti
            # # Dti=(1*7*24*60*60)/(3600*24*365.25) ##1 week time gap, with unit concerted to year
            # #built N+1 time gaps for the earthquake event between 2012-2015; the left 2015-2015.9will be used for training
            # # ti = np.linspace(2012,2015,1000) ##1000 ti are generated between 2012-2015
            # rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
            lambda_th=np.percentile(rate_ti,lambda_percentile)
            # lambda_th=6##unit per week
            Y_1=np.array([1 if rate>lambda_th else 0 for rate in rate_ti])
            Dtj=Dti+4
            Dtj_df=timedelta(days=1*Dtj)###2 week Dti
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
    DD=np.linspace(2,24,8)
    pvalue_Q_list=[]
    pvalue_whp_list=[]
    pvalue_QP_list=[]
    pvalue_int_qdt_list=[]
    pvalue_QT_list=[]
    pvalue_int_QT_list=[]
    try_time=10
    for i in FM_list:
        pvalue_Q_ave_one_list=[]
        pvalue_whp_ave_one_list=[]
        pvalue_QP_ave_one_list=[]
        pvalue_int_qdt_ave_one_list=[]
        pvalue_QT_ave_one_list=[]
        pvalue_int_QT_ave_one_list=[]
        n=i.shape[0]//2
        for ii in range(try_time):
            # df_FM_us = pd.concat([i.iloc[:N//2],i.iloc[n:n+N//2]])
            df_FM_us = pd.concat([i.iloc[:n].sample(n=N//2),i.iloc[n:].sample(n=N//2)])
            Y_Dtj_us=pd.Series(np.concatenate([np.zeros(N//2), np.ones(N//2)]),index=range(len(df_FM_us)))
            select=FeatureSelector(n_jobs=0,ml_task='classification')
            df_FM_us.index=Y_Dtj_us.index
            select.fit_transform(df_FM_us,Y_Dtj_us)
                
            fts=select.features
            pvs=select.p_values

            fts_Q=[(ind,s) for ind,s in enumerate(fts) if ('norm_HN12_flow(t/h)' in s and 'diff_norm_HN12_flow(t/h)' not in s)]
            pvs_Q=pvs[[i[0] for i in fts_Q][:]]
            pvs_Q= [x for x in pvs_Q if ~np.isnan(x)]
            

            
            fts_whp=[(ind,s) for ind,s in enumerate(fts) if ('diff_norm_HN12_flow(t/h)' in s)]
            pvs_whp=pvs[[i[0] for i in fts_whp][:]]
            pvs_whp=[x for x in pvs_whp if ~np.isnan(x)]
            

            fts_QP=[(ind,s) for ind,s in enumerate(fts) if ('norm_Integ_HN12(t)' not in s)]
            pvs_QP=pvs[[i[0] for i in fts_QP][:]]
            pvs_QP=[x for x in pvs_QP if ~np.isnan(x)]

            # fts_int_qdt=[(ind,s) for ind,s in enumerate(fts) if ('Integ_qdt(t)' in s and 'diff_Integ_qdt' not in s)]
            # pvs_int_qdt=pvs[[i[0] for i in fts_int_qdt][:]]
            # pvs_int_qdt=[x for x in pvs_int_qdt if ~np.isnan(x)]

            # fts_QT=[(ind,s) for ind,s in enumerate(fts) if ('Q*T_norm' in s and 'diff_Q*T_norm' not in s and 'Integ_Q*T_norm' not in s)]
            # pvs_QT=pvs[[i[0] for i in fts_QT][:]]
            # pvs_QT=[x for x in pvs_QT if ~np.isnan(x)]

            # fts_int_QT=[(ind,s) for ind,s in enumerate(fts) if ('Integ_Q*T_norm' in s and 'diff_Integ_Q*T_norm' not in s)]
            # pvs_int_QT=pvs[[i[0] for i in fts_int_QT][:]]
            # pvs_int_QT=[x for x in pvs_int_QT if ~np.isnan(x)]

            len_pvs_Q=len(pvs_Q)
            len_pvs_whp=len(pvs_whp)
            len_pvs_QP=len(pvs_QP)
            # len_pvs_int_qdt=len(pvs_int_qdt)
            # len_pvs_QT=len(pvs_QT)
            # len_pvs_int_QT=len(pvs_int_QT)

            # minlen_pvs=min(len_pvs_Q,len_pvs_whp,len_pvs_QP,len_pvs_int_qdt,len_pvs_QT,len_pvs_int_QT)##maka a same len of pvalue_Q, whp, Qp
            minlen_pvs=min(len_pvs_Q,len_pvs_whp,len_pvs_QP)
            pvalue_Q_ave_one_list.append(np.sum(np.log10(pvs_Q[:minlen_pvs])))
            pvalue_whp_ave_one_list.append(np.sum(np.log10(pvs_whp[:minlen_pvs])))
            pvalue_QP_ave_one_list.append(np.sum(np.log10(pvs_QP[:minlen_pvs])))
            # pvalue_int_qdt_ave_one_list.append(np.sum(np.log10(pvs_int_qdt[:minlen_pvs])))
            # pvalue_QT_ave_one_list.append(np.sum(np.log10(pvs_QT[:minlen_pvs])))
            # pvalue_int_QT_ave_one_list.append(np.sum(np.log10(pvs_int_QT[:minlen_pvs])))
            
            
            # print('hi')




        pvalue_Q_list.append(np.mean(np.array(pvalue_Q_ave_one_list)))
        pvalue_whp_list.append(np.mean(np.array(pvalue_whp_ave_one_list)))
        pvalue_QP_list.append(np.mean(np.array(pvalue_QP_ave_one_list)))
        # pvalue_int_qdt_list.append(np.mean(np.array(pvalue_int_qdt_ave_one_list)))
        # pvalue_QT_list.append(np.mean(np.array(pvalue_QT_ave_one_list)))
        # pvalue_int_QT_list.append(np.mean(np.array(pvalue_int_QT_ave_one_list)))
    
    pvalue_Q_list_diffPercentile=np.array(pvalue_Q_list).reshape((len(percentile_list),len(DD)))
    pvalue_whp_list_diffPercentile=np.array(pvalue_whp_list).reshape((len(percentile_list),len(DD)))
    pvalue_QP_list_diffPercentile=np.array(pvalue_QP_list).reshape((len(percentile_list),len(DD)))
    # pvalue_int_qdt_list_diffPercentile=np.array(pvalue_int_qdt_list).reshape((len(percentile_list),len(DD)))
    # pvalue_QT_list_diffPercentile=np.array(pvalue_QT_list).reshape((len(percentile_list),len(DD)))
    # pvalue_int_QT_list_diffPercentile=np.array(pvalue_int_QT_list).reshape((len(percentile_list),len(DD)))

    for i in range(len(percentile_list)):
        ax.plot(DD,pvalue_Q_list_diffPercentile[i,:],c='r',marker='s',ms=5,label='q')
        ax.plot(DD,pvalue_whp_list_diffPercentile[i,:],c='k',marker='s',ms=5,label='$\.q$')
        ax.plot(DD,pvalue_QP_list_diffPercentile[i,:],c='b',marker='s',ms=5,label=r'Q')
        # ax.plot(DD,pvalue_int_qdt_list_diffPercentile[i,:],c='g',marker='s',ms=5,label=r'$\int Qdt$')
        # ax.plot(DD,pvalue_QT_list_diffPercentile[i,:],c='c',marker='s',ms=5,label=r'$Q \times T$')
        # ax.plot(DD,pvalue_int_QT_list_diffPercentile[i,:],c='m',marker='s',ms=5,label=r'$\int Q \times T dt$')

        ax.set_xlabel('$\Delta t$(days) --time scales for averaging microseismicity rates',fontsize=12)
        ax.set_ylabel('Sum of $log_{10}$ p_values for all features',fontsize=12)
        ax.set_title('HN12 well, $\lambda_{{th}}$ = {} percentile $\lambda$'.format(percentile_list[i]),fontsize=12)
    ax.legend(loc='best',fontsize=12)
    ax.text(0.95, 0.05, 'b',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.setp(ax.get_xticklabels(), Fontsize=10)
    plt.setp(ax.get_yticklabels(), Fontsize=10)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\HN12\HN12_datastreams_comparison_0808.png',dpi=500)
    plt.show()
    '''  
    ## the first 10 most significant features for injection rate
    fts_Q=[(ind,s) for ind,s in enumerate(fts) if 'Flow_rate' in s]
    pvs_Q=pvs[[i[0] for i in fts_Q][:]]
    fts_whp=[(ind,s) for ind,s in enumerate(fts) if 'WHP' in s]
    pvs_whp=pvs[[i[0] for i in fts_whp][:]]

    fts_Q_P=[(ind,s) for ind,s in enumerate(fts) if 'Q*P' in s]
    pvs_Q_P=pvs[[i[0] for i in fts_Q_P][:]]


    fts_Q_10=fts_Q[:10]
    pvs_Q_10=pvs[[i[0] for i in fts_Q_10][:]]
    fts_whp_10=fts_whp[:10]
    pvs_whp_10=pvs[[i[0] for i in fts_whp_10][:]]
    fts_Q_P_10=fts_Q_P[:10]
    pvs_Q_P_10=pvs[[i[0] for i in fts_Q_P_10][:]]
    ##get the feature values for each significant features
    FM_sig_Q=df_FM[[i[1] for i in fts_Q_10]]
    FM_sig_whp=df_FM[[i[1] for i in fts_whp_10]]
    FM_sig_Q_P=df_FM[[i[1] for i in fts_Q_P_10]]

    FM_sig_Q_LB_1=FM_sig_Q.loc[Y_Dtj.index[Y_Dtj==1]]##get the feature values when lable vector=1 for Q
    FM_sig_Q_LB_0=FM_sig_Q.loc[Y_Dtj.index[Y_Dtj==0]]##get the feature values when lable vector=0 for Q
    FM_sig_whp_LB_1=FM_sig_whp.loc[Y_Dtj.index[Y_Dtj==1]]##get the feature values when lable vector=1 for whp
    FM_sig_whp_LB_0=FM_sig_whp.loc[Y_Dtj.index[Y_Dtj==0]]##get the feature values when lable vector=0 for whp
    FM_sig_Q_P_LB_1=FM_sig_Q_P.loc[Y_Dtj.index[Y_Dtj==1]]##get the feature values when lable vector=1 for q*p
    FM_sig_Q_P_LB_0=FM_sig_Q_P.loc[Y_Dtj.index[Y_Dtj==0]]##get the feature values when lable vector=0 for q*p
    ##plot histogram
    # Nbins = int(np.sqrt(len(FM_sig_Q_LB_1[fts_Q_10[0][1]]))/2)
    ##plot the Q features distribution
    
    f,axq = plt.subplots(2,5,figsize=(19,11))
    f.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.97)
    axq=axq.ravel()
    Nbins=60
    for i in range(len(fts_Q_10)):
        bin_edges = np.linspace(np.min(FM_sig_Q_LB_1[fts_Q_10[i][1]]), np.max(FM_sig_Q_LB_1[fts_Q_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_1[fts_Q_10[i][1]], bin_edges)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        bin_edges = np.linspace(np.min(FM_sig_Q_LB_0[fts_Q_10[i][1]]), np.max(FM_sig_Q_LB_0[fts_Q_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_LB_0[fts_Q_10[i][1]], bin_edges)
        axq[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        axq[i].set_xlabel('\n'.join(wrap('{} and p_value ={:.2e}'.format(fts_Q_10[i][1],pvs_Q_10[i]),30)))
        axq[i].set_ylabel('Frequency')
        axq[i].legend(loc='best')
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/day, $\Delta t_i$={} days,$\Delta t_j$={} days'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_Q_Features_Lambda_th_{}_dti_{}d.png'.format(lambda_th,int(Dti)),dpi=500)
    # plt.tight_layout()
    # plt.show()  
    #plot the whp feature distribution
    fig,axw = plt.subplots(2,5,figsize=(19,11))
    fig.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.97)
    axw=axw.ravel()
    Nbins=60
    for i in range(len(fts_whp_10)):
        bin_edges = np.linspace(np.min(FM_sig_whp_LB_1[fts_whp_10[i][1]]), np.max(FM_sig_whp_LB_1[fts_whp_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_whp_LB_1[fts_whp_10[i][1]], bin_edges)
        axw[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        bin_edges = np.linspace(np.min(FM_sig_whp_LB_0[fts_whp_10[i][1]]), np.max(FM_sig_whp_LB_0[fts_whp_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_whp_LB_0[fts_whp_10[i][1]], bin_edges)
        axw[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        axw[i].set_xlabel('\n'.join(wrap('{} and p_value ={:.2e}'.format(fts_whp_10[i][1],pvs_whp_10[i]),30)))
        axw[i].set_ylabel('Frequency')
        axw[i].legend(loc='best')
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/day, $\Delta t_i$={} days,$\Delta t_j$={} days'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_whp_Features_Lambda_th_{}_dti_{}d.png'.format(lambda_th,int(Dti)),dpi=500)
    #plot the Q*P feature distribution
    ff,axqp = plt.subplots(2,5,figsize=(19,11))
    ff.subplots_adjust(hspace=0.2,wspace=0.2,top=0.95,bottom=0.05,left=0.05,right=0.97)
    axqp=axqp.ravel()
    Nbins=60
    for i in range(len(fts_Q_P_10)):
        bin_edges = np.linspace(np.min(FM_sig_Q_P_LB_1[fts_Q_P_10[i][1]]), np.max(FM_sig_Q_P_LB_1[fts_Q_P_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_P_LB_1[fts_Q_P_10[i][1]], bin_edges)
        axqp[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=1')
        bin_edges = np.linspace(np.min(FM_sig_Q_P_LB_0[fts_Q_P_10[i][1]]), np.max(FM_sig_Q_P_LB_0[fts_Q_P_10[i][1]]), Nbins+1)
        h,e = np.histogram(FM_sig_Q_P_LB_0[fts_Q_P_10[i][1]], bin_edges)
        axqp[i].bar(e[:-1], height = h, width = e[1]-e[0], alpha=0.5,label='Label vector=0')
        axqp[i].set_xlabel('\n'.join(wrap('{} and p_value ={:.2e}'.format(fts_Q_P_10[i][1],pvs_Q_P_10[i]),30)))
        axqp[i].set_ylabel('Frequency')
        axqp[i].legend(loc='best')
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/day, $\Delta t_i$={} days,$\Delta t_j$={} days'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_QxP_Features_Lambda_th_{}_dti_{}d.png'.format(lambda_th,int(Dti)),dpi=500)





    plt.show()
    '''  







if __name__ == "__main__":
    
    main()
'''
for i in ti:
    if i+Dtj<=ti[-1]:
        min_index=np.where(ti>=(i+Dtj))[0][0]## find the minimum value index that bigger than ti+Dtj
        Y_Dtj.append(Y_1[min_index])
        t_Dtj.append(i+Dtj)
'''    
    
'''
f,ax=plt.subplots(1,1,figsize=(9,4))
ax.plot(ti,rate_ti,'k',linewidth=1.0)
ax.plot([min(ti),max(ti)],[lambda_th,lambda_th],'r--',linewidth=1.0,label=r'$\lambda_{th}$')
ax.set_xlabel('Time')
ax.set_ylabel('Seismicity rate, $year^{-1}$')
ax.legend(loc='best')
plt.tight_layout()


fig,ax=plt.subplots(1,1,figsize=(9,4))
# ax.plot(ti,Y_1,'k',linewidth=1.0,label=r'$Y_{01}$')
ax.plot(t_Dtj,Y_Dtj,'b--',linewidth=1.0,label=r'$Y_{\Delta{t_j}}$')
ax.set_xlabel('Time')
ax.set_ylabel('label vector')
ax.legend(loc='best')
'''

    

# print(ti)

