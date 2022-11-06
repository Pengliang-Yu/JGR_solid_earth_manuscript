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

    df_Q=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Qtotal_0726.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ind=np.where(df_Q.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    t_Q=df_Q.index[:ind]
    # print('hi')
    # t_Q=[datetimeify(i) for i in df_Q.index[:26089]] ##26089 is the time of 2015-01-01 00:00:0
    Q_total=df_Q['Q_total(t/h)'][:ind]
    df_20=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk20_0622.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    df_23=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk23_0726.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    df_24=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk24_0722.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ind20=np.where(df_20.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    ind23=np.where(df_23.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    ind24=np.where(df_24.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    t_20=df_20.index[:ind20]
    t_23=df_23.index[:ind23]
    t_24=df_24.index[:ind24]
    Q_20=df_20['Flow_rate(t/h)'][:ind20]
    Q_23=df_23['Flow_rate(t/h)'][:ind23]
    Q_24=df_24['Flow_rate(t/h)'][:ind24]
    # fp,ax=plt.subplots(1,1,figsize=(12,7))
    fp,ax=plt.subplots(1,1,figsize=(12,7))
    # l1=ax.plot(ti,rate_ti,'r',linewidth=1.0,label='Seismicity rate')
    # l2=ax.plot([min(ti),max(ti)],[lambda_th,lambda_th],'r--',linewidth=1.0,label=r'$\lambda_{th}$')
    Q_total=Q_total*1000/60/60
    Q_20=Q_20*1000/60/60
    Q_23=Q_23*1000/60/60
    Q_24=Q_24*1000/60/60
    # ax1=ax.twinx()
    l3=ax.plot(t_Q,Q_total,'b-',linewidth=1.0,label=r'Total injection rate')
    l4=ax.plot(t_20,Q_20,'k-',linewidth=1.5,label=r'Well #1')
    l5=ax.plot(t_23,Q_23,'g-',linewidth=1.5,label=r'Well #2')
    l6=ax.plot(t_24,Q_24,'r-',linewidth=1.5,label=r'Well #3')
    ind_sep_24=np.where(df_24.index==datetime.strptime('20130711','%Y%m%d'))[0][-1]
    # l6_line_sep=ax1.plot([df_24.index[ind_sep_24],df_24.index[ind_sep_24]],1.2*np.array([0,Q_total[ind_sep_24]]),'k--',linewidth=2.,label=r'Time of RK24 injectivity decline')
    l6_line_sep=ax.plot([df_24.index[ind_sep_24],df_24.index[ind_sep_24]],1.3*np.array([0,Q_total[ind_sep_24]]),'k--',linewidth=2.,label=r'Time of Well #3 injectivity decline')
    ind_sep_23=np.where(df_23.index==datetime.strptime('20130908','%Y%m%d'))[0][-1]
    # l5_line_sep=ax1.plot([df_23.index[ind_sep_23],df_23.index[ind_sep_23]],1.3*np.array([0,Q_total[ind_sep_23]]),'k-.',linewidth=2.,label=r'Time of flow rate transferred to RK23 ')
    l5_line_sep=ax.plot([df_23.index[ind_sep_23],df_23.index[ind_sep_23]],1.4*np.array([0,Q_total[ind_sep_23]]),'k-.',linewidth=2.,label=r'Time of flow rate transferred to Well #2 ')
    ls=l6_line_sep+l5_line_sep+l3+l4+l5+l6
    lgs=[l.get_label() for l in ls]
    leg=ax.legend(ls,lgs,bbox_to_anchor=(0.,1.02,1.,.102),loc='lower left',ncol=3,mode='expand',borderaxespad=0.,fontsize=14)
    # leg=ax.legend(ls,lgs,loc='upper right')
    # renderer = fp.canvas.get_renderer()
    # shift = max([t.get_window_extent(renderer).width for t in leg.get_texts()])
    # for t in leg.get_texts():
    #     t.set_ha('right')## set the legend content right-alignment
    #     t.set_position((shift,0))

    ax.set_xlabel('Time (year)',fontsize=14)
    # ax.set_ylabel('Seismicity rate, $week^{-1}$',fontsize=14)
    # ax.set_xlim([min(t_Q)*0.99,max(t_Q)*1.01])
    # ax.set_ylim([min(rate_ti)*0.99,max(rate_ti)*1.1])
    ax.set_ylim([0,max(Q_total)*1.05])
    ax.set_ylabel('Flow rate ($kg/s$)',fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    # plt.setp(ax1.get_yticklabels(), fontsize=12)
    # plt.setp(ax1.get_xticklabels(), fontsize=12)

    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\Q_total\lambda_vs_T_vs_Q_new1227.png',dpi=500)

    # fig,ax=plt.subplots(1,1,figsize=(9,4))
    # # ax.plot(ti,Y_1,'k',linewidth=1.0,label=r'$Y_{01}$')
    # ax.plot(t_Dtj,Y_Dtj,'b--',linewidth=1.0,label=r'$Y_{\Delta{t_j}}$')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('label vector')
    # ax.legend(loc='best')

    




    '''
    select=FeatureSelector(n_jobs=0,ml_task='classification')
    Y_Dtj=pd.Series(Y_Dtj,index=range(len(Y_Dtj)))
    df_FM.index=Y_Dtj.index
    select.fit_transform(df_FM,Y_Dtj)
    # Nfts=10
    fts=select.features
    pvs=select.p_values
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
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/week, $\Delta t_i$={} weeks,$\Delta t_j$={} weeks'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_Q_Features_Lambda_th_{}_dti_{}w.png'.format(int(lambda_th),int(Dti)),dpi=500)
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
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/week, $\Delta t_i$={} weeks,$\Delta t_j$={} weeks'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_whp_Features_Lambda_th_{}_dti_{}w.png'.format(int(lambda_th),int(Dti)),dpi=500)
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
    plt.suptitle('RK20_Earthquake rate threhold $\lambda_{{th}}$ = {}/week, $\Delta t_i$={} weeks,$\Delta t_j$={} weeks'.format(lambda_th,Dti,Dtj_df),fontsize=14)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\RK20\RK20_QxP_Features_Lambda_th_{}_dti_{}w.png'.format(int(lambda_th),int(Dti)),dpi=500)


    '''


    plt.show()  







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

