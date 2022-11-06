import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import chi2
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
from pandas._libs.tslibs.timestamps import Timestamp
from copy import copy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from scipy.stats import chi2
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

def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))


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
    ##decluster
    # ind_at_3big=np.where(np.array(mag)>=3)[0]##the index position of event larger than 2.5
    # time_at_3big=np.array(times)[ind_at_3big]
    # week_num=1.### decluster 1 weeks events after big events
    # weeks_dt=timedelta(days=7*week_num)
    # for j in range(len(time_at_3big)):
    #     times_decluster=[times[i] for i in range(len(times)) if(times[i]<=time_at_3big[j] or times[i]>=time_at_3big[j]+weeks_dt)]
    #     times=times_decluster
    # df_FM=pd.read_csv(r'Z:\MLproject\whakaari-master\features\test_features_Q_total_0726.csv',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    # ti=[datetimeify(i) for i in df_FM.index]
    df_Q=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Qtotal_0726_int_qdt.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ind=np.where(df_Q.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    t_Q=df_Q.index[:ind]
    # print('hi')
    # t_Q=[datetimeify(i) for i in df_Q.index[:26089]] ##26089 is the time of 2015-01-01 00:00:0
    # Q_total=df_Q[['Q_total(t/h)']][:ind]
    df_20=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk20_0622_int_qdt.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    df_23=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk23_0726_int_qdt.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    df_24=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_Rk24_0722_int_qdt.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    ind20=np.where(df_20.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    ind23=np.where(df_23.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    ind24=np.where(df_24.index<=datetime.strptime('20150101','%Y%m%d'))[0][-1]
    t_20=df_20.index[:ind20]
    t_23=df_23.index[:ind23]
    t_24=df_24.index[:ind24]

    df_Q['Q_total(t/h)_norm']=(df_Q['Q_total(t/h)']-np.mean(df_Q['Q_total(t/h)']))/np.std(df_Q['Q_total(t/h)'])
    df_Q['Integ_qdt(t)_norm']=(df_Q['Integ_qdt(t)']-np.mean(df_Q['Integ_qdt(t)']))/np.std(df_Q['Integ_qdt(t)'])
    df_20['Flow_rate(t/h)_norm']=(df_20['Flow_rate(t/h)']-np.mean(df_20['Flow_rate(t/h)']))/np.std(df_20['Flow_rate(t/h)'])
    df_20['Integ_qdt(t)_norm']=(df_20['Integ_qdt(t)']-np.mean(df_20['Integ_qdt(t)']))/np.std(df_20['Integ_qdt(t)'])
    df_23['Flow_rate(t/h)_norm']=(df_23['Flow_rate(t/h)']-np.mean(df_23['Flow_rate(t/h)']))/np.std(df_23['Flow_rate(t/h)'])
    df_23['Integ_qdt(t)_norm']=(df_23['Integ_qdt(t)']-np.mean(df_23['Integ_qdt(t)']))/np.std(df_23['Integ_qdt(t)'])
    df_24['Flow_rate(t/h)_norm']=(df_24['Flow_rate(t/h)']-np.mean(df_24['Flow_rate(t/h)']))/np.std(df_24['Flow_rate(t/h)'])
    df_24['Integ_qdt(t)_norm']=(df_24['Integ_qdt(t)']-np.mean(df_24['Integ_qdt(t)']))/np.std(df_24['Integ_qdt(t)'])





    
    # Q_20=df_20[['Flow_rate(t/h)']][:ind20]
    # Q_23=df_23[['Flow_rate(t/h)']][:ind23]
    # Q_24=df_24[['Flow_rate(t/h)']][:ind24]
    ##as df_20 start from 2012-01-10, other wells starts from 2012-01-01, then make sure all data start from 2012-01-10
    ind_Q_total_start=np.where(df_Q.index<=datetime.strptime('20120110','%Y%m%d'))[0][-1]
    ind_RK20_start=np.where(df_20.index<=datetime.strptime('20120110','%Y%m%d'))[0][-1]
    ind_RK23_start=np.where(df_23.index<=datetime.strptime('20120110','%Y%m%d'))[0][-1]
    ind_RK24_start=np.where(df_24.index<=datetime.strptime('20120110','%Y%m%d'))[0][-1]
    Q_total=df_Q[['Q_total(t/h)_norm','Integ_qdt(t)_norm']][ind_Q_total_start:ind]
    Q_20=df_20[['Flow_rate(t/h)_norm','Integ_qdt(t)_norm']][ind_RK20_start:ind20]
    Q_23=df_23[['Flow_rate(t/h)_norm','Integ_qdt(t)_norm']][ind_RK23_start:ind23]
    Q_24=df_24[['Flow_rate(t/h)_norm','Integ_qdt(t)_norm']][ind_RK24_start:ind24]
    t_20=Q_20.index
    t_23=Q_23.index
    t_24=Q_24.index

    ti = copy(t_20)
    # df_eq=pd.read_csv(r'Z:\MLproject\whakaari-master\data\earthquake_rate.dat',index_col=0)
    # rate_eq=df_eq['lambda/week']
    Dti=1.#week
    Dti_df=timedelta(days=7*Dti)
    rate_ti = alt(times, end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/Dti

    def norm_asym(m,s,x):
        return (x>m)*np.exp(-0.5*(x-m)**2/s**2)/s/np.sqrt(2*np.pi)*2
    def KDE_asym(points, bandwidth, x):
        '''
            scale accounts for # eqs, conversion to weekly
        '''
        scale = np.sum((points>x[0])&(points<x[-1]))*7
        return np.mean([norm_asym(pt,bandwidth,x) for pt in points],axis=0)*scale

    ts = np.array([(t-ti[0]).total_seconds()/86400 for t in times])
    tv = np.array([(t-ti[0]).total_seconds()/86400 for t in ti])
    # rate_ti = KDE_asym(ts,3.5,tv)
    df_alldata=Q_total
    df_alldata.rename(columns={'Integ_qdt(t)_norm':'Q_total_Integ(t)_norm'},inplace=True)
    df_alldata['Q_RK20(t/h)_norm']=Q_20['Flow_rate(t/h)_norm']
    df_alldata['Q_RK20_Integ(t)_norm']=Q_20['Integ_qdt(t)_norm']
    df_alldata['Q_RK23(t/h)_norm']=Q_23['Flow_rate(t/h)_norm']
    df_alldata['Q_RK23_Integ(t)_norm']=Q_23['Integ_qdt(t)_norm']
    df_alldata['Q_RK24(t/h)_norm']=Q_24['Flow_rate(t/h)_norm']
    df_alldata['Q_RK24_Integ(t)_norm']=Q_24['Integ_qdt(t)_norm']
    df_alldata['eq_rate(/week)']=rate_ti
    
    ind_part1=datetime.strptime('20130710','%Y%m%d')
    ind_p1=np.where(df_alldata.index<=ind_part1)[0][-1]
    ind_part2=datetime.strptime('20130909','%Y%m%d')
    ind_p2=np.where(df_alldata.index>=ind_part2)[0][0]
    df_alldata_part1=df_alldata.iloc[ df_alldata.index<=ind_part1,:]
    df_alldata_part2=df_alldata.iloc[ (df_alldata.index>=ind_part2),:]

    ### 1-2 weeks injection rate lag compared with lambda rate
    # week_num=1
    day_num=5
    weeks_dt=timedelta(days=day_num)
    rate_ti_start=datetime.strptime('20130909','%Y%m%d')+weeks_dt
    rate_ti_start_index=np.where(df_alldata_part2.index==rate_ti_start)[0][0]
    df_rate=df_alldata_part2['eq_rate(/week)'][rate_ti_start_index:]
    Y_data=df_rate
    length=len(df_rate)
    X_data=df_alldata_part2.drop('eq_rate(/week)',axis=1)
    X_data=X_data.iloc[:length,:]


    # df_alldata_parts=pd.concat([df_alldata_part1,df_alldata_part2])
    # groups=[1 if i<= ind_p1 else 2 for i in range(len(df_alldata_parts.index))]
    # df_alldata_parts['Groups']=groups
    # df_alldata_parts['Time']=df_alldata_parts.index
    # df_alldata_parts.index=range(len(df_alldata_parts))

    ###fit part 1 data to multi-variate linear regression model ##Method 1: 
    LR_part1_method1 = LinearRegression(positive=True)
    # X_data=df_alldata_part2.drop('eq_rate(/week)',axis=1)
    X_data=X_data.drop(['Q_total(t/h)_norm','Q_total_Integ(t)_norm'],axis=1)
    # Y_data=df_alldata_part2['eq_rate(/week)']
    # Y_data=np.log10(Y_data)
    LR_part1_method1.fit(X_data,Y_data)
    

    R_2_model=LR_part1_method1.score(X_data,Y_data)## R^2 of the model
    print('R^2 is:',R_2_model)
    ad_R2=1-(1-R_2_model)*(len(Y_data)-1)/(len(Y_data)-X_data.shape[1]-1)
    print('adjusted R^2 is:',ad_R2)
    coeffs_model=LR_part1_method1.coef_  ##coefficient of the each varaible
    print('the coefficients of the model are',coeffs_model)
    intercept_model=LR_part1_method1.intercept_ ## intercept of the fit model
    print('the intercept of the model is',intercept_model)
    Y_model=LR_part1_method1.predict(X_data)
    r2score=r2_score(Y_data,Y_model)
    print('r2 score is:',r2score)
    MSE=mean_squared_error(Y_data,Y_model)
    print('mean squared error is:',MSE)
    print('root ean squared error is',np.sqrt(MSE))
    '''
    ##calculate the LLK
    # times_part1=[i for i in times if (i >=datetime.strptime('20120110','%Y%m%d') and i <=datetime.strptime('20130710','%Y%m%d'))]
    # times_part1_num=[(i-t0).total_seconds()/(3600*24*365.25)+2012 for i in times_part1]
    times_part2=[i for i in times if (i >=datetime.strptime('20130909','%Y%m%d') and i <=datetime.strptime('20150101','%Y%m%d'))]
    times_part2_num=[(i-t0).total_seconds()/(3600*24*365.25)+2012 for i in times_part2]
    df_alldata_part2_index_num=[(i-t0).total_seconds()/(3600*24*365.25)+2012 for i in df_alldata_part2.index]
    
    ri=np.interp(times_part2_num,df_alldata_part2_index_num,Y_model)# interpolate earthquake rate at earthquake times
    ri=np.interp(times_part2_num,df_alldata_part2_index_num,Y_model/7*365.25)##convert to year unit rate
    ri_up=[max(rii, 1.e-8) for rii in ri]
    i1 = np.argmin(abs(np.array(df_alldata_part2_index_num)-times_part2_num[-1]))#find the index of last event
    crt = np.sum(Y_model[:i1])*(df_alldata_part2_index_num[1]-df_alldata_part2_index_num[0])##compute cumulative rate to last event(rectangular rule)
    LLK_part2=np.sum(np.log(ri_up))-crt
    print('the LLK of part 2 for a year rate is',LLK_part2)
    '''
    '''
    ##method1: calcuate the pvalues for each coefficients and the confidence levels of the coefficients
    ## got from this link: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    from scipy import stats

    # lm = LinearRegression()
    # lm.fit(X,y)
    params = np.append(LR_part1_method1.intercept_,LR_part1_method1.coef_)
    predictions = LR_part1_method1.predict(X_data)

    newX = pd.DataFrame({"Constant":np.ones(len(X_data))}).join(pd.DataFrame(np.array(X_data)))
    MSE = (sum((Y_data-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    # p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
    # p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(X[0])))) for i in ts_b]
    p_values =[2.*(1.-stats.t.cdf(np.abs(i),(len(newX)-len(newX.columns)-1.))) for i in ts_b]

    # sd_b = np.round(sd_b,10)
    # ts_b = np.round(ts_b,4)
    # p_values = np.round(p_values,10)
    # params = np.round(params,10)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["P_value"] = [params,sd_b,ts_b,p_values]
    print(myDF3)
    '''
    ##method 2 to calcuate the pvalues for each coefficients and the confidence levels of the coefficients
    ##got from: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    beta_hat = [LR_part1_method1.intercept_] + LR_part1_method1.coef_.tolist()
    # compute the p-values
    from scipy.stats import t
    # add ones column
    X1 = np.column_stack((np.ones(len(X_data)), X_data))
    # standard deviation of the noise.
    sigma_hat = np.sqrt(np.sum(np.square(Y_data - X1@beta_hat)) / (len(X_data) - X1.shape[1]))
    # estimate the covariance matrix for beta 
    beta_cov = np.linalg.inv(X1.T@X1)
    ##calculate the variance
    VAR_b=sigma_hat**2*(beta_cov.diagonal())
    ##caculate the standard error
    SE=np.sqrt(VAR_b)
    # the t-test statistic for each variable from the formula from above figure
    t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
    # compute 2-sided p-values.
    p_vals = t.sf(np.abs(t_vals), len(X_data)-X1.shape[1])*2 
    # print(t_vals)
    # print(p_vals)
    ## confidence levels of coefficients-95%: https://www.econometrics-with-r.org/5-2-cifrc.html; https://stattrek.com/online-calculator/t-distribution.aspx;http://www.stat.yale.edu/Courses/1997-98/101/confint.htm;https://en.wikipedia.org/wiki/Confidence_interval
    confi_lower=beta_hat-1.96*SE
    confi_upper=beta_hat+1.96*SE
    err=1.96*SE
    columns_all=list(zip(beta_hat,err,SE,t_vals,p_vals,confi_lower,confi_upper))
    columns_name=["Coefficients","1.96SE","Standard Errors","t values","P_value",'0.025_confi.','0.975_confi']
    index_list=['intercept']
    index_list.extend(X_data.columns)
    mysummary = pd.DataFrame(columns_all,columns=columns_name,index=index_list)
    
    
    print(mysummary)
    ##calculate the AIC 
    k=len(np.where(mysummary['Coefficients']!=0)[0])##number of non-zero coefficients
    N=len(Y_data)
    sigma=np.sqrt(np.mean(Y_data))
    # columns_name_out=['Y_data','Y_model']
    # columns_all_out=list(zip(Y_data,Y_model))
    # out_file = pd.DataFrame(columns_all_out,columns=columns_name_out,index=Y_data.index)
    # out_file .to_csv(r'Z:\MLproject\whakaari-master\data\AIC_normalized_part2.dat', index=True)
    AIC=2*k+N*np.log(2*np.pi)+2*N*np.log(sigma)+1/(sigma**2)*np.sum((Y_model-Y_data)**2)
    print('AIC is:',AIC)
    ##method 2: using the polynominal
    # model_poly = Pipeline([('poly', PolynomialFeatures(degree=1)),('linear', LinearRegression(fit_intercept=True))])
    # model_p=model_poly.fit(X_data,Y_data)
    # model_p.named_steps['linear'].coef_
    # model_p.named_steps['linear'].intercept_
    # Y_model_poly=model_p.predict(X_data)
    # model_p.score(X_data,Y_data)



    '''
    ##method3 using the statsmodel module
    # poly=PolynomialFeatures(degree=1)
    # X_=poly.fit_transform(X_data)
    X = sm.add_constant(X_data)
    sm_model=sm.OLS(Y_data,X).fit()
    # sm_model.params
    Y_sm_model=sm_model.predict(X)
    r2score_sm_model=r2_score(Y_data,Y_sm_model)
    MSE_sm_model=mean_squared_error(Y_data,Y_sm_model)
    print(sm_model.summary())
    
    
    ##method#3 improvement: ignore the two terms with p value larger than 0.05
    X_data_s=df_alldata_part1.drop(['eq_rate(/week)','Q_total_Integ(t)_norm','Q_RK23_Integ(t)_norm'],axis=1)
    X_s = sm.add_constant(X_data_s)
    sm_model_s=sm.OLS(Y_data,X_s).fit()
    Y_sm_model_s=sm_model_s.predict(X_s)
    print(sm_model_s.summary())
    '''
    # Y_model=10**Y_model
    lower_Y_model, upper_Y_model=chi2.interval(0.90,Y_model)
    f,ax=plt.subplots(1,1,figsize=(8,3.5))
    ax.plot(Y_data.index,Y_data,'k-',label='Observed post-Sept. 2013 data')
    # ax.plot(df_alldata_part1.index,Y_model,color='r',linestyle='-',marker='o',label='Fitted data-sklearn_linear')
    # ax.plot(Y_data.index,Y_model,'r-',label='Regression model_{}week lag'.format(week_num))
    ax.plot(Y_data.index,Y_model,'r-',label='Post-Sept. 2013 regression model')
    ax.fill_between(Y_data.index,lower_Y_model,upper_Y_model,color='r',alpha=0.2,label='90% confidence intervals')
    # ax.plot(df_alldata_part2.index,Y_sm_model,'b-',label='Fitted data-statsmodels_OLS')
    # ax.plot(df_alldata_part1.index,Y_sm_model_s,'g-',label='Fitted data-statsmodels_OLS_improved')
    ax.set_xlabel('Time (year)',fontsize=12)
    ax.set_ylabel('Microseismicity rate ($week^{-1}$)',fontsize=12)
    plt.setp(ax.get_xticklabels(), Fontsize=10)
    plt.setp(ax.get_yticklabels(), Fontsize=10)
    ax.set_xlim([Y_data.index[0],Y_data.index[-1]])
    ax.legend(loc='best',fontsize=12)
    ax.text(0.95, 0.05, 'b',fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\regression_fitting\linear_fitting_Qall_part2_norm_NNLS_{}Dlag_JA_1026.png'.format(day_num),dpi=500)
    plt.show()  
    
    # fp,ax=plt.subplots(1,1,figsize=(17,8))
    # l1=ax.plot(ti,rate_ti,'r',linewidth=1.0,label='Seismicity rate')
    # l2=ax.plot([min(ti),max(ti)],[lambda_th,lambda_th],'r--',linewidth=1.0,label=r'$\lambda_{th}$')
    
    # ax1=ax.twinx()
    # l3=ax1.plot(t_Q,Q_total,'b-',linewidth=1.0,label=r'Total injection rate')
    # l4=ax1.plot(t_20,Q_20,'k-',linewidth=1.0,label=r'RK20')
    # l5=ax1.plot(t_23,Q_23,'g-',linewidth=1.0,label=r'Rk23')
    # l6=ax1.plot(t_24,Q_24,'c-',linewidth=1.0,label=r'RK24')

    # ind_sep_24=np.where(df_24.index==datetime.strptime('20130711','%Y%m%d'))[0][-1]
    # l6_line_sep=ax1.plot([df_24.index[ind_sep_24],df_24.index[ind_sep_24]],1.2*np.array([0,Q_total[ind_sep_24]]),'k--',linewidth=1.5,label=r'Time of RK24 injectivity decline')
    # ind_sep_23=np.where(df_23.index==datetime.strptime('20130908','%Y%m%d'))[0][-1]
    # l5_line_sep=ax1.plot([df_23.index[ind_sep_23],df_23.index[ind_sep_23]],1.3*np.array([0,Q_total[ind_sep_23]]),'k-.',linewidth=1.5,label=r'Time of flow rate transferred to RK23 ')
    # ls=l6_line_sep+l5_line_sep+l1+l2+l3+l4+l5+l6
    # lgs=[l.get_label() for l in ls]
    # leg=ax.legend(ls,lgs,bbox_to_anchor=(0.,1.02,1.,.102),loc='lower left',ncol=4,mode='expand',borderaxespad=0.)
    # ax.set_xlabel('Time(year)')
    # ax.set_ylabel('Seismicity rate, $week^{-1}$')
    # # ax.set_xlim([min(t_Q)*0.99,max(t_Q)*1.01])
    # ax.set_ylim([min(rate_ti)*0.99,max(rate_ti)*1.1])
    # ax1.set_ylim([0,max(Q_total)*1.05])
    # ax1.set_ylabel('Flow rate, $t/h$')
    
    # plt.tight_layout()
    # 

    # plt.show()  







if __name__ == "__main__":
    
    main()
