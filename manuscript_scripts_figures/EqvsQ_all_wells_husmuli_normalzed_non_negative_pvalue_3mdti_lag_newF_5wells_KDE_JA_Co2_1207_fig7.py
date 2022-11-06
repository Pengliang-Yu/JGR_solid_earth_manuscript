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
    
    Dti_df=timedelta(days=7*Dti)###1 week Dti
    EQs=list()
    rate_ti2=list()
    for i in ti_FM:
        EQs.append(len([j for j in times if j<=i and j>(i-Dti_df)]))
    
    rate_ti=[EQ/Dti for EQ in EQs]   ## unit in week
    return rate_ti
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

    df_Q=pd.read_csv(r'Z:\MLproject\whakaari-master\data\tremor_data_husmuli_hour_unit_norm2015_CO2.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    t_all=df_Q.index
    ti = copy(t_all)
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
    # rate_ti = KDE_asym(ts,3.5*4*3,tv)
    Dti=4*3
    # rate_ti=cal_lambda(Dti,t_all,times)
    Dti_df=timedelta(days=7*Dti)###1 week Dti
    rate_ti = alt(np.array(times), end=ti[-1], window=Dti_df, start=ti[0], step=ti[-1]-ti[-2])/(Dti)
    rate_ti=[rr if rr >0 else 0.167 for rr in rate_ti]
    df_alldata=df_Q
    df_alldata['eq_rate(/week)']=rate_ti
    ##### 1-2 weeks injection rate lag compared with lambda rate
    # week_num=2
    day_num=0
    weeks_dt=timedelta(days=day_num)
    rate_ti_start=df_alldata.index[0]+weeks_dt
    rate_ti_start_index=np.where(df_alldata.index==rate_ti_start)[0][0]
    df_rate=df_alldata['eq_rate(/week)'][rate_ti_start_index:]
    Y_data=df_rate
    length=len(df_rate)
    X_data=df_alldata.drop('eq_rate(/week)',axis=1)
    # X_data=X_data.drop(['norm_HN14_flow(t/h)','norm_Integ_HN14(t)'],axis=1)
    X_data=X_data.iloc[:length,:]
    ###fit part 1 data to multi-variate linear regression model ##Method 1: 
    LR_method1 = LinearRegression(positive=True)
    # X_data=df_alldata.drop('eq_rate(/week)',axis=1)
    X_data=X_data.drop(['norm_Q_total(t/h)','norm_Integ_Q_total(t)'],axis=1)
    # Y_data=df_alldata['eq_rate(/week)']
    Y_data=np.log10(Y_data)
    LR_method1.fit(X_data,Y_data)

    R_2_model=LR_method1.score(X_data,Y_data)## R^2 of the model
    print('R^2 is:',R_2_model)
    coeffs_model=LR_method1.coef_  ##coefficient of the each varaible
    print('the coefficients of the model are',coeffs_model)
    intercept_model=LR_method1.intercept_ ## intercept of the fit model
    print('the intercept of the model is',intercept_model)
    Y_model=LR_method1.predict(X_data)
    r2score=r2_score(Y_data,Y_model)
    print('r2 score is:',r2score)
    MSE=mean_squared_error(10**Y_data,10**Y_model)
    print('mean squared error is:',MSE)
    print('root ean squared error is',np.sqrt(MSE))
    ##calculate the LLK
    # times_part1=[i for i in times if (i >=datetime.strptime('20120110','%Y%m%d') and i <=datetime.strptime('20130710','%Y%m%d'))]
    # times_part1_num=[(i-t0).total_seconds()/(3600*24*365.25)+2012 for i in times_part1]
    # df_alldata_part1_index_num=[(i-t0).total_seconds()/(3600*24*365.25)+2012 for i in df_alldata_part1.index]
    
    # ri=np.interp(times_part1_num,df_alldata_part1_index_num,Y_model)# interpolate earthquake rate at earthquake times
    # ri=np.interp(times_part1_num,df_alldata_part1_index_num,Y_model/7*365.25)
    # ri_up=[max(rii, 1.e-8) for rii in ri]
    # i1 = np.argmin(abs(np.array(df_alldata_part1_index_num)-times_part1_num[-1]))#find the index of last event
    # crt = np.sum(Y_model[:i1])*(df_alldata_part1_index_num[1]-df_alldata_part1_index_num[0])##compute cumulative rate to last event(rectangular rule)
    # LLK_part1=np.sum(np.log(ri_up))-crt
    # print('the LLK of part 1 for a year rate is',LLK_part1)
    
    ##method 2 to calcuate the pvalues for each coefficients and the confidence levels of the coefficients
    ##got from: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    beta_hat = [LR_method1.intercept_] + LR_method1.coef_.tolist()
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
    # Y_model=[yy if yy>0 else 1e-8 for yy in Y_model]
    Y_model=10**Y_model
    lower_Y_model, upper_Y_model=chi2.interval(0.90,Y_model)
    
    f,ax=plt.subplots(1,1,figsize=(6,4))
    ax.plot(Y_data.index,10**Y_data,'k-',label='Observed data')
    # ax.plot(Y_data.index,Y_data,'k-',label='Observed data')
    # ax.plot(df_alldata_part1.index,Y_model,color='r',linestyle='-',marker='o',label='Fitted data-sklearn_linear')
    # ax.plot(Y_data.index,Y_model,'r-',label='Regression model_{}week lag'.format(week_num))
    ax.plot(Y_data.index,Y_model,'r-',label='Regression model')
    ax.fill_between(Y_data.index,lower_Y_model,upper_Y_model,color='r',alpha=0.15,label='90% confidence intervals of regession model')
    # ax.plot(df_alldata_part1.index,Y_sm_model,'b-',label='Fitted data-statsmodels_OLS')
    # ax.plot(df_alldata_part1.index,Y_sm_model_s,'g-',label='Fitted data-statsmodels_OLS_improved')
    ax.set_xlabel('Time (year)',fontsize=12)
    ax.set_ylabel('Microseismicity rate ($week^{-1}$)',fontsize=12)
    plt.setp(ax.get_xticklabels(), Fontsize=10)
    ax.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.savefig(r'Z:\MLproject\whakaari-master\Husmuli_plots\regression_fitting\linear_husmuli_norm_NNLS_1w_decluster_2015_KDE_{}dLag_newF_5Well_{}m_CO2_train_JA1226.png'.format(day_num,Dti/4),dpi=500)
    plt.show()  
    






if __name__ == "__main__":
    
    main()
