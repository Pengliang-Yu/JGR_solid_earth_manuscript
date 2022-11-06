import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel
from whakaari import datetimeify
from datetime import timedelta, datetime
import numpy as np
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

    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    data_streams = ['norm_HN12_flow(t/h)','diff_norm_HN12_flow(t/h)',
        'norm_Integ_HN12(t)']##RK20 well injection data
    fm = ForecastModel(ti='2015-01-01', tf='2020-12-07 07:00:00', window=2., overlap=0.75, 
            look_forward=14., data_streams=data_streams, root='test')
    t0=datetimeify('2015-01-03')## end of first time window, if I use 2 days as a timedow, I should the end of first window should be 2012-01-12
    t1=datetimeify('2020-12-07 07:00:00')
    Feature_matrix=fm._extract_features(t0,t1)
    # print(Feature_matrix)

if __name__ == "__main__":
    #forecast_dec2019()
    main()
# tf=t1
# ti=t0
# dt=timedelta(seconds=600)
# window=2
# iw=int(window*6*24)
# io=int(0.75*iw)
# Nw = int(np.floor(((tf-ti)/dt)/(iw-io)))
# # Nw=ForecastModel.
# df,wd=fm._construct_windows(Nw=Nw,ti=t0)
# print(df)
# from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
# cfd=ComprehensiveFCParameters()##generate 64 column headers
# print(cdf)