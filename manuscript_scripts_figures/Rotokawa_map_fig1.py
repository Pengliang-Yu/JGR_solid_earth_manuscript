'''
Code for georeferencing an image (.png) and plot on top
'''
import pandas as pd  
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
#import shutil, os
textsize = 14.
f,ax=plt.subplots(1,1,figsize=(9,8))
# f = plt.figure(figsize=[9.5,7.5])
# ax = plt.axes([0.18,0.25,0.70,0.50])
path_base_image = 'rk_inj_zoom.png'
if path_base_image:
    img=mpimg.imread(path_base_image)
else:
    raise 'no path base image'
#element, lon, lat, elev
#img_c_bl, 176.198906, -38.619617, 0
#img_c_br, 176.210610, -38.619625, 0
#img_c_tl, 176.198833, -38.613191, 0
#img_c_tr, 176.210666, -38.613174, 0

# extent: floats (left, right, bottom, top)
ext_img =  176.198906, 176.210610, -38.619625, -38.613191
if ext_img:
    ext = ext_img 
else:
    raise 'no external (bound) values for image image'
# plot image
ax.imshow(img, extent = ext, alpha=0.0)
#plt.show()
if True: # plot extras 
    # plot wells
    # well 1
    wls_tracks = np.genfromtxt('well_RK20.csv', delimiter=',')
    # ax.plot(wls_tracks[0,1], wls_tracks[0,0], 'k*', markersize=10)
    ax.plot(wls_tracks[:,1], wls_tracks[:,0], 'r-', linewidth=3.0)##Well#1 is rk20, markersize=10,
    #well RK23
    rk23=np.genfromtxt('well_RK23.csv', delimiter=',')
    # ax.plot(rk23[0,1], rk23[0,0], 'b*', markersize=10)
    ax.plot(rk23[:,1], rk23[:,0], 'r-', linewidth=4.0)##well#2 is rk23,markersize=10,
    ##well Rk24
    rk24=np.genfromtxt('well_RK24.csv', delimiter=',')
    # ax.plot(rk24[0,1], rk24[0,0], 'g*', markersize=10)
    ax.plot(rk24[:,1], rk24[:,0], 'r-', linewidth=3.0)#well#3 is Rk24,markersize=10,
    # plot Faults
    # fault 1
    flt_track = np.genfromtxt('foult_IFF.csv', delimiter=',')
    ax.plot(flt_track[:,1], flt_track[:,0], 'k--', markersize=10,linewidth=2)
    ##fault CFF
    cff_track = np.genfromtxt('foult_CFF.csv', delimiter=',')
    ax.plot(cff_track[:,1], cff_track[:,0], 'k--', markersize=10,linewidth=2)


    ##plot seismicity event
    df=pd.read_csv(r'Z:\MLproject\event_information_noid.csv')
    date_all=df['eventTIME']
    lat_all=df['event_latitude']
    lon_all=df['event_longitude']
    mag_all=df['event_magnitude']
    # depth_all=df['event_depth']/1000.
    # distance_filter=np.where((lon_all>176.19)&(lon_all<176.22)&(lat_all>=-38.625)&(lat_all<-38.60)&(depth_all<6.0)&(mag_all>0.5))#&(lon_all>=176.15)&(lon_all<=176.25)
    distance_filter=np.where((lon_all>176.19)&(lon_all<176.22)&(lat_all>=-38.625)&(lat_all<-38.60)&(mag_all>0.5))#&(lon_all>=176.15)&(lon_all<=176.25)
    date=date_all[distance_filter[0]]
    lat=lat_all[distance_filter[0]]
    lon=lon_all[distance_filter[0]]
    mag=mag_all[distance_filter[0]]
    t0=datetime.strptime('20120101','%Y%m%d')
    times=[]
    for each_date in date:
        t=datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ")
        dt=t-t0
        times.append(dt.total_seconds()/(3600*24*365.25)+2012)
    s=(np.max(mag)-mag)/(np.max(mag)-np.min(mag))
    s=s*15+1
    # coolwarm=matplotlib.cm.get_cmap('coolwarm_r')
    from matplotlib import cm
    CS=ax.scatter(lon,lat,s=s,c=times,cmap=cm.get_cmap('viridis'))
    # ax.set_aspect('equal',adjustable='box')
    
# ax.legend(loc=1, prop={'size': 10})	
# leg=ax.legend(bbox_to_anchor=(0.,-0.15,1.05,0.),loc='lower left',ncol=5,mode='expand',borderaxespad=0.,fontsize=14)
ax.set_xlabel('Longitude [°]', size = textsize)
ax.set_ylabel('Latitude [°]', size = textsize)
#
# ax.set_title('Map of Rotokawa injection zone', size = textsize)

ax.set_xlim([176.195,176.22])
ax.set_ylim([-38.625,-38.60])
plt.setp(ax.get_xticklabels(), Fontsize=12)
plt.setp(ax.get_yticklabels(), Fontsize=12)
ax.ticklabel_format(useOffset=False)
mean_lat=np.mean(lat)
f=1.0/np.cos(mean_lat*np.pi/180)
ax.set_aspect(f,adjustable='box')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
# plt.colorbar(im, cax=cax)
cbar=plt.colorbar(CS,ax=ax,cax=cax,ticks=[2012,2013,2014,2015,2016])
cbar.ax.tick_params(labelsize=12)
# from matplotlib.ticker import FormatStrFormatter
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()

#
# save 
# plt.savefig('map_update.png', dpi=300, edgecolor='k',format='png')	
plt.savefig('Rotokawa0531.png',dpi=600)
plt.show()
#plt.close(f)