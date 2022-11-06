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
f,ax=plt.subplots(1,1,figsize=(6.5,6.5))
print('hi')
# f = plt.figure(figsize=[9.5,7.5])
# ax = plt.axes([0.18,0.25,0.70,0.50])
# path_base_image = 'rk_inj_zoom.png'
# if path_base_image:
#     img=mpimg.imread(path_base_image)
# else:
#     raise 'no path base image'
#element, lon, lat, elev
#img_c_bl, 176.198906, -38.619617, 0
#img_c_br, 176.210610, -38.619625, 0
#img_c_tl, 176.198833, -38.613191, 0
#img_c_tr, 176.210666, -38.613174, 0

# extent: floats (left, right, bottom, top)
# ext_img =  176.198906, 176.210610, -38.619625, -38.613191
# if ext_img:
#     ext = ext_img 
# else:
#     raise 'no external (bound) values for image image'
# # plot image
# ax.imshow(img, extent = ext, alpha=0.0)
# #plt.show()
if True: # plot extras 
    '''
    # plot wells
    HN09= np.genfromtxt('HN09_new.csv', delimiter=',')
    # ax.plot(HN09[0,1], HN09[0,0], 'k*', markersize=10)
    ax.plot(HN09[:-3,1], HN09[:-3,0], 'r-', linewidth=2.5)##Well#1 is rk20
    # well 1
    HN12= np.genfromtxt('HN12.csv', delimiter=',')
    # ax.plot(HN12[0,1], HN12[0,0], 'b*', markersize=10)
    ax.plot(HN12[:,1], HN12[:,0], 'r-', linewidth=2.5)##Well#1 is rk20
    #well RK23
    HN14= np.genfromtxt('HN14.csv', delimiter=',')
    # ax.plot(HN14[0,1], HN14[0,0], 'r*', markersize=10)
    ax.plot(HN14[:,1], HN14[:,0], 'r-', linewidth=2.5)##Well#1 is rk20

    HN16= np.genfromtxt('HN16.csv', delimiter=',')
    # ax.plot(HN16[0,1], HN16[0,0], 'g*', markersize=10)
    ax.plot(HN16[:,1], HN16[:,0], 'r-', linewidth=2.5)##Well#1 is rk20
    HN17= np.genfromtxt('HN17.csv', delimiter=',')
    # ax.plot(HN17[0,1], HN17[0,0], 'y*', markersize=10)
    ax.plot(HN17[:,1], HN17[:,0], 'r-', linewidth=2.5)##Well#1 is rk20,, label = 'HN17'
    # plot Faults
    # fault 1
    flt_track = np.genfromtxt('fault1.csv', delimiter=',')
    ax.plot(flt_track[:,1], flt_track[:,0], 'k--', linewidth=2)
    ##fault CFF
    f2_track = np.genfromtxt('fault2.csv', delimiter=',')
    ax.plot(f2_track[:,1], f2_track[:,0], 'k--', linewidth=2)
    f3_track = np.genfromtxt('fault3.csv', delimiter=',')
    ax.plot(f3_track[:,1], f3_track[:,0], 'k--', linewidth=2)
    '''
    ##plot seismicity event
    df=pd.read_csv(r'Z:\MLproject\whakaari-master\data\Husmuli_earthquake_events.dat',index_col=0, parse_dates=[0,], infer_datetime_format=True)
    date_all=df.index
    mag_all=df['magnitude']
    lat_all=df['latitude']
    lon_all=df['longitude']
    t0=datetime.strptime('20120101','%Y%m%d')
    date_filter=np.where((mag_all>0.2)&(date_all>=t0))## filter data to magnitude completeness
    date=date_all[date_filter[0]]
    mag=mag_all[date_filter[0]]

    t0=datetime.strptime('20120101','%Y%m%d')
    times=[]
    for each_date in date:
        # t=datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ")
        dt=each_date-t0
        times.append(dt.total_seconds()/(3600*24*365.25)+2012)
    # times=list(date)
    lat=lat_all[date_filter[0]]
    lon=lon_all[date_filter[0]]
    mag=mag_all[date_filter[0]]
    # t0=datetime.strptime('20120101','%Y%m%d')
    # times=[]
    # for each_date in date:
    #     t=datetime.strptime(each_date,"%Y-%m-%dT%H:%M:%S.%fZ")
    #     dt=t-t0
    #     times.append(dt.total_seconds()/(3600*24*365.25)+2012)
    s=(np.max(mag)-mag)/(np.max(mag)-np.min(mag))
    s=s*15+1
    from shapely.geometry import Point
    import geopandas as gpd
    from geopandas import GeoDataFrame
    geometry=[Point(xy) for xy in zip(lon, lat)]
    gdf = GeoDataFrame(geometry=geometry,crs='EPSG:4326')
    # gdf_4326=gdf.to_crs("EPSG:4326")
    gdf.plot(ax=ax,marker='o',c=times,markersize=s)





    '''
    from matplotlib import cm
    CS=ax.scatter(lon,lat,s=s,c=times,cmap=cm.get_cmap('viridis'))
    '''
    # cbar=plt.colorbar(CS,ax=ax)
    # cbar.ax.tick_params(labelsize=12)
    # ax.set_aspect('equal',adjustable='box')

plt.show()
'''    
# ax.legend(loc=1, prop={'size': 10})	
# leg=ax.legend(bbox_to_anchor=(0.,-0.2,1.,0.),loc='lower left',ncol=3,mode='expand',borderaxespad=0.,fontsize=12)
ax.set_xlabel('Longitude [°]', size = textsize)
ax.set_ylabel('Latitude [°]', size = textsize)
#
# ax.set_title('Map of Rotokawa injection zone', size = textsize)
ax.set_ylim([64.04,64.07])
ax.set_xlim([-21.425,-21.37])
# ax.set_aspect(1./ax.get_data_ratio())
f=1.0/np.cos(60*np.pi/180)
# ax.set_aspect('equal', adjustable='box')
ax.set_aspect(f,adjustable='box')
plt.setp(ax.get_xticklabels(), Fontsize=12)
plt.setp(ax.get_yticklabels(), Fontsize=12)
ax.ticklabel_format(useOffset=False)


from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="3%", pad=0.05)

cbar=plt.colorbar(CS, ax=ax,fraction=0.047*f*0.5)
# cbar=plt.colorbar(CS,ax=ax,cax=cax,fraction=0.047*f)#,ticks=[2015,2016,2017,2018,2019,2020]
cbar.ax.tick_params(labelsize=12)
# from matplotlib.ticker import FormatStrFormatter
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()

#
# save 
# plt.savefig('map_update.png', dpi=300, edgecolor='k',format='png')	
# plt.savefig('husmuli0531.png',dpi=600)
plt.show()
#plt.close(f)
'''