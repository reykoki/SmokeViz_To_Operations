import sys

from pyorbital import astronomy
import numpy as np
import pyproj
from datetime import datetime
from datetime import timedelta
import cartopy.crs as ccrs

import os
import glob
from datetime import datetime
import s3fs
import pytz
from suntime import Sun
from datetime import timedelta
from helper_functions import *

data_dir = './data/'

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))

    return lcc_proj
def get_first_closest_file(band, fns, dt, sat_num):
    diff = timedelta(days=100)
    matching_band_fns = [s for s in fns if band in s]
    for fn in matching_band_fns:
        s_e = fn.split('_')[3:5]
        start = s_e[0]
        s_dt = datetime.strptime(start[1:-3], '%Y%j%H%M')
        s_dt = pytz.utc.localize(s_dt)
        if diff > abs(s_dt - dt):
            diff = abs(s_dt - dt)
            best_start = start
            best_end = s_e[1]
            best_fn = fn
    #fn_str = 'C{}_G{}_{}_{}'.format(band, sat_num, best_start, best_end)
    fn_str = 'G{}_{}_{}'.format(sat_num, best_start, best_end[:-3])
    return best_fn, fn_str

def get_additional_band_file(band, fn_str, fns):
    best_band_fn = [s for s in fns if band in s and fn_str in s]
    return best_band_fn[0]

def get_closest_file(fns, dt, sat_num, bands):
    use_fns = []
    band_init = 'C'+str(bands[0]).zfill(2)
    best_band_fn, fn_str = get_first_closest_file(band_init, fns, dt, sat_num)
    #use_fns.append(best_band_fn)
    for band in bands:
        band = 'C'+str(band).zfill(2)
        best_band_fn = get_additional_band_file(band, fn_str, fns)
        use_fns.append(best_band_fn)
    return use_fns

# get western and eastern most points lat/lon
def west_east_lat_lon(lat, lon, res, img_size): # img_size - number of pixels
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_w, lat_w = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_e, lat_e = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    return (lat_w, lon_w), (lat_e, lon_e)

# valid times are when the furthest lat/lon away from the sat has a solar zenith angle <90
def valid_times_from_szas(lat, lon, img_times):
    szas = np.zeros(len(img_times))
    for i, t in enumerate(img_times):
        szas[i] = astronomy.sun_zenith_angle(t, lon, lat)
    thresh = 90
    valid_indices = np.where(szas<thresh)[0]
    valid_times = []
    for idx in valid_indices:
        valid_times.append(img_times[idx])
    return valid_times

def get_best_sat_from_szas(w_coords, e_coords, img_times):
    mid_time = img_times[int(len(img_times)/2)]
    sza_west = astronomy.sun_zenith_angle(mid_time, w_coords[1], w_coords[0])
    sza_east = astronomy.sun_zenith_angle(mid_time, e_coords[1], e_coords[0])
    # closer to sunrise
    if sza_west >= sza_east:
        lat, lon = w_coords[0], w_coords[1]
        sat = '17'
    else:
        lat, lon = e_coords[0], e_coords[1]
        sat = '16'
    return sat, lat, lon

def get_sat(lat, lon, dt, res=1000, img_size=256):
    img_times = [dt]
    w_coords, e_coords = west_east_lat_lon(lat, lon, res, img_size)
    sat, lat, lon = get_best_sat_from_szas(w_coords, e_coords, img_times)
    valid_times = valid_times_from_szas(lat, lon, img_times)
    return sat

# check if its between sunrise on the west coast and sunset on hte east coast
# check that the sun is shining on entire CONUS
def check_sunrise_sunset(dt):
    west_lon = -124.8
    west_lat = 24.5
    east_lon = -71.1
    east_lat = 45.93
    east = Sun(east_lat, east_lon)
    west = Sun(west_lat, west_lon)
    sunset = east.get_sunset_time(dt)
    sunrise = west.get_sunrise_time(dt)
    print('for the datetime {}:\nsunrise is at: {}\nsunset is at: {}'.format(dt, sunrise, sunset))
    if sunrise > sunset:
        sunset = west.get_sunset_time(dt + timedelta(days=1))
    #if sunrise > dt or sunset < dt:
    #    raise ValueError('your request is before/after the sunrise/sunset for conus on {}'.format(dt) )
    else:
        sat_num = '16' # closer to sunset
    return sunrise, sunset

def get_filelist(dt, fs, lat, lon, sat_num, product, scope, bands):
    hr, dn, yr = get_dt_str(dt)
    full_filelist = fs.ls("noaa-goes{}/{}{}/{}/{}/{}/".format(sat_num, product, 'C', yr, dn, hr))
    if sat_num == '17' and len(full_filelist) == 0:
        if yr <= 2018:
            sat_num = '16'
            print("YOU WANTED 17 BUT ITS NOT LAUNCHED")
        elif yr >= 2022:
            sat_num = '18'
        full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, 'C', yr, dn, hr))
    use_fns = get_closest_file(full_filelist, dt, sat_num, bands)
    return use_fns

def download_goes(dt, lat=None, lon=None, sat_num='16', product='ABI-L1b-Rad', check_sun='True', scope='C', bands=list(range(1,4))):
    # will check sunrise for specified lat/lon

    if lat and lon:
        sat_num = get_sat(lat,lon,dt)

    #if check_sun and lat and lon:
    #    check_sunrise_sunset_lat_lon(dt, lat, lon)
    # will check sunrise for CONUS
    if check_sun:
        check_sunrise_sunset(dt)

    goes_dir = data_dir + 'goes/'
    fs = s3fs.S3FileSystem(anon=True)

    use_fns = get_filelist(dt, fs, lat, lon, sat_num, product, scope, bands)
    file_locs = []
    for file_path in use_fns:
        fn = file_path.split('/')[-1]
        dl_loc = goes_dir+fn
        file_locs.append(dl_loc)
        if os.path.exists(dl_loc):
            print("{} already exists".format(fn))
        else:
            print('downloading {}'.format(fn))
            fs.get(file_path, dl_loc)
    if len(file_locs) > 0:
        return file_locs
    else:
        print('ERROR NO FILES FOUND FOR TIME REQUESTED: ', dt)

def main(time_list, lat=None, lon=None, sat_num='16', product='ABI-L1b-Rad', scope='F', check_sun=True, bands=[1,2,3]):
    if scope != 'C' and scope != 'F':
        raise ValueError('scope value of {} is invalid. Choose C for conus and F for full disk'.format(scope))
    sat_nums = ['16', '17', '18', '19']
    if sat_num not in sat_nums:
        raise ValueError('sat_num value of {} is invalid. choose 16 for GOES-EAST and 17 or 18 for GOES-WEST'.format(sat_num))
    for dt in time_list:
        download_goes(dt, lat, lon, sat_num, product, scope, bands)

#if __name__ == '__main__':
#    lat = sys.argv[2]
#    lon = sys.argv[3]
#    sat_num = sys.argv[4]
#    main(input_dt, lat, lon, sat_num)

def get_time_list(input_dt):
    dt = pytz.utc.localize(datetime.strptime(input_dt, '%Y/%m/%d %H:%M'))
    num_hrs = 4
    time_list =[dt]
    for hr in range(1, num_hrs+1):
        time_list.append(dt + timedelta(hours=hr))
    return time_list


if __name__ == '__main__':
    #input_dt = sys.argv[1]
    #lat = sys.argv[2]
    #lon = sys.argv[3]
    #input_dt = "2022/01/04 21:00"
    #lat = 32.91
    #lon = -93.18
    dt_str = '2018/11/16 16:15'
    lat = '40.7'
    lon = '-120.3'
    time_list = get_time_list(dt_str)
    #main(input_dt, bands=bands)
    main(time_list, lat, lon, bands = [1,2,3])

