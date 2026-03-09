from collections import Counter
import matplotlib.pyplot as plt
import math
from glob import glob
import numpy as np
import skimage
from datetime import datetime


def get_lat_lon(coords_fn):
    lat_lon = skimage.io.imread(coords_fn, plugin='tifffile')
    lat = lat_lon[:,:,0]
    lon = lat_lon[:,:,1]
    return lat[::int(np.ceil(lat_lon.shape[0]/5)),0], lon[-1,::int(np.ceil(lat_lon.shape[0]/5))]

def get_data(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/" + fn)[0]
    truth_fn = glob(data_loc + "truth/" + fn)[0]
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    truths = skimage.io.imread(truth_fn, plugin='tifffile')
    lat, lon = get_lat_lon(fn,data_loc)
    return RGB, truths, lat, lon

def get_datetime_from_fn(fn):
    start = fn.split('_s')[-1].split('_e')[0][0:13]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    start_readable = start_dt.strftime('%Y/%m/%d %H:%M UTC')
    return start_readable

def get_mesh(num_pixels):
    x = np.linspace(0,num_pixels-1,num_pixels)
    y = np.linspace(0,num_pixels-1,num_pixels)
    X, Y = np.meshgrid(x,y)
    return X,Y

# use highest factorial where the numbers are closest together so for n = 16 do (4, 4) not (2, 8)
def get_num_col_row(n):
    a = np.round(math.sqrt(n), 0)
    while n%a > 0:
        a -= 1
    if a == 1 and n > 3:
        fact_1, fact_2 = get_num_col_row(n+1)
        return fact_1, fact_2
    return int(a), int(n/a)

def check_same_time(fns):
    t = fns[0].split('_')[2][1:-5]
    for fn in fns:
        fn_t = fn.split('_')[2][1:-5]
        if fn_t != t:
            raise ValueError("The times for these files do not match! \n {}".format(fns))

def get_sat_num_from_fns(fns):
    sat_nums = []
    for fn in fns:
        sat_num = fn.split('G')[-1].split('_')[0]
        sat_nums.append(sat_num)
    # if there is more than one sat num, choose the one with more files
    if len(set(sat_nums)) > 1:
        sat_file_cnt = Counter(sat_nums)
        return sat_file_cnt.most_common(1)[0][0]
    return sat_nums[0]

def get_fns_from_input_dt(date_str, data_loc, sat_num):
    if sat_num==None:
        fns = glob("{}data/C*s{}*.tif".format(data_loc, date_str))
        sat_num = get_sat_num_from_fns(fns)
    fns = glob("{}data/C*_G{}_*s{}*.tif".format(data_loc, sat_num, date_str))
    fns.sort()
    return fns

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def plot_all_bands(fn_head, data_loc="./data/", sat_num=None):
    #fns = get_fns_from_input_dt(input_start, data_loc, sat_num)
    fns = glob("{}data/C[01]*{}*.tif".format(data_loc, fn_head))
    fns.sort()
    check_same_time(fns)
    if len(fns) == 1:
        plot_band(fns[0])
        return
    n_row, n_col = get_num_col_row(len(fns))
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*3,n_row*3))
    band_to_mu = {'C01': '0.47', 'C02': '0.64', 'C03': '0.86', 'C04': '1.37', 'C05': '1.61', 'C06': '2.24', 'C07': '3.9', 'C08': '6.2', 'C09': '6.9' , 'C10': '7.3' , 'C11': '8.5' , 'C12': '9.6' , 'C13': '10.3', 'C14': '11.2', 'C15': '12.3', 'C16': '13.3'}
    if n_row == 1:
        for idx, fn in enumerate(fns):
            band = fn.split('_')[0]
            data = skimage.io.imread(fn, plugin='tifffile')
            data = normalize(data)
            ax[idx].imshow(data, cmap='Greys_r')
            ax[idx].set_yticks([])
            ax[idx].set_xticks([])
            ax[idx].set_title(band,fontsize=20)
    else:
        while len(fns)>1:
            for row in range(n_row):
                for col in range(n_col):
                    fn = fns.pop(0)
                    band = fn.split('/')[-1].split('_')[0]
                    mu = band_to_mu[band]
                    data = skimage.io.imread(fn, plugin='tifffile')
                    data = normalize(data)
                    ax[row][col].imshow(data, cmap='Greys_r')
                    ax[row][col].set_yticks([])
                    ax[row][col].set_xticks([])
                    ax[row][col].set_title(r'{} $\mu m$'.format(mu),fontsize=14)
                    if len(fns) == 0:
                        break
    if col < n_col-1:
        for col_idx in range(col + 1, n_col):
            ax[row][col_idx].set_yticks([])
            ax[row][col_idx].set_xticks([])
            ax[row][col_idx].axis('off')
    plt.suptitle(get_datetime_from_fn(fn), fontsize=26)
    plt.show()

def plot_composite(composite, fn_head, data_loc="./data/"):
    data_fn = glob("{}data/{}_{}*.tif".format(data_loc, composite, fn_head))[0]
    coords_fn = glob("{}coords/{}.tif".format(data_loc, fn_head))[0]
    print(get_datetime_from_fn(coords_fn))
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    lat, lon = get_lat_lon(coords_fn)
    plt.figure(figsize=(8, 6),dpi=100)
    plt.imshow(RGB)
    plt.yticks(np.linspace(0,RGB.shape[0]-1,5), np.round(lat,2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    plt.xticks(np.linspace(0,RGB.shape[0]-1,5), np.round(lon,2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    plt.title(composite,fontsize=24)
    plt.tight_layout(pad=0)
    plt.show()

def plot_band(fn, data_loc="./data/"):
    data_fn = data_loc + "data/" + fn
    coords_fn = data_loc + "coords/" + 'G' + data_fns[0].split('_G')[-1]
    band = fn.split('_')[0]
    band_data = skimage.io.imread(data_fn, plugin='tifffile')
    band_data = normalize(band_data)
    lat, lon = get_lat_lon(coords_fn)
    plt.figure(figsize=(8, 6),dpi=100)
    plt.imshow(band_data, cmap='Greys_r')
    plt.yticks(np.linspace(0,RGB.shape[0]-1,5), np.round(lat,2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    plt.xticks(np.linspace(0,RGB.shape[0]-1,5), np.round(lon,2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    plt.title(band,fontsize=24)
    plt.tight_layout(pad=0)
    plt.show()

def main(data_fns):
    #plot_all_bands(input_start)
    #plot_all_bands('G16_s20232672101174_e20232672103547_40.0_-105.27')
    plot_band('C03_G16_s20232672101174_e20232672103547_40.0_-105.27.tif', data_loc="./data/")
    #for data_fn in data_fns:
    #    plot_band(data_fn)


#if __name__ == '__main__':
    #data_fns = sys.argv[1]
    #main(data_fns)


if __name__ == '__main__':
    input_dt = '2023/09/24 21:00'
    input_start = '20232672101174'
    data_fns = ['C01_G16_s20232672101174_e20232672103547_40.0_-105.27.tif',  'C03_G16_s20232672101174_e20232672103547_40.0_-105.27.tif', 'C02_G16_s20232672101174_e20232672103547_40.0_-105.27.tif', "C09_G16_s20232672101174_e20232672103547_40.0_-105.27.tif","C09_G16_s20232672101174_e20232672103547_40.0_-105.27.tif", "C09_G16_s20232672101174_e20232672103547_40.0_-105.27.tif", "C11_G16_s20232672101174_e20232672103547_40.0_-105.27.tif", "C16_G16_s20232672101174_e20232672103547_40.0_-105.27.tif"]
    main(input_start)

