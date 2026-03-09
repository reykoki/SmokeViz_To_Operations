import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import pyproj
from helper_functions import *
from datetime import datetime
import pytz
import cartopy.crs as ccrs
# Define a function to return a Lambert Conformal Conic (LCC) projection
def get_proj():
    lcc_proj = ccrs.LambertConformal(
        central_longitude=262.5,        # Centered roughly on central CONUS
        central_latitude=38.5,          # Standard parallel and projection center latitude
        standard_parallels=(38.5, 38.5),# True at 38.5°N; reduces distortion near this latitude
        globe=ccrs.Globe(               # Define Earth shape with spherical approximation
            semimajor_axis=6371229,
            semiminor_axis=6371229     # Use spherical Earth model (radius in meters)
        )
    )
    return lcc_proj


from satpy import Scene
from pyresample import create_area_def

# Define a function to create a Satpy Scene and resample it to a custom projection and resolution
def get_scn(fns, to_load, extent, res=3000, proj=get_proj(), reader='abi_l1b', print_info=False):
    # Create a Scene object with the specified reader and list of file paths
    scn = Scene(reader=reader, filenames=fns)

    # Load the specified datasets or composites (without generating them yet)
    scn.load(to_load, generate=False)

    # Define a custom area using the provided projection, resolution (in meters), and extent
    my_area = create_area_def(
        area_id='my_area',
        projection=proj,
        resolution=res,
        area_extent=extent
    )

    # Optionally print available datasets and composite names in the Scene object
    if print_info:
        print("Available channels in the Scene object:\n", scn.available_dataset_names())
        print("\nAvailable composites:\n", scn.available_composite_names())
        print("\nArea definition:\n", my_area)

    # Resample the Scene data to the defined area; returns a new Scene object
    new_scn = scn.resample(my_area)
    return new_scn


from satpy.enhancements.enhancer import get_enhanced_image

# Extract and format RGB image data from the Scene object for plotting
def get_RGB(scn, composite):
    # Apply Satpy's enhancement pipeline to get display-ready RGB data (e.g., True Color)
    RGB = get_enhanced_image(scn[composite]).data.compute().data  # Dask array → NumPy array

    # Reorder dimensions from (bands, height, width) to (height, width, bands) for plotting
    RGB = np.einsum('ijk->jki', RGB)

    # Replace NaN values with 0 (black) for cleaner display
    RGB[np.isnan(RGB)] = 0

    return RGB


def get_sat_fns(dt_str, lat, lon):
    # Convert the string to a naive datetime object, then localize it to UTC
    dt = pytz.utc.localize(datetime.strptime(dt_str, '%Y/%m/%d %H:%M'))
    sat_fns = glob(f"./data/goes/*G1*_s{dt.strftime("%Y%j%H")}*")
    if len(sat_fns) == 0:
        print("you need to download the satellite files from a head node")
        print(f'python scripts/grab_goes.py "{dt_str}" "{lat}" "{lon}"')
    else:
        print(sat_fns)
    return sat_fns


# Compute the bounding box (extent) in projected coordinates around a lat/lon center
def get_extent(lat, lon, res, img_size):
    # Convert the projection object to a callable pyproj transformer
    lcc_proj = pyproj.Proj(get_proj())

    # Project the center latitude and longitude into the Lambert Conformal grid (in meters)
    center = lcc_proj(lon, lat)

    # Compute half the image width/height in meters
    dist = int(img_size / 2 * res)

    # Calculate the bounding box around the center point
    x0 = center[0] - dist
    y0 = center[1] - dist
    x1 = center[0] + dist
    y1 = center[1] + dist

    # Return the extent as [xmin, ymin, xmax, ymax]
    return [x0, y0, x1, y1]

def coords_from_lat_lon(lat,lon, res=1000, img_size=256): # img_size - number of pixels
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_0, lat_0 = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_1, lat_1 = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    lats = np.linspace(lat_1, lat_0, 5)
    lons = np.linspace(lon_0, lon_1, 5)
    return lats, lons

def create_composite(sat_fns, lat, lon, composites=['cimss_true_color_sunz_rayleigh', 'airmass'], res=1000, img_size=512):
    # Create a filename header using the GOES timestamp and coordinates
    fn_head = 'G' + sat_fns[0].split('_G')[-1].split('_c')[0] + '_' + lat + '_' + lon

    # Calculate spatial extent (in meters) around the specified location
    extent = get_extent(lat, lon, res, img_size)

    # Load and resample the selected composite products into a Satpy Scene object
    scn = get_scn(sat_fns, composites, extent, res)

    # Generate and save each composite as a 3-channel RGB image
    for composite in composites:
      data = get_RGB(scn, composite)
      #save_data(data, composite, fn_head)

    # Return the filename header and the Scene object for reference
    return data

def plot_data(lat, lon, res, img_size, data, dt_str):
    lats, lons = coords_from_lat_lon(lat, lon, res, img_size)
    # Plot the composite image with geolocated axes
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(data)
    # Set Y-axis ticks and labels using latitude values
    plt.yticks(np.linspace(0, data.shape[0] - 1, len(lats)), np.round(lats, 2), fontsize=12)
    plt.ylabel('latitude (degrees)', fontsize=16)
    # Set X-axis ticks and labels using longitude values
    plt.xticks(np.linspace(0, data.shape[1] - 1, len(lons)), np.round(lons, 2), fontsize=12)
    plt.xlabel('longitude (degrees)', fontsize=16)
    # Add a title and adjust layout
    plt.title(dt_str, fontsize=20)
    plt.tight_layout(pad=0)
    plt.show()
    print(f"{dt_str} ({lat}, {lon})")
