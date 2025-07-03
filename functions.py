import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.path import Path as PATH
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
from pathlib import Path
from sqlalchemy import text
import ipywidgets
from scipy.stats import binned_statistic_2d
from pyproj import Transformer, Geod, CRS
from geographiclib.geodesic import Geodesic
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from geographiclib.polygonarea import PolygonArea
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from scipy.integrate import quad


MAIN_PATH = Path.cwd()

def get_files_with_substring(directory):
    
    image_files_list = []
    roll_angle_files_list = []      
    
    for filename in directory.iterdir():
        if 'RAW_SubArray' in filename.stem:
            image_files_list.append(filename)
        elif ('Attitude' in filename.stem) or ('COR_Lightcurve-DEFAULT' in filename.stem):
            roll_angle_files_list.append(filename)
        else:
            continue
            #raise NameError(f"{filename} is an unexpected file that does not contain 'RAW_SubArray', 'Attitude', or 'COR_Lightcurve-DEFAULT.")
    return image_files_list, roll_angle_files_list

def read_images(file):
    
    metadata_idx = 9
    
    hdu = fits.open(file)
    images = hdu[1].data
    header = hdu[1].header
    metadata = hdu[metadata_idx].data
    overscan_left = hdu[7].data
    blank_left = hdu[2].data
    hdu.close()

    nb_images = np.shape(images)[0]    
    height_images = np.shape(images)[1]
    width_images = np.shape(images)[2]

    cx = header['X_WINOFF']
    cy = header['Y_WINOFF']

    
    return images, header, metadata, nb_images, height_images, width_images, overscan_left, blank_left, cx, cy

def read_attitude(file):
    
    hdu = fits.open(file)
    time = np.array(hdu[1].data['MJD_TIME'], '>f8').byteswap().newbyteorder()
    roll_angle = np.array(hdu[1].data['SC_ROLL_ANGLE'], '>f4').byteswap().newbyteorder()
    hdu.close()
    
    return time, roll_angle

def read_lc(file):
    
    hdu = fits.open(file)
    roll_angle = np.array(hdu[1].data['ROLL_ANGLE'], '>f4').byteswap().newbyteorder()
    hdu.close()
    
    return roll_angle

def read_SAA_map(file):
    
    hdu = fits.open(file)
    SAA_flag = hdu[1].data['SAA_FLAG']
    lat = np.array(hdu[1].data['LATITUDE'], '>i4').byteswap().newbyteorder()
    lon = np.array(hdu[1].data['LONGITUDE'], '>i4').byteswap().newbyteorder()
    hdu.close()
    
    df_SAA = pd.DataFrame({
    'SAA_FLAG': SAA_flag,
    'latitude': lat,
    'longitude': lon
    })
    
    return df_SAA

def find_closest_indices(arr1, arr2):
    closest_indices = []
    for val1 in arr1:
        closest_index = np.abs(arr2 - val1).argmin()
        closest_indices.append(closest_index)
    return closest_indices

def get_last_DarkFrame_and_BadPixelMap(folder_path):
    
    # Get list of files in the folder
    files = os.listdir(folder_path)
    
    #num_files=5 # number of files to check
    num_files=len(files) # number of files to check - now it checks all
    # Get file paths and their corresponding modification times
    file_times = [(os.path.join(folder_path, file), os.path.getmtime(os.path.join(folder_path, file))) for file in files]
    
    # Sort files by modification time in descending order
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)
    
    BadPixelMap = None
    DarkFrame = None
    for file in sorted_files[:num_files]:
        
        if ((BadPixelMap is None) and ('BadPixelMap' in file[0])):
                BadPixelMap = file[0]
        elif ((DarkFrame is None) and ('DarkFrame' in file[0])):
                DarkFrame = file[0]
        else:
            continue
        
    return BadPixelMap, DarkFrame

def coordinates_of_cosmic_pixels(binary_data):
    layers = len(binary_data)
    rows = len(binary_data[0]) if layers > 0 else 0
    columns = len(binary_data[0][0]) if rows > 0 else 0
    result = []

    def get_coords(x, y, val, visited, current_layer):
        visit = [(x,y)]
        coordinates = []

        while visit:
            cx, cy = visit.pop()
            if (cx, cy) in visited:
                continue

            visited.add((cx, cy))
            coordinates.append((cx, cy, current_layer))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < columns and (nx, ny) not in visited and binary_data[current_layer][nx][ny] == val:
                    visit.append((nx, ny))
    
        return coordinates
    
    for layer in range(layers):
        visited = set()
        result.append([])
        for j in range(rows):
            for k in range(columns):
                if (j, k) not in visited and binary_data[layer][j][k] != 0:
                    val = binary_data[layer][j][k]
                    coordinates = get_coords(j, k, val, visited, layer)
                    if len(coordinates) > 1:
                        result[layer].append(coordinates)
    return(result)

def bias_correction(overscan_left, blank_left, image):
    subarray = image.copy().astype('float64')

    mean_overscan = []
    mean_blank = []
    mean_general = []
    corrected_array = []

    for i in range(len(overscan_left)):
        mean_overscan.append(np.nanmean(overscan_left[i]))
        mean_blank.append(np.nanmean(blank_left[i]))

    if len(mean_overscan) == len(mean_blank):
        for i in range(len(mean_overscan)):

            mean = (mean_overscan[i] + mean_blank[i])/2
            mean_general.append(mean)

    for i in range(len(subarray)):
        corrected_array.append(subarray[i] - mean_general[i])

    corrected_array = np.array(corrected_array)
    return corrected_array

def read_BadPixelMap_and_DarkFrame(file):
    hdu = fits.open(file)

    map = hdu[1].data
    return map

def cut_maps(map, cx, cy):
    
    if len(map.shape) == 2:
        new_map = map[cy:cy+200, cx:cx+200]
    elif len(map.shape) == 3:
        new_map = map[0, cy:cy+200, cx:cx+200]

    return new_map

def bad_pixel_correction(new_bad_pixel_map, new_dark_frame, image):
    bad_pixel_map = new_bad_pixel_map.copy().astype('float64')
    dark_frame = new_dark_frame.copy().astype('float64')
    subarray = image.copy().astype('float64')

    #find the non-zero elements in the bad pixel map and log the coordinates
    coordinates_non_zero_elements = []
    for i in range(len(bad_pixel_map)):
        for j in range(len(bad_pixel_map[i])):
            if bad_pixel_map[i][j] != 0:
                coordinates_non_zero_elements.append((i,j))

    readout_noise = 3.5
    corrected_image = [layer[:] for layer in subarray]

    for (x,y) in coordinates_non_zero_elements:
        for z in range(len(subarray)):
            value = subarray[z][x][y]
            if value > 0:
                corrected_image[z][x][y] = corrected_image[z][x][y] - dark_frame[x][y]
            elif value < 0:
                corrected_image[z][x][y] = corrected_image[z][x][y] - readout_noise

    corrected_image = np.array(corrected_image)
    return corrected_image

def get_pixel_adu(images, coordinates):
    adu = []
    for layer in coordinates: # access each cosmic (>= 2 elements):
        group_values = []
        for coord in layer: # acess each pixel in cosmic, coordinate tuple (x coordinate, y coordinate, # of layer)
            for (x,y,z) in coord:
                group_values.append(images[z][x][y])
        adu.append(group_values)
    
    return adu

def calculate_electrons(adu, gain):
    number_of_electrons = []
    sum_electrons = []
    for group in adu:
        group_electrons = []
        electrons = 0
        for i in group:
            electron_count = gain*i
            electrons += electron_count
            group_electrons.append(electron_count)
        number_of_electrons.append(group_electrons)
        sum_electrons.append(electrons)
    
    return number_of_electrons, sum_electrons

def calculate_flux(adu, gain, exposure_time, stacking_order):
    exp_time = exposure_time/stacking_order
    pixel_flux = []
    flux_cosmic = []
    for group in adu:
        group_values = []
        sum_fluxes = 0
        for i in group:
            flux = (i*gain)/exp_time
            sum_fluxes += flux
            group_values.append(flux)
        flux_cosmic.append(sum_fluxes)
        pixel_flux.append(group_values)

    flux_cosmic = np.array(flux_cosmic)
       
    return(pixel_flux, flux_cosmic)

def get_edges_mask(image):
    
    edges_mask = image == 0
    
    return edges_mask

def subtract_temporal_median_image(images):
    '''
    Compute the median for all pixels through the cube to get the median image and remove it from each image in the data cube
    '''
    median_subtracted_images = images.copy().astype('float64')

    median_image_per_pixel = np.nanmedian(median_subtracted_images, axis=(0))
    
    #subtract median image from each image in the data cube
    for i in range(len(median_subtracted_images)):
        median_subtracted_images[i] = median_subtracted_images[i] - median_image_per_pixel

    median_subtracted_images = np.abs(median_subtracted_images)

    return median_subtracted_images

def subtract_median_image(images, circular_mask):
    
    median_images = images.copy().astype('float64')

    for i in range(len(images)):
        mask_zeros = median_images[i] != 0
        median_image = np.nanmedian(median_images[i][mask_zeros])
        median_images[i][mask_zeros] = median_images[i][mask_zeros] - median_image

    median_images = np.abs(median_images)
    return median_images

def derotate_SAA(attitude_file, images, metadata_images, nb_images, height_images, width_images):
    
    images = images.astype(np.float64)

    # Science visit
    time_attitude, roll_angle = read_attitude(attitude_file)
    
    # # Find the roll_angle associated with each images
    time_images = metadata_images['MJD_TIME']
    closest_time_images = find_closest_indices(time_images, time_attitude)
    roll_angle_images = roll_angle[closest_time_images]
    
    # Find the roll_angle associated with each images
    time_images_utc = pd.to_datetime(np.array(metadata_images['UTC_TIME']), format='%Y-%m-%dT%H:%M:%S.%f', utc=True)
    time_images_utc_jd = time_images_utc.to_julian_date()
    # closest_time_images = find_closest_indices(metadata_images['MJD_TIME'], time_attitude)
    # roll_angle_images = roll_angle[closest_time_images]

    derotated_openCV_images = np.zeros((nb_images, width_images, height_images), dtype=np.float64)
    first_image_angle = roll_angle_images[0]

    # Derotate each image
    for i,image in enumerate(images):
        rotate_by = -(roll_angle_images[i] - first_image_angle)
        rotation_matrix = cv2.getRotationMatrix2D((width_images / 2, height_images / 2), rotate_by, 1)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (width_images, height_images))
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width_images, height_images), flags=cv2.INTER_NEAREST)
        derotated_openCV_images[i] = rotated_image
            
    return derotated_openCV_images, time_images_utc, time_images_utc_jd, roll_angle_images

def derotate_images(attitude_file, images, nb_images, height_images, width_images):
    
    images = images.astype(np.float64)
    roll_angle = read_lc(attitude_file)
    
    derotated_openCV_images = np.zeros((nb_images, width_images, height_images), dtype=np.float64)
    first_image_angle = roll_angle[0]

    # Derotate each image
    for i,image in enumerate(images):
        rotate_by = -(roll_angle[i] - first_image_angle)
        rotation_matrix = cv2.getRotationMatrix2D((width_images / 2, height_images / 2), rotate_by, 1)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (width_images, height_images))
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width_images, height_images), flags=cv2.INTER_NEAREST)
        derotated_openCV_images[i] = rotated_image
    
    return derotated_openCV_images

def apply_mask_to_images(images, mask, value):
    
    masked_images = images.copy()
    
    for i in range(len(masked_images)):
        masked_images[i][mask] = value
        
    return masked_images  

def create_contaminant_mask(derotated_openCV_images, edges_mask, saa_or_science, enlarge_mask = True, inspect_threshold = False, threshold = 0):
    
    image_for_mask = np.nanmedian(derotated_openCV_images, axis=(0)) # median
    
    ### LUISE Threshold finding, to be activated ###
    # image_for_hist = image_for_mask[image_for_mask != 0] # removed 0
    
    # threshold = find_threshold(image_for_hist)[2]

    #print(find_threshold(image_for_mask))

    if saa_or_science == 'SAA':
        mask = image_for_mask > threshold
    else:
        mask_central_star = create_circular_mask(np.shape(derotated_openCV_images)[1], 25)
        mask = (image_for_mask > threshold) | ~mask_central_star
        
        
    med_or_mean = 'median'
        
    mask = mask.astype(np.float64)
    
    final_mask = mask
    
    if enlarge_mask:
        # Define the kernel for dilation
        kernel = np.ones((3,3), dtype=np.float64)
        extension_mask = 2
        enlarged_mask = cv2.dilate(mask, kernel, iterations= extension_mask) == 1
        final_mask = enlarged_mask
    
    
                
    if inspect_threshold:
        
        hist_median_image = image_for_mask.copy()
        
        # Correct edge to 255
        hist_median_image[edges_mask] = 255
        all_pixels_median = hist_median_image.flatten()
        data_without_0 = all_pixels_median[all_pixels_median != 255]
        
        # Cumulative distribution function (CDF)
        sorted_data = np.sort(data_without_0)
        cumulative = np.linspace(0, 1, len(sorted_data))

        # Plot median image and pixels values histogram and CDF
        fig, ax = plt.subplots(ncols = 3, figsize=(15,4))
        im = ax[0].imshow(image_for_mask, origin='lower', cmap = 'viridis')
        plt.colorbar(im)
        ax[0].set_title(f'{med_or_mean} image', weight = 'bold')
        ax[1].hist(data_without_0.flatten(), bins = len(np.unique(data_without_0.flatten())))
        ax[1].set_ylim(0,100)
        ax[1].axvline(threshold, c = 'C1', label='Threshold')
        ax[1].legend()
        ax[1].set_title('Histogram of the pixel values', weight = 'bold')
        ax[2].plot(sorted_data, cumulative)
        ax[2].axvline(threshold, c = 'C1', label='Threshold')
        ax[2].legend()
        ax[2].set_title('CDF of the pixel values', weight = 'bold')
        
        image_masked = image_for_mask.copy()
        image_masked[final_mask] = 0
        
        # Plot median image and pixels values histogram and CDF
        fig, ax = plt.subplots(ncols = 4, figsize=(20,4))
        
        ax[0].imshow(image_for_mask, origin='lower', cmap = 'viridis')
        ax[0].set_title(f'{med_or_mean} image', weight = 'bold')

        ax[1].imshow(mask, origin='lower', cmap = 'viridis')
        ax[1].set_title('mask', weight = 'bold')

        ax[2].imshow(enlarged_mask, origin='lower', cmap = 'viridis')
        ax[2].set_title('enlarged mask', weight = 'bold')

        im= ax[3].imshow(image_masked, origin='lower', cmap = 'viridis')
        ax[3].set_title('Applied mask', weight = 'bold')
        plt.colorbar(im)

        plt.show()   

    return final_mask, threshold

def detect_cosmics(masked_images, threshold): 

    # convert masked images to binary
    binary_images = masked_images.copy()
    for i,image in enumerate(binary_images):
        _, img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary_images[i] = img
        
    # Initialize the counter of cosmics per image
    loc_cosmics  = []
    stats_cosmics = []
    centroids_cosmics = []
    # images_contours = [] # List of contours for all images (size of nb of images)

    
    # 1. Find the countours
    # 2. Go through all contours in images, and remove the ones that are only one pixels
    # 3. Store the remaining in the images_contours
    # 4. Get the number of pixels in the largest cosmic
    # for i,image in enumerate(binary_images):
        # curr_image_contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # indices_of_1_pixel_contours = []
        # contours_list = list(curr_image_contours)
        # for j,contour in enumerate(contours_list): # if there is only one pixel, remove this contour 
        #     nb_pixels_in_contour = len(contour)
        #     if (nb_pixels_in_contour == 1):
        #         indices_of_1_pixel_contours.append(j)
        # for index in sorted(indices_of_1_pixel_contours, reverse=True):
        #     del contours_list[index]
        # nb_cosmics[i] = len(contours_list)
        # images_contours.append(contours_list)
    
    # new method, use the connectedComponentsWithStats function to recover the number of pixels in the blob. The above method doesn't work as intended, as only the pixels of the contours are returned, and not the ones inside.
    for i,image in enumerate(binary_images):
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8))
        loc_cosmics.append(labels) 
        stats_cosmics.append(stats) 
        centroids_cosmics.append(centroids)

      
    return binary_images, loc_cosmics, stats_cosmics, centroids_cosmics

def cosmics_metrics(visit_labeled_cosmics, visit_info_cosmics, nb_non_masked_pixels, pixel_size, total_exp_time):
    
    nb_cosmics_arr = np.array([])
    nb_pixels_largest_cosmics_arr = np.array([])
    percentage_cosmic_pixels_arr = np.array([])
    density_cosmics_arr = np.array([])
    
    for image_info_cosmic in visit_info_cosmics:

        # get stats for all identified cosmics, excluding the first row which is the background
        cosmics_stats = image_info_cosmic[1:,-1]
        if (len(cosmics_stats) == 0) or (len(cosmics_stats[cosmics_stats != 1]) == 0):
            nb_cosmics_arr = np.append(nb_cosmics_arr,0)
            nb_pixels_largest_cosmics_arr = np.append(nb_pixels_largest_cosmics_arr,0)  
            percentage_cosmic_pixels_arr = np.append(percentage_cosmic_pixels_arr,0)     
            density_cosmics_arr = np.append(density_cosmics_arr,0)
            continue
        else:
            stats_no_1_pix = cosmics_stats[cosmics_stats != 1] # exclude single pixels
            # nb_cosmics
            nb_cosmics = len(stats_no_1_pix)
            nb_cosmics_arr = np.append(nb_cosmics_arr,nb_cosmics) 
            # largest cosmic
            nb_pixels_largest_cosmics = np.max(stats_no_1_pix)
            nb_pixels_largest_cosmics_arr = np.append(nb_pixels_largest_cosmics_arr, nb_pixels_largest_cosmics)
            # percentage affected pixels
            nb_pix_affected_no_1_pix = np.sum(stats_no_1_pix)
            percentage_cosmic_pixels = (nb_pix_affected_no_1_pix/nb_non_masked_pixels)*100          
            percentage_cosmic_pixels_arr = np.append(percentage_cosmic_pixels_arr, percentage_cosmic_pixels)
            # density cosmics
            cm2_analysed = nb_non_masked_pixels*((pixel_size*1e2)**2) # cm2
            density_cosmics = nb_cosmics/cm2_analysed/total_exp_time
            density_cosmics_arr = np.append(density_cosmics_arr,density_cosmics)
            
    # percentage affected pixels per sec
    percentage_cosmic_pixels_per_sec_arr = percentage_cosmic_pixels_arr/total_exp_time
    
    return nb_cosmics_arr, density_cosmics_arr, nb_pixels_largest_cosmics_arr, percentage_cosmic_pixels_arr, percentage_cosmic_pixels_per_sec_arr

# def find_largest_cosmic(pix_cosmics):
#     """
#     Use to identify the cosmic ray with the largest number of pixel. With more than a 100 we are very probably looking at a satelitte trail or straylight.
#     """
    
#     if len(pix_cosmics) > 0:
#         nb_pixels_cosmic = []
#         for cosmic in pix_cosmics:
#             nb_pixels_cosmic.append(np.shape(cosmic)[0])
            
#         return int(np.max(nb_pixels_cosmic))
#     else:
#         return 0


# def cosmic_fraction(nb_remaining_pixels, contours):
    
#     # nb_cosmic_pixels = np.array([])
#     # nb_cosmic_pixels = 0
#     # for i in range(len(contours)):
#     #     for j in range(len(contours[i])):
#     #         nb_cosmic_pixels += len(contours[i][j])
            
#     nb_cosmic_pixels = np.array([])
#     for i in range(len(contours)): # iterate through images
#         nb_cosmic_pixels_current_image = 0 # nb pixels affected by cosmic in the current image
#         for j in range(len(contours[i])): # iterate through each CR identified in the current image
#             nb_cosmic_pixels_current_image += len(contours[i][j]) # add the nb of pixels in a given CR
#         pixel_fraction = nb_cosmic_pixels_current_image / nb_remaining_pixels
#         nb_cosmic_pixels = np.append(nb_cosmic_pixels, pixel_fraction)

#     # return pixel_fraction
#     return nb_cosmic_pixels


def reshape_flatten_images(images_type, images):
    
    size = int(np.sqrt(len(images[images_type].iloc[0])))
    reshaped_masked_images = np.zeros((len(images),size,size))
    images = images[images_type].values
    for i in range(len(images)):
        reshaped_masked_images[i] = images[i].reshape(size,size)
    
    return reshaped_masked_images

def unfold_interp_fold(data_RES, full_table, order = 3):
    
    """
    Unfold, interpolate, and fold back longitude values to ensure continuity.
    
    Parameters:
        RES_ORB_reindexed (pandas.DataFrame): DataFrame containing reindexed orbital data with 'JD' (Julian Date),
                                              'LATITUDE', and 'LONGITUDE' columns.
        data_RES (pandas.DataFrame): DataFrame with original 'LATITUDE' and 'LONGITUDE' data.
        images (pandas.DataFrame): DataFrame the images.
        order (int, optional): The order of the interpolation spline. Defaults to 3.
    
    Returns:
        pandas.DataFrame: DataFrame with unfolded, interpolated, and folded back longitude values. Merged with images data.
        
    This function takes orbital data, unfolds discontinuous longitude values, interpolates the latitude and longitude 
    values using spline interpolation, folds back the longitude values to the range [-180, 180] degrees, and then merges
    the data with additional images information. The unfolded and interpolated data, along with images details,
    is returned as a pandas DataFrame.
    """
    
    final_RES_ORB = full_table.copy() # RES_ORB_reindexed.copy()
    unfolded_data_RES = data_RES.copy()
    unfolded_data_RES['LONGITUDE'] -= 180

    # # Identify the points of discontinuity
    discontinuity_indices = np.where(np.abs(np.diff(unfolded_data_RES['LONGITUDE'])) > 180)[0]

    # # Adjust longitude values to ensure continuity
    for idx in discontinuity_indices:
        unfolded_data_RES['LONGITUDE'].values[idx + 1:] -= 360
        
    from scipy.interpolate import make_interp_spline
    cs_lat = make_interp_spline(unfolded_data_RES.index.values, unfolded_data_RES['LATITUDE'].values, k = order)
    cs_lon = make_interp_spline(unfolded_data_RES.index.values, unfolded_data_RES['LONGITUDE'].values, k = order)

    # Interpolate and construct a dataframe with time entries from both 
    final_RES_ORB['LATITUDE']  = cs_lat(final_RES_ORB.index.sort_values().values)
    final_RES_ORB['LONGITUDE'] = cs_lon(final_RES_ORB.index.sort_values().values)

    # Fold back RES_ORB_reindexed
    final_RES_ORB['LONGITUDE'] %= 360
    final_RES_ORB['LONGITUDE'] -= 180
       
    return final_RES_ORB

def create_circular_mask(size, radius):

    # Build circular mask 
    #mask_circular = np.zeros((size, size))
    center_x, center_y = size // 2, size // 2
    # Create a grid of (x,y) coordinates
    y, x = np.ogrid[:size, :size]
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Create a mask for points within the radius
    mask_circular = distance_from_center > radius
    # Apply the mask to the array
    #mask_circular[mask] = 1
    
    return mask_circular

def genreate_diagnostic_plots(derotated_openCV_images, images, subtracted_median_images, temporal_median_substracted_images, threshold_noise, threshold_cosmics, mask, masked_images, binary_images, n, nb_cosmics, title, plot_name, show_plot):
    
    output_plots = MAIN_PATH / "output_plots"
    output_plots.mkdir(parents=True, exist_ok=True)

    
    plt.close('all')
    fig, axs = plt.subplots(ncols = 8, nrows=2, figsize=(35,8))
    plt.subplots_adjust(left=0.05, right=0.99)
    median_image = np.nanmedian(derotated_openCV_images, axis=(0))
    mean_image = np.nanmean(derotated_openCV_images, axis=(0))
    
    # ax[0,0].hist(np.log(images[n]+1).flatten(), bins = 100)
    # ax[1,0].imshow(np.log(images[n]+1), origin = "lower")
    # ax[0,0].set_title("Original subArray (log)")
    # ax[0,0].hist(images[n].flatten(), bins = 100)
    axs[1,0].imshow(images[n], origin = "lower")
    axs[0,0].set_title("Original subArray")
    axs[0,0].set_ylim(0,100)
    axs[0,1].hist(subtracted_median_images[n].flatten(), bins = 100)#len(np.unique(subtracted_median_images[n].flatten())))
    axs[1,1].imshow(subtracted_median_images[n], origin = "lower")
    axs[0,1].set_title("median value removed")
    axs[0,1].set_ylim(0,100)
    axs[0,2].hist(temporal_median_substracted_images[n].flatten(), bins = 100)#len(np.unique(temporal_median_substracted_images[n].flatten())))
    axs[1,2].imshow(temporal_median_substracted_images[n], origin = "lower")
    axs[0,2].set_title("median of pixels removed")
    axs[0,2].set_ylim(0,100)
    # ax[0,3].hist(derotated_openCV_images[n].flatten(), bins = len(np.unique(derotated_openCV_images[n].flatten())))
    # ax[1,3].imshow(derotated_openCV_images[n], origin = "lower")
    # ax[0,3].set_title("derotated converted image")
    # ax[0,3].hist(np.log(derotated_openCV_images[n].flatten()+1), bins = len(np.unique(derotated_openCV_images[n].flatten())))
    axs[1,3].imshow(np.log(derotated_openCV_images[n]+1), origin = "lower")
    axs[0,3].set_ylim(0,500)
    axs[0,3].set_title("log(derotated converted image)")
    #ax[0,3].set_xscale('log')
    axs[0,4].hist(median_image.flatten(), bins = len(np.unique(median_image.flatten())))
    axs[0,4].axvline(threshold_noise, c = 'C2', linewidth = 2)
    axs[1,4].imshow(median_image, origin = "lower",norm=colors.LogNorm())
    axs[0,4].set_xlim(0,5*threshold_noise)
    axs[0,4].set_title("median image of visit")
    #ax[0,4].set_xscale('log')
    # ax[0,5].hist(mean_image.flatten(), bins = len(np.unique(mean_image.flatten())))
    # ax[0,5].axvline(threshold_noise, c = 'C2', linewidth = 2)
    # ax[1,5].imshow(mean_image, origin = "lower")
    # ax[0,5].set_ylim(0,100)
    # ax[0,5].set_title("mean image of visit")
    axs[1,5].imshow(mask, origin = "lower")
    axs[0,5].set_ylim(0,100)
    axs[0,5].set_title("designed mask")
    # ax[0,6].hist(np.log(masked_images[n]+0.1).flatten(), bins = len(np.unique(masked_images[n].flatten())))
    # ax[1,6].imshow(np.log(masked_images[n]+0.1), origin = "lower")
    # ax[0,6].axvline(np.log(threshold_cosmics+0.1), c = 'C2', linewidth = 2)
    # ax[0,6].set_title("log(masked image + 0.1)")
    axs[0,6].hist(masked_images[n].flatten(), bins = 100)# len(np.unique(masked_images[n].flatten())))
    axs[1,6].imshow(masked_images[n], origin = "lower")
    axs[0,6].axvline(threshold_cosmics, c = 'C2', linewidth = 2)
    axs[0,6].set_title("masked image")
    axs[0,6].set_xlim(0,5*threshold_cosmics)
    axs[0,6].set_ylim(0,100)
    axs[0,7].hist(binary_images[n].flatten(), bins = len(np.unique(binary_images[n].flatten())))
    axs[1,7].imshow(binary_images[n], origin = "lower")
    axs[0,7].text(0.25,0.9,f"{int(nb_cosmics)} detected cosmics", weight = 'bold', transform=axs[0,7].transAxes)
    axs[0,7].set_title("detected cosmics")
    fig.suptitle(title)
    plt.savefig(output_plots/plot_name, dpi = 600,format = 'png')
    
    rows, cols = axs.shape
    for row in range(rows):
        for col in range(cols):
            ax = axs[row,col]
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f"{output_plots}/{plot_name}_{row}_{col}", dpi = 600,format = 'png', bbox_inches=extent)

    # ax[1,0].savefig(f"{output_plots}/{plot_name}" + "_original_subarray", dpi = 600,format = 'png')
    # ax[1,1].savefig(f"{output_plots}/{plot_name}" + "_median_value_removed", dpi = 600,format = 'png')
    # ax[1,2].savefig(f"{output_plots}/{plot_name}" + "_median_pixels_removed", dpi = 600,format = 'png')
    # ax[1,3].savefig(f"{output_plots}/{plot_name}" + "_log_derotated", dpi = 600,format = 'png')
    # ax[1,4].savefig(f"{output_plots}/{plot_name}" + "_median_image_of_visit", dpi = 600,format = 'png')
    # ax[1,5].savefig(f"{output_plots}/{plot_name}" + "_designed_mask", dpi = 600,format = 'png')
    # ax[1,6].savefig(f"{output_plots}/{plot_name}" + "_masked_image", dpi = 600,format = 'png')
    # ax[1,7].savefig(f"{output_plots}/{plot_name}" + "_detected_cosmics", dpi = 600,format = 'png')
        
    #if show_plot:
        #plt.show()

def find_threshold(image, inspect_hist=True):
    images = image.copy()

    flatten_images = np.ndarray.flatten(images)
    flatten_images = flatten_images[flatten_images!= 0]

    ### using scipy stats ###

    hist, bin_edges = np.histogram(flatten_images, bins=1000)

    bin_treshold = 50 # threshold for minimum bin population

    n = len(hist)

    #x = np.linspace(10,500,1000)
    x = np.zeros((n),dtype=float) 
    for i in range(n):
        x[i] = (bin_edges[i+1]+bin_edges[i])/2

    
    hist_where = np.where(hist >= bin_treshold)
    x = x[hist_where]
    y = hist[hist_where]

    ### Gaussian ###
    # mu_initial = x[np.argmax(y)]              
    # sigma_initial = np.std(flatten_images)
    # amp_initial = np.max(y)

    # popt,pcov=curve_fit(gaussian,x,y,p0=[amp_initial, mu_initial, sigma_initial])

    # mu_fit =  popt[1]
    # sigma_fit = popt[2]

    # threshold = mu_fit + (5*sigma_fit)

    # if inspect_hist:
    #     x_hist = x
    #     params = popt
    #     fit = gaussian(x_hist, *params)

    #     fig,ax = plt.subplots()
    #     ax.hist(flatten_images, bins = 10000)
    #     ax.plot(x_hist, fit, 'r', label='Gaussian Fit')
    #     plt.title('Histogram inspection')
    #     plt.show()

    # return mu_fit, sigma_fit, threshold, popt, x


    ### Exponential ###

    rate_of_decay = np.min(y)/np.max(y)
    max_index = np.argmax(y)

    m_initial = np.max(y)
    t_initial = rate_of_decay
    b_initial = x[max_index]

    popt,pcov=curve_fit(exponential,x,y,p0=[m_initial, t_initial, b_initial])

    m_fit = popt[0]
    t_fit = popt[1]
    b_fit = popt[2]

    threshold = 50 #arbitrary for testing purposes

    m_array = [5000, 7500, 10000]
    t_array = [0.05, 0.1, 0.15]
    b_array = [1, 1.5, 2]

    if inspect_hist:
        x_hist = x
        for i in range(len(m_array)):
            for j in range(len(t_array)):
                for k in range(len(b_array)):
                    m = m_array[i]
                    t = t_array[j]
                    b = b_array[k]

                    print('Params:', m, t, b)
                    fit = exponential(x_hist, m, t, b)


                    fig,ax = plt.subplots()
                    ax.hist(flatten_images, bins = 1000)
                    ax.plot(x_hist, fit, 'r')
                    plt.title('Histogram inspection')
                    plt.show()

    return m_fit, t_fit, b_fit, threshold, popt, x

    ### Hyperbolic ###

    # max_index = np.argmax(y)

    # a_initial = 1
    # p_initial = 5
    # #-(x[max_index])
    # q_initial = 0

    # popt,pcov=curve_fit(hyperbolic,x,y,p0=[a_initial, p_initial, q_initial])

    # a_fit = popt[0]
    # p_fit = popt[1]
    # q_fit = popt[2]

    # threshold = 50 #arbitrary for testing purposes

    # if inspect_hist:
    #     x_hist = x
    #     params = popt
    #     fit = exponential(x_hist, *params)

    #     fig,ax = plt.subplots()
    #     ax.hist(flatten_images, bins = 10000)
    #     ax.plot(x_hist, fit, 'r', label='Exponential Fit')
    #     plt.title('Histogram inspection')
    #     plt.show()
    
    # return a_fit, p_fit, q_fit, threshold, popt, x

def gaussian(x,amp,mu,sigma):
    return amp*np.exp(-(x-mu)**2/2*sigma**2)

def exponential(x, m, t, b):
    return m*np.exp(-(t*x)+b)

def hyperbolic(x, a, p, q):
    return (a/(x+p))+q

# def remove_straylight(masked_images):
#     masked_array = masked_images.copy()

#     #calculate median/mean of each image in array (background) -> to do: test if mean or median works better
#     median_per_image = []
#     mean_per_image = []
#     for i in range(len(masked_array)):
#         median_per_image.append(np.median(masked_array[i]))
#         mean_per_image.append(np.mean(masked_array[i]))

#     #calculate median/mean of new array
#     median_new = np.median(median_per_image)
#     mean_new = np.mean(median_per_image)

#     background_threshold = median_new + 10 #arbitrary number for now -> see what will filter stray light the best

#     #create binary array to flag straylight images -> straylight images are flagged as 0
#     straylight_binary_array = (median_per_image > background_threshold)

#     return straylight_binary_array

def remove_straylight_new(masked_images,edges_circular_mask,contaminant_mask,loc_cosmics):
    # Now not taking into account the masked (circular subArray, masked stars + CR) pixels
    masked_array = masked_images.copy()
    
    
    mean_per_image_stars_mask = []
    mean_per_image_full_cosmics_mask = []
    for i,image in enumerate(masked_array):
        ## Full mask (circular subArray + stars + cosmics)
        cosmics_mask = loc_cosmics[i] > 0
        mask_stars = edges_circular_mask | contaminant_mask.astype(bool) 
        image[mask_stars] = np.nan # convert to nans the pixels outside the circular crop of the subArray and the masked stars
        mean_per_image_stars_mask.append(np.nanmean(image)) # mean with the stars masked
        # Now set to nan the cosmic affected pixels as well
        image[cosmics_mask] = np.nan # convert to nans the pixels affected by cosmics
        mean_per_image_full_cosmics_mask.append(np.nanmean(image)) # mean with the stars masked

    # compute percent increase of mean with the star mask vs the full mask
    percent_increase_star_masked_vs_cosmic_masked = ((np.array(mean_per_image_stars_mask)/np.array(mean_per_image_full_cosmics_mask))-1)*100

    #calculate mean of new array
    mean_visit_stars_mask = np.nanmedian(mean_per_image_stars_mask)
    mean_visit_full_cosmics_mask = np.nanmean(mean_per_image_full_cosmics_mask)
    
    thresh_mean_cosmic_mask = np.mean(mean_visit_full_cosmics_mask)+5
    thresh_diff = 200

    # Create binary array to flag straylight images -> straylight images are flagged as 1
    # Flag points as stray light if they show a large mean and a small diff between the star masked images and the full masked (inc. cosmics) images
    straylight_binary_array = (mean_per_image_full_cosmics_mask > thresh_mean_cosmic_mask) & (percent_increase_star_masked_vs_cosmic_masked < thresh_diff)

    # background_threshold = mean_new + 10 #arbitrary number for now -> see what will filter stray light the best
    # straylight_binary_array = (median_per_image > background_threshold)

    return straylight_binary_array

def check_positions(positions):
   coordinates_index = []
   for i in range(len(positions)):
      this_visit = i
      for j in range(len(positions[i])):
         for k in range(len(positions[i][j])):
               this_element = positions[i][j][k]
               this_element_index = {j,k}
               counter = 0

               if this_element != 0:
                  # check right
                  if k < len(positions[i][j])-1:
                     if positions[i][j][k+1] != 0:
                        counter += 1
                     
                  # check left
                  if k > 0:
                     if positions[i][j][k-1] != 0:
                        counter += 1

                  # check upper
                  if j > 0:
                     if positions[i][j-1][k] != 0:
                        counter += 1
                     
                  # check lower
                  if j < len(positions[i])-1:
                     if positions[i][j+1][k]:
                        counter += 1
                     
                #   # check upper right
                #   if j > 0 and k < len(positions[i][j])-1:
                #      if positions[i][j-1][k+1]:
                #         counter += 1
                     
                #   # check upper left
                #   if j > 0 and k > 0:
                #      if positions[i][j-1][k-1]:
                #         counter += 1
                     
                #   # check lower right
                #   if j < len(positions[i])-1 and k < len(positions[i][j])-1:
                #      if positions[i][j+1][k+1]:
                #         counter += 1
                     
                #   # check lower left
                #   if j < len(positions[i])-1 and k > 0:
                #      if positions[i][j+1][k-1]:
                #         counter += 1

               if counter > 0:
                  coordinates_index.append(this_element_index)
                  coordinates_index.append("visit %s" %this_visit)

            
   return coordinates_index


##################################################
########### Functions analysis notebooks #########
##################################################

def plot_function(data_to_plot, title):
    
    def plotimg(idx):
        # Update histogram
        ax[0].clear()
        ax[0].hist(data_to_plot[int(idx)].flatten(), bins = len(np.unique(data_to_plot[int(idx)].flatten())))
        # Update image
        img.set_data(data_to_plot[int(idx),:,:])
        im = ax[1].imshow(data_to_plot[int(idx),:,:], origin='lower', cmap = 'viridis')
        ax[0].axvline(np.nanmedian(data_to_plot[int(idx),:,:]), c = 'r',  alpha = 0.5) # median pixel value
        ax[0].axvline(np.nanmean(data_to_plot[int(idx),:,:]), c = 'g', alpha = 0.5) # mean pixel value
        #plt.colorbar(im)
        fig.canvas.draw_idle()
        
    fig, ax = plt.subplots(ncols = 2, figsize=(12,4))
    img = ax[1].imshow(data_to_plot[0], origin='lower')
    #colorbar = plt.colorbar(img)
    ax[0].set_xlabel('Brightness')
    ax[0].set_ylabel('Nb of pixels')
    plt.suptitle(title, weight = 'bold')

    ipywidgets.interact(plotimg, idx = ipywidgets.FloatSlider(value=0,min=0,max=np.shape(data_to_plot)[0]-1,step=1))
    plt.show()
    
def apply_filters(data, filter_names, values, reverse_filters):
    
    # Available filters ['latitude-','latitude+','visit','density_cosmics','nb_cosmics','no_straylight','largest_cosmics','percentage_cosmics','percentage_cosmics_per_s', 'exp_time','n_exp']
    list_filter_names = ['time-','time+','latitude-','latitude+','longitude-','longitude-','visit','density_cosmics','nb_cosmics','no_straylight','largest_cosmics','percentage_cosmics','percentage_cosmics_per_s', 'exp_time', 'n_exp']
    
    filtered_data = data.copy()
    
    for filter_name, value, reverse in zip(filter_names,values, reverse_filters): 
        
        if reverse:
            inf = '>'
            sup = '<'
        else: 
            inf = '<'
            sup = '>'
            
        if filter_name == 'time-':
            print(f"Keep data {sup} {value}")
            filter_to_apply = filtered_data['time'] >= value
        elif filter_name == 'time+':
            print(f"Keep data {inf} {value}")
            filter_to_apply = filtered_data['time'] <= value
        elif filter_name == 'exp_time':
            print(f"Keep data with exposure time {inf} {value}")
            filter_to_apply = filtered_data['exp_time'] <= value
        elif filter_name == 'n_exp':
            print(f"Keep data with {inf} {value} n_exp (stacked images)")
            filter_to_apply = filtered_data['n_exp'] <= value
        elif filter_name == 'latitude-':
            print(f"Kepp data with latitude {sup} {value}")
            filter_to_apply = filtered_data['LATITUDE'] > value
        elif filter_name == 'latitude+':
            print(f"Keep with latitude {inf} {value}")
            filter_to_apply = filtered_data['LATITUDE'] < value  
        elif filter_name == 'longitude-':
            print(f"Kepp data with longitude {sup} {value}")
            filter_to_apply = filtered_data['LONGITUDE'] > value
        elif filter_name == 'longitude+':
            print(f"Keep with longitude {inf} {value}")
            filter_to_apply = filtered_data['LONGITUDE'] < value  
        elif filter_name == 'visit':
            print(f"Keep data only from {value}")
            filter_to_apply = filtered_data['visit_ID'] == value
        elif filter_name == 'density_cosmics':
            print(f"Keep data only with a density of cosmics {sup} {value}")
            filter_to_apply = filtered_data['density_cosmics'] > value        
        elif filter_name == 'nb_cosmics':
            print(f"Keep data only with a number of cosmics {sup} {value}")
            filter_to_apply = filtered_data['nb_cosmics'] > value        
        elif filter_name == 'no_straylight':
            if reverse:
                print(f"Keep data only affected with straylight")
            else:
                print(f"Keep data only not affected with straylight")
            filter_to_apply = filtered_data['straylight_boolean']
            filter_to_apply = ~filter_to_apply
        elif filter_name == 'largest_cosmics':
            print(f"Keep data only with a largest cosmic {inf} {value}")
            filter_to_apply = filtered_data['largest_cosmics'] < value
        elif filter_name == 'percentage_cosmics':
            print(f"Keep data images with {inf} {value}% of pixels affected by cosmics")
            filter_to_apply = filtered_data['percentage_cosmics'] < value
        elif filter_name == 'percentage_cosmics_per_s':
            print(f"Keep data images with {inf} {value}% of pixels affected by cosmics")
            filter_to_apply = filtered_data['percentage_cosmics_per_s'] < value
        else:
            print(f"{filter_name} not in {list_filter_names}")   
            
        # apply filter 
        if reverse:
            filter_to_apply = ~filter_to_apply

        nb_points = len(filtered_data)
        filtered_data = filtered_data[filter_to_apply]
        nb_datapoints_remomoved =  nb_points - len(filtered_data)
        print(f"Removed {nb_datapoints_remomoved} data points, kept {len(filtered_data)}")
        
    return filtered_data

def bin_data(x,y,c,interp_grid_size = 2.5, type = None):
    
    # Bin and maks SAA mask contour
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -90, 90

    # interpolation can be 'None, 'base_grid' or 'fine_grid'
    
    from scipy.interpolate import RBFInterpolator
    bin_size = 5
    x_bins = np.arange(lon_min, lon_max+bin_size, bin_size)
    y_bins = np.arange(lat_min, lat_max+bin_size, bin_size)
    
    # bin
    ret = binned_statistic_2d(x, y, c, statistic='median', bins=[x_bins, y_bins])

    # Get the array of bin values and the bin edges
    statistic = ret.statistic.T
    x_edges = ret.x_edge
    y_edges = ret.y_edge

    # Compute the bin centers
    bin_centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    bin_centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(bin_centers_x, bin_centers_y)

    # Flatten the arrays for easier indexing
    points = np.column_stack([X.ravel(), Y.ravel()])
    values = statistic.ravel()

    # Remove NaN values from the points and values
    mask = ~np.isnan(values)
    valid_points = points[mask]
    valid_values = values[mask]
    
    

    # # Define a finer grid for finer interpolation
    # nb_points_finer_grid = 200
    finer_x = np.arange(lon_min-interp_grid_size, lon_max+interp_grid_size, interp_grid_size)
    finer_y = np.arange(lat_min-interp_grid_size, lat_max+interp_grid_size, interp_grid_size)
    Finer_X, Finer_Y = np.meshgrid(finer_x, finer_y)
    # finer_x = np.linspace(bin_centers_x.min(), bin_centers_x.max(), nb_points_finer_grid)  # More points
    # finer_y = np.linspace(bin_centers_y.min(), bin_centers_y.max(), nb_points_finer_grid)  # More points
    # Finer_X, Finer_Y = np.meshgrid(finer_x, finer_y)

    # # Flatten the finer grid for interpolation
    finer_points = np.column_stack([Finer_X.ravel(), Finer_Y.ravel()])

    # Use the same valid_points and valid_values from the previous example
    interpolator = RBFInterpolator(valid_points, valid_values, kernel='linear', smoothing = 0)

    # print(f"Interpolating on the {bin_size} degrees grid")
    # values = interpolator(points)
    # statistic = values.reshape(X.shape)
    # lon_mesh = X
    # lat_mesh = Y
    # elif interpolation == 'fine_grid':
    print(f"Interpolating on a {interp_grid_size} grid")
    finer_values = interpolator(finer_points)
    statistic = finer_values.reshape(Finer_X.shape)
    lon_mesh = Finer_X
    lat_mesh = Finer_Y
    if type == 'density_cosmics':
        # set low values to 1
        mask_low_values = statistic < 1
        statistic[mask_low_values] = 1
    elif (type == 'percentage_cosmics') or (type == 'percentage_cosmics_per_s'):
        # set low values to 0.01
        mask_low_values = statistic < 0.01
        statistic[mask_low_values] = 0.01
    elif 'SAA':
        return interpolator
    else:
        raise ValueError("Please set type to convert low values, conversion ignored")
        
    return statistic, bin_centers_x, bin_centers_y, lon_mesh, lat_mesh

def circle_points(lat, lon, radius, num_points=100):
    """
    Calculate the latitude and longitude points that form a circle of given radius
    centered on a given latitude and longitude.

    :param lat: Latitude of the center in degrees
    :param lon: Longitude of the center in degrees
    :param radius: Radius of the circle in degrees (approx. for small circles)
    :param num_points: Number of points to generate along the circle
    :return: Two numpy arrays (lats, lons) representing the circle's latitude and longitude points
    """
    # Convert radius from degrees to radians
    radius_rad = np.deg2rad(radius)

    # Generate equally spaced angles around the circle
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the latitude and longitude points
    latitudes = np.arcsin(np.sin(np.deg2rad(lat)) * np.cos(radius_rad) +
                          np.cos(np.deg2rad(lat)) * np.sin(radius_rad) * np.cos(angles))
    
    longitudes = np.deg2rad(lon) + np.arctan2(np.sin(angles) * np.sin(radius_rad) * np.cos(np.deg2rad(lat)),
                                               np.cos(radius_rad) - np.sin(np.deg2rad(lat)) * np.sin(latitudes))
    
    # Convert the latitude and longitude from radians to degrees
    latitudes = np.rad2deg(latitudes)
    longitudes = np.rad2deg(longitudes)
    
    sort_lon = np.argsort(longitudes)

    return latitudes[sort_lon], longitudes[sort_lon]

def get_SAA_mask(file):

    SAA_file = Path("ref_files") / file # post-LTAN

    data_SAA = read_SAA_map(SAA_file)

    # Plot SAA mask
    lon_SAA1D = data_SAA['longitude']
    lat_SAA1D = data_SAA['latitude']
    SAA_mask1D = data_SAA['SAA_FLAG']
    
    # Get unique sorted longitude and latitude values
    lon_SAA2D = np.sort(np.unique(lon_SAA1D))
    lat_SAA2D = np.sort(np.unique(lat_SAA1D))

    # Reshape `values` to fit the grid (assuming it's ordered in row-major order)
    SAA_mask2D = SAA_mask1D.values.reshape(len(lat_SAA2D), len(lon_SAA2D))


    # # Bin and maks SAA mask contour
    lon_min, lon_max = -180, 180
    lat_min, lat_max = -90, 90
        
    # SAA_map_bins_lon = 3
    # SAA_map_bins_lat = 2
    # x_bins_SAA = np.arange(lon_min + SAA_map_bins_lon, lon_max,SAA_map_bins_lon)
    # y_bins_SAA = np.arange(lat_min + SAA_map_bins_lat, lat_max,SAA_map_bins_lat)

    # #SAA_masked_binned = binned_statistic_2d(x, y, c, statistic='median', bins=[x_bins_SAA, y_bins_SAA]).statistic.T
    # lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # # Interpolate data onto the grid
    # from scipy.interpolate import griddata
    # value_grid = griddata((lon, lat), SAA_mask, (lon_mesh, lat_mesh), method='nearest')

    return SAA_mask2D, lat_SAA2D, lon_SAA2D, SAA_mask1D, lat_SAA1D, lon_SAA1D, lat_min, lat_max, lon_min, lon_max
    # return value_grid, lat_mesh, lon_mesh, lat_min, lat_max, lon_min, lon_max

def latlon_to_cartesian(lat, lon, radius=6378000):
    lat_radians = np.radians(lat)
    lon_radians = np.radians(lon)

    x = radius * np.cos(lat_radians) * np.cos(lon_radians)
    y = radius * np.cos(lat_radians) * np.sin(lon_radians)
    z = radius * np.sin(lat_radians)
    
    return np.stack([x,y,z], axis=-1)

def bin_data_polar(x,y,c,interp_grid_size = 2.5, lon_min = None, lon_max = None, lat_min = None, lat_max = None, type = None,):
    from scipy.interpolate import RBFInterpolator

    # set bins and mask SAA contour
    if lon_min is None and lon_max is None:
        lon_min, lon_max = -180, 180
    else:
        lon_min = lon_min
        lon_max = lon_max

    if lat_min is None and lat_max is None:
        lat_min, lat_max = -90, 90
    else:
        lat_min = lat_min
        lat_max = lat_max

    bin_size = 5
    x_bins = np.arange(lon_min, lon_max + bin_size, bin_size)
    y_bins = np.arange(lat_min, lat_max + bin_size, bin_size)
    
    # bin data
    ret = binned_statistic_2d(x,y,c, statistic='median', bins=[x_bins, y_bins])
    statistic = ret.statistic.T
    x_edges = ret.x_edge
    y_edges = ret.y_edge

    # compute bin centers
    bin_centers_x = (x_edges[:-1] + x_edges[1:]) / 2
    bin_centers_y = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(bin_centers_x,bin_centers_y)

    # flatten the arrays
    points_lonlat = np.column_stack([X.ravel(), Y.ravel()])
    values = statistic.ravel()

    # remove NaNs
    mask = ~np.isnan(values)
    valid_points_lonlat = points_lonlat[mask]
    valid_values = values[mask]

    # convert valid lon/lat points into 3D cartesian
    valid_lats = valid_points_lonlat[:, 1]
    valid_lons = valid_points_lonlat[:, 0]
    valid_points_cartesian = latlon_to_cartesian(valid_lats, valid_lons)

    # define finer interpolation grid
    finer_x = np.arange(lon_min, lon_max + interp_grid_size, interp_grid_size)
    finer_y = np.arange(lat_min, lat_max + interp_grid_size, interp_grid_size)
    Finer_X, Finer_Y = np.meshgrid(finer_x, finer_y)

    # convert finer interpolation grid to 3D cartesian
    finer_lats = Finer_Y.ravel()
    finer_lons = Finer_X.ravel()
    finer_cartesian = latlon_to_cartesian(finer_lats, finer_lons)


    # RBF interpolation in cartesian 3D
    interpolator = RBFInterpolator(valid_points_cartesian, valid_values, kernel='linear', smoothing =0)
    finer_values = interpolator(finer_cartesian)
    interpolated = finer_values.reshape(Finer_X.shape)

    lon_mesh, lat_mesh = Finer_X, Finer_Y

    if type == 'density_cosmics':
        # set low values to 1
        mask_low_values = interpolated < 1
        interpolated[mask_low_values] = 1
    elif type in ('percentage_cosmics', 'percentage_cosmics_per_s'):
        # set low values to 0.01
        mask_low_values = interpolated < 0.01
        interpolated[mask_low_values] = 0.01
    elif 'SAA' in (type or ''):
        return interpolator
    else:
        raise ValueError("Please set type to convert low values, conversion ignored")
    

    return interpolated, bin_centers_x, bin_centers_y, lon_mesh, lat_mesh

def wrap_longitude(longitude):
    """
    Function to ensure that the longitude coordinates are wrapped to [-180, 180].
    """
    if longitude < -180:
        return longitude + 360
    elif longitude > 180:
        return longitude - 360
    else:
        return longitude

def wrap_latitude(latitude):
    """
    Function to ensure that the latitude coordinates are wrapped to [90, 90].
    """

    if latitude < -90:
        return latitude + 180
    elif latitude > 90:
        return latitude - 180
    else:
        return latitude

def area_in_kilometres(paths, radius):
    """
    Function to compute the spherical area of a polygon.
    """
    geod = Geodesic(radius, 0) #spherical model with no flattening
    poly = geod.Polygon()
    area_individual_polygons = []
    perimeter_individual_polygons = []

    for path in paths:
        if path is not None:
            total_area_km = 0.0
            vertices = path.vertices
            wrapped_vertices = [(wrap_longitude(lat), wrap_latitude(lon)) for lat, lon in vertices]
            codes = path.codes
            indices = np.where(codes == PATH.MOVETO)[0]
            vertices_segments = np.split(wrapped_vertices, indices)[1:]
        #vertices_segments = np.split(vertices, indices)[1:]
        # code_segments = np.split(codes, indices)[1:]

        # for codes, vertices in zip(code_segments, vertices_segments):
        #     #print(vertices)
        #     poly.Clear()
        #     for pts in vertices[:-1]:
        #         poly.AddPoint(pts[1], pts[0])

            for vertices in vertices_segments:
                poly.Clear()
                for pts in vertices[:-1]:
                    poly.AddPoint(pts[1], pts[0])

            
                _, perimeter, area = poly.Compute()
                perimeter = perimeter/1e3
                perimeter_individual_polygons.append(perimeter)
                area_individual_polygons.append(area)
            # print(area_individual_polygons)
            # print(perimeter_individual_polygons)

            #print(area_individual_polygons)
            area_individual_polygons = np.array(area_individual_polygons)
            area_individual_polygons = -np.sort(-area_individual_polygons)

            if len(area_individual_polygons) > 0:
                first_item = area_individual_polygons[0]
                total_area_km = area_individual_polygons[0]
                for i in range(1, len(area_individual_polygons)):
                    if area_individual_polygons[i] > 0.1*first_item or area_individual_polygons[i] == first_item:
                        total_area_km += area_individual_polygons[i]
                    else:
                        total_area_km -= area_individual_polygons[i]

            return perimeter_individual_polygons, total_area_km/1e6
        else:
            return 0.0

def area_in_degrees(paths):
    """
    Function to compute the spherical area of polygon.
    """
    area_individual_polygons = []

    for path in paths:
        if path is not None:
            vertices = path.vertices
            codes = path.codes
            indices = np.where(codes == PATH.MOVETO)[0]
            vertices_segments = np.split(vertices, indices)[1:]
            # code_segments = np.split(codes, indices)[1:]

            # for codes, vertices in zip(code_segments, vertices_segments):
            #         area = Polygon(vertices[1:]).area
            #         area_individual_polygons.append(area)

            for vertices in vertices_segments:
                area = Polygon(vertices[1:]).area
                area_individual_polygons.append(area)

        
            area_individual_polygons = np.array(area_individual_polygons)
            area_individual_polygons = -np.sort(-area_individual_polygons)
            
            total_area_deg = 0
            if len(area_individual_polygons) > 0:
                first_item = area_individual_polygons[0]
                total_area_deg = area_individual_polygons[0]
                for i in range(1, len(area_individual_polygons)):
                        if area_individual_polygons[i] > 0.1*first_item or area_individual_polygons[i] == first_item:
                            total_area_deg += area_individual_polygons[i]
                        elif area_individual_polygons[i] == 0:
                            total_area_deg = 0
                        else:
                            total_area_deg -= area_individual_polygons[i]


            return total_area_deg

        else:
            return 0.0

def location_change(paths):
    for path in paths:
        vertices = path.vertices
        lon = vertices[:,0]
        lat = vertices[:,1]
  
        # measure change in location
        north = np.max(lat) if lat.size > 0 else 0
        south = np.min(lat) if lat.size > 0 else 0
        east = np.max(lon) if lon.size > 0 else 0
        west = np.min(lon) if lon.size > 0 else 0
        coords = (north, east, south, west)

    return coords