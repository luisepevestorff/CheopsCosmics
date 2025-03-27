import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
import fnmatch

def get_files_with_substring(directory, substring):
    matching_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, f"*{substring}*"):
                matching_files.append(os.path.join(root, filename))
    return matching_files

def read_imagettes(file, type):
    
    if type == 'SAA':
        metadata_idx = 2
    elif type == 'science':
        metadata_idx = 9
    else:
        raise('Please set type to either SAA or science')
    hdu = fits.open(file)
    images = hdu[1].data
    header = hdu[1].header
    metadata = hdu[metadata_idx].data
    hdu.close()
    
    nb_imagettes = np.shape(images)[0]
    height_imagettes = np.shape(images)[1]
    width_imagettes = np.shape(images)[2]

    return images, header, metadata, nb_imagettes, height_imagettes, width_imagettes

def read_attitude(file):
    
    hdu = fits.open(file)
    time = np.array(hdu[1].data['MJD_TIME'], '>f8').byteswap().newbyteorder()
    roll_angle = np.array(hdu[1].data['SC_ROLL_ANGLE'], '>f4').byteswap().newbyteorder()
    hdu.close()
    
    return time, roll_angle

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
    
    num_files=5 # number of files to check
    
    # Get file paths and their corresponding modification times
    file_times = [(os.path.join(folder_path, file), os.path.getmtime(os.path.join(folder_path, file))) for file in files]
    
    # Sort files by modification time in descending order
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)
    
    BadPixelMap = None
    DarkFrame = None
    
    for file in sorted_files[:num_files]:
        if ((BadPixelMap is None) & ('BadPixelMap' in file[0])):
                BadPixelMap = file[0]
        elif ((DarkFrame is None) & ('DarkFrame' in file[0])):
                DarkFrame = file[0]
        else:
            continue
        
    return BadPixelMap, DarkFrame

def get_edges_mask(imagette):
    
    edges_mask = imagette == 0
    
    return edges_mask

def subtract_temporal_median_image(imagettes):
    '''
    Compute the median for all pixels through the cube to get the median image and remove it from each imagette in the data cube
    '''
    median_subtracted_imagettes = imagettes.copy().astype('float64')

    median_image_per_pixel = np.nanmedian(median_subtracted_imagettes, axis=(0))
   

    # subtract median image from data cube
    #median_subtracted_imagettes = imagettes - median_image_per_pixel

    #subtract median image from each imagette in the data cube
    for i in range(len(median_subtracted_imagettes)):
        median_subtracted_imagettes[i] = median_subtracted_imagettes[i] - median_image_per_pixel

    median_subtracted_imagettes = np.abs(median_subtracted_imagettes)

    return median_subtracted_imagettes

def subtract_median_image(imagettes):
    median_imagettes = imagettes.copy().astype('float64')


    for i in range(len(imagettes)):
        mask_zeros = median_imagettes[i] != 0
        median_image = np.nanmedian(median_imagettes[i][mask_zeros])
        median_imagettes[i][mask_zeros] = median_imagettes[i][mask_zeros] - median_image


    median_imagettes = np.abs(median_imagettes)
    return median_imagettes

def convert_to_openCV(imagettes, nb_imagettes, height_imagettes, width_imagettes):

   # imagettes = imagettes.astype(np.uint8)

    openCV_imagettes = np.zeros((nb_imagettes, width_imagettes, height_imagettes), dtype=np.uint8)

    for i in range(len(imagettes)):
        #openCV_imagettes[i] = cv2.normalize(imagettes[i], None, 0, 255, cv2.NORM_MINMAX)
        ## Optionally, use cv2.convertScaleAbs() for quick conversion and scaling
        openCV_imagettes[i] = cv2.convertScaleAbs(imagettes[i])
        
    return openCV_imagettes

def derotate_images(attitude_file, imagettes, metadata_imagettes, nb_imagettes, height_imagettes, width_imagettes):
    
    imagettes = imagettes.astype(np.uint8)

    # Science visit
    time_attitude, roll_angle = read_attitude(attitude_file)
    
    # # Find the roll_angle associated with each images
    time_imagettes = metadata_imagettes['MJD_TIME']
    closest_time_imagettes = find_closest_indices(time_imagettes, time_attitude)
    roll_angle_imagettes = roll_angle[closest_time_imagettes]
    
    # Find the roll_angle associated with each images
    time_imagettes_utc = pd.to_datetime(np.array(metadata_imagettes['UTC_TIME']), format='%Y-%m-%dT%H:%M:%S.%f', utc=True)
    time_imagettes_utc_jd = time_imagettes_utc.to_julian_date()
    # closest_time_imagettes = find_closest_indices(metadata_imagettes['MJD_TIME'], time_attitude)
    # roll_angle_imagettes = roll_angle[closest_time_imagettes]

    derotated_openCV_imagettes = np.zeros((nb_imagettes, width_imagettes, height_imagettes), dtype=np.uint8)
    first_image_angle = roll_angle_imagettes[0]

    # Derotate each image
    for i,image in enumerate(imagettes):
        rotate_by = -(roll_angle_imagettes[i] - first_image_angle)
        rotation_matrix = cv2.getRotationMatrix2D((width_imagettes / 2, height_imagettes / 2), rotate_by, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width_imagettes, height_imagettes))
        derotated_openCV_imagettes[i] = rotated_image
        
    # return derotated_openCV_imagettes, time_imagettes
    
    return derotated_openCV_imagettes, time_imagettes_utc, time_imagettes_utc_jd

def apply_mask_to_imagettes(imagettes, mask, value):
    
    masked_imagettes = imagettes.copy()
    
    for i in range(len(masked_imagettes)):
        masked_imagettes[i][mask] = value

    print('apply mask shape ', masked_imagettes.shape)
        
    return masked_imagettes  

def create_contaminant_mask(derotated_openCV_imagettes, threshold, edges_mask, enlarge_mask = True, inspect_threshold = False, inspect_mask = False):
   
   
    # Determine if we use the median or the mean to design the mask
    
    # For imagettes with no contaminant, the median will be 0 for all pixels
    
    median_imagette = np.nanmedian(derotated_openCV_imagettes, axis=(0)) # median
    mean_imagette = np.nanmean(derotated_openCV_imagettes, axis=(0)) # mean

    
    # unique_values = np.unique(median_imagette)
    # if (len(unique_values) == 1) & (np.unique(unique_values[0]) == 0): # If the only value is 0 for all pixels of the image
    #     image_for_mask = median_imagette
    #     med_or_mean = 'median'
    # else:
    #     image_for_mask = median_imagette # Mean is better suited to capture the uneven rotation of the PSF in order to build the mask
    #     med_or_mean = 'mean'
    
    ### Create the mask on the mean/median image
    
    if any(median_imagette.flatten() > 25): # If no pixel is higher than 10, there are no contaminants
        image_for_mask = mean_imagette
        mask = image_for_mask > threshold
        med_or_mean = 'mean'
    else:     
        threshold = 25
        image_for_mask = median_imagette
        mask = image_for_mask > threshold
        med_or_mean = 'median'
    
    mask = mask.astype(np.uint8)
    
    final_mask = mask
    
    if enlarge_mask:
        # Define the kernel for dilation
        kernel = np.ones((7,7), dtype=np.uint8)
        extension_mask = 2
        enlarged_mask = cv2.dilate(mask, kernel, iterations= extension_mask) == 1
        final_mask = enlarged_mask
    
    if (med_or_mean == 'median') & (inspect_mask):
        print('No contaminant found, mask is empty, going straight to cosmic rays detection')
        plt.figure()
        plt.imshow(final_mask)
        plt.title('Mask')
        plt.show()
        return final_mask
    
                
    if inspect_threshold:
        
        hist_median_imagette = image_for_mask.copy()
        
        # Correct edge to 255
        hist_median_imagette[edges_mask] = 255
        all_pixels_median = hist_median_imagette.flatten()
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
        ax[1].axvline(threshold, c = 'C1', label='Threshold')
        ax[1].legend()
        ax[1].set_title('Histogram of the pixel values', weight = 'bold')
        ax[2].plot(sorted_data, cumulative)
        ax[2].axvline(threshold, c = 'C1', label='Threshold')
        ax[2].legend()
        ax[2].set_title('CDF of the pixel values', weight = 'bold')
                
        plt.show()
            
    if inspect_mask:
        
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
        
        
    # Count the number of pixel that are neither in the mask, neither on the edges. I.E., the pixels used for detection
    
    #get_edges_mask(imagette)

    return final_mask
    
def detect_cosmics(masked_imagettes, threshold): 

    # convert masked imagettes to binary
    binary_images = masked_imagettes.copy()
    for i,image in enumerate(binary_images):
        _, img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary_images[i] = img
        
    # Initialize the counter of cosmics per image
    nb_cosmics = np.zeros(np.shape(masked_imagettes)[0])
    imagettes_contours = [] # List of contours for all imagettes (size of nb of imagettes)

    # 1. Find the countours
    # 2. Go through all contours in imagettes, and remove the ones that are only one pixels
    # 3. Store the remaining in the imagettes_contours
    for i,image in enumerate(binary_images):
        curr_imagette_contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        indices_of_1_pixel_contours = []
        contours_list = list(curr_imagette_contours)
        for j,contour in enumerate(contours_list): # if there is only one pixel, remove this contour 
            nb_pixels_in_contour = len(contour)
            if (nb_pixels_in_contour == 1):
                indices_of_1_pixel_contours.append(j)
        for index in sorted(indices_of_1_pixel_contours, reverse=True):
            del contours_list[index]
        nb_cosmics[i] = len(contours_list)
        imagettes_contours.append(contours_list)
        
    return binary_images, nb_cosmics, imagettes_contours

def reshape_flatten_images(imagettes_type, images):
    
    size = int(np.sqrt(len(images['raw_imagettes'].iloc[0])))
    reshaped_masked_imagettes = np.zeros((len(images),size,size))
    imagettes = images[imagettes_type].values
    for i in range(len(imagettes)):
        reshaped_masked_imagettes[i] = imagettes[i].reshape(size,size)
    
    return reshaped_masked_imagettes 
        
def unfold_interp_fold(data_RES, full_table, order = 3):
    
    """
    Unfold, interpolate, and fold back longitude values to ensure continuity.
    
    Parameters:
        RES_ORB_reindexed (pandas.DataFrame): DataFrame containing reindexed orbital data with 'JD' (Julian Date),
                                              'LATITUDE', and 'LONGITUDE' columns.
        data_RES (pandas.DataFrame): DataFrame with original 'LATITUDE' and 'LONGITUDE' data.
        imagettes (pandas.DataFrame): DataFrame the imagettes.
        order (int, optional): The order of the interpolation spline. Defaults to 3.
    
    Returns:
        pandas.DataFrame: DataFrame with unfolded, interpolated, and folded back longitude values. Merged with imagettes data.
        
    This function takes orbital data, unfolds discontinuous longitude values, interpolates the latitude and longitude 
    values using spline interpolation, folds back the longitude values to the range [-180, 180] degrees, and then merges
    the data with additional imagettes information. The unfolded and interpolated data, along with imagettes details,
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

def remove_stray_light(mask):
    '''
    mask: masked imagette
    '''

    masked_imagettes = mask.copy()

    imagettes_wo_zero = masked_imagettes[np.nonzero(masked_imagettes)]

    median_image = np.nanmedian(imagettes_wo_zero)

    for i in range(len(mask)):
        masked_imagettes[i] - masked_imagettes[i] - median_image
    
    masked_imagettes = np.abs(masked_imagettes)

    return masked_imagettes

