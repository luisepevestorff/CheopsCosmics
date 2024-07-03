import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import os
import cv2
from pathlib import Path
from sqlalchemy import text

MAIN_PATH = Path.cwd() / "Pipeline_new/SAA_ORs_reduction_pipeline_v_0.1.0/SAA_ORs_reduction_pipeline_v_0.1.0"

def get_files_with_substring(directory):
    
    image_files_list = []
    roll_angle_files_list = []      
    
    for filename in directory.iterdir():
        if 'RAW_SubArray' in filename.stem:
            image_files_list.append(filename)
        elif ('Attitude' in filename.stem) or ('COR_Lightcurve-DEFAULT' in filename.stem):
            roll_angle_files_list.append(filename)
        else:
            raise NameError(f"{filename} is an unexpected file that does not contain 'RAW_SubArray', 'Attitude', or 'COR_Lightcurve-DEFAULT.")
    return image_files_list, roll_angle_files_list

def read_images(file):
    
    metadata_idx = 9
    
    hdu = fits.open(file)
    images = hdu[1].data
    header = hdu[1].header
    metadata = hdu[metadata_idx].data
    hdu.close()

    nb_images = np.shape(images)[0]    
    height_images = np.shape(images)[1]
    width_images = np.shape(images)[2]

    
    return images, header, metadata, nb_images, height_images, width_images

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

def create_contaminant_mask(derotated_openCV_images, edges_mask, saa_or_science, enlarge_mask = True, inspect_threshold = False, inspect_mask = False, threshold = 0, inspect_hist = False):
    
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

        # plt.show()
        
        
    # if inspect_hist:

    #     x_hist = find_threshold(image_for_mask)[4]
    #     params = find_threshold(image_for_mask)[3]
    #     gauss = gaussian(x_hist, *params)

    #     fig,ax = plt.subplots()
    #     ax.hist(image_for_hist.flatten(), bins = 1000)
    #     ax.plot(x_hist, gauss, 'r', label='Gaussian Fit')
    #     plt.title('inspect hist')
    #     plt.show()

    

    return final_mask, threshold
    
def detect_cosmics(masked_images, threshold): 

    # convert masked images to binary
    binary_images = masked_images.copy()
    for i,image in enumerate(binary_images):
        _, img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary_images[i] = img
        
    # Initialize the counter of cosmics per image
    nb_cosmics = np.zeros(np.shape(masked_images)[0])
    images_contours = [] # List of contours for all images (size of nb of images)

    # 1. Find the countours
    # 2. Go through all contours in images, and remove the ones that are only one pixels
    # 3. Store the remaining in the images_contours
    for i,image in enumerate(binary_images):
        curr_image_contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        indices_of_1_pixel_contours = []
        contours_list = list(curr_image_contours)
        for j,contour in enumerate(contours_list): # if there is only one pixel, remove this contour 
            nb_pixels_in_contour = len(contour)
            if (nb_pixels_in_contour == 1):
                indices_of_1_pixel_contours.append(j)
        for index in sorted(indices_of_1_pixel_contours, reverse=True):
            del contours_list[index]
        nb_cosmics[i] = len(contours_list)
        images_contours.append(contours_list)
        
    return binary_images, nb_cosmics, images_contours

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

def database_query(start_time, end_time, copy):
    
    from pymd import PyMD
    from pymd.model import Visit
    
    if copy:
        import subprocess 
        # copy the database here
        print('\n Copying db ...')
        command = 'scp chps_ops@chpscn02:/opt/monitoring_dashboard_database/monitoring_dashboard.db .'
        subprocess.run(command, shell=True)
        database= Path.cwd() / "monitoring_dashboard.db"
    else: 
        database=Path("/projects/astro/cheops/processing/chpscn02/opt/monitor4cheops/Operations/monitoring_dashboard.db")
        
    
    print(f'\n Getting all visits ID (except M&C) from {start_time} to {end_time} ...')

    # format time 
    filename_format = "%Y-%m-%d %H:%M:%S.000"
    start_time_str = start_time.strftime(filename_format)
    end_time_str = end_time.strftime(filename_format)

    db = PyMD(name=database)
    visit_id_list = []

    ## QUERY for 
    with db.session() as session:
            sql = f"""
                select * 
                from visit 
                where 
                    start_time > '{start_time_str}'
                    and start_time < '{end_time_str}'
                    and visit_counter is not null 
                    and ((programme_type >= 10 and programme_type < 40) 
                        or (programme_type >=40 and programme_type < 50))"""
            for row in session.execute(text(sql)):
                visit_id_list.append(row.formatted_visit_id)
                #target_name_list.append(row.target_name)
                #exptime_list.append(row.exptime)
            
            # Alternatively:

            # query = session.query(Visit).filter(
            #       (Visit.exptime < 10) 
            #     & Visit.visit_counter.isnot(None)
            #     & (
            #          ((Visit.programme_type >= 10) & (Visit.programme_type < 30))
            #        | ((Visit.programme_type >= 40) & (Visit.programme_type < 50)))
            #     )
            # for row in query.all():
            #     print(row.formatted_visit_id)
            
            
    return visit_id_list      


def genreate_diagnostic_plots(derotated_openCV_images, images, subtracted_median_images, temporal_median_substracted_images, threshold_noise, threshold_cosmics, mask, masked_images, binary_images, n, nb_cosmics, title, plot_name, show_plot):
    
    plt.close('all')
    fig, ax = plt.subplots(ncols = 8, nrows=2, figsize=(35,8))
    plt.subplots_adjust(left=0.05, right=0.99)
    median_image = np.nanmedian(derotated_openCV_images, axis=(0))
    mean_image = np.nanmean(derotated_openCV_images, axis=(0))
    
    # ax[0,0].hist(np.log(images[n]+1).flatten(), bins = 100)
    # ax[1,0].imshow(np.log(images[n]+1), origin = "lower")
    # ax[0,0].set_title("Original subArray (log)")
    ax[0,0].hist(images[n].flatten(), bins = 100)
    ax[1,0].imshow(images[n], origin = "lower")
    ax[0,0].set_title("Original subArray")
    ax[0,0].set_ylim(0,100)
    ax[0,1].hist(subtracted_median_images[n].flatten(), bins = 100)#len(np.unique(subtracted_median_images[n].flatten())))
    ax[1,1].imshow(subtracted_median_images[n], origin = "lower")
    ax[0,1].set_title("median value removed")
    ax[0,1].set_ylim(0,100)
    ax[0,2].hist(temporal_median_substracted_images[n].flatten(), bins = 100)#len(np.unique(temporal_median_substracted_images[n].flatten())))
    ax[1,2].imshow(temporal_median_substracted_images[n], origin = "lower")
    ax[0,2].set_title("median of pixels removed")
    ax[0,2].set_ylim(0,100)
    # ax[0,3].hist(derotated_openCV_images[n].flatten(), bins = len(np.unique(derotated_openCV_images[n].flatten())))
    # ax[1,3].imshow(derotated_openCV_images[n], origin = "lower")
    # ax[0,3].set_title("derotated converted image")
    # ax[0,3].hist(np.log(derotated_openCV_images[n].flatten()+1), bins = len(np.unique(derotated_openCV_images[n].flatten())))
    ax[1,3].imshow(np.log(derotated_openCV_images[n]+1), origin = "lower")
    ax[0,3].set_ylim(0,500)
    ax[0,3].set_title("log(derotated converted image)")
    #ax[0,3].set_xscale('log')
    ax[0,4].hist(median_image.flatten(), bins = len(np.unique(median_image.flatten())))
    ax[0,4].axvline(threshold_noise, c = 'C2', linewidth = 2)
    ax[1,4].imshow(median_image, origin = "lower",norm=colors.LogNorm())
    ax[0,4].set_xlim(0,5*threshold_noise)
    ax[0,4].set_title("median image of visit")
    #ax[0,4].set_xscale('log')
    # ax[0,5].hist(mean_image.flatten(), bins = len(np.unique(mean_image.flatten())))
    # ax[0,5].axvline(threshold_noise, c = 'C2', linewidth = 2)
    # ax[1,5].imshow(mean_image, origin = "lower")
    # ax[0,5].set_ylim(0,100)
    # ax[0,5].set_title("mean image of visit")
    ax[1,5].imshow(mask, origin = "lower")
    ax[0,5].set_ylim(0,100)
    ax[0,5].set_title("designed mask")
    # ax[0,6].hist(np.log(masked_images[n]+0.1).flatten(), bins = len(np.unique(masked_images[n].flatten())))
    # ax[1,6].imshow(np.log(masked_images[n]+0.1), origin = "lower")
    # ax[0,6].axvline(np.log(threshold_cosmics+0.1), c = 'C2', linewidth = 2)
    # ax[0,6].set_title("log(masked image + 0.1)")
    ax[0,6].hist(masked_images[n].flatten(), bins = 100)# len(np.unique(masked_images[n].flatten())))
    ax[1,6].imshow(masked_images[n], origin = "lower")
    ax[0,6].axvline(threshold_cosmics, c = 'C2', linewidth = 2)
    ax[0,6].set_title("masked image")
    ax[0,6].set_xlim(0,5*threshold_cosmics)
    ax[0,6].set_ylim(0,100)
    ax[0,7].hist(binary_images[n].flatten(), bins = len(np.unique(binary_images[n].flatten())))
    ax[1,7].imshow(binary_images[n], origin = "lower")
    ax[0,7].text(0.25,0.9,f"{int(nb_cosmics)} detected cosmics", weight = 'bold', transform=ax[0,7].transAxes)
    ax[0,7].set_title("detected cosmics")
    fig.suptitle(title)
    savePath = MAIN_PATH / "output_plots" / plot_name
    plt.savefig(savePath, dpi = 600,format = 'png')
        
    if show_plot:
        plt.show()
        
def find_threshold(image, inspect_hist=False):
    images = image.copy()

    flatten_images = np.ndarray.flatten(images)
    flatten_images = flatten_images[flatten_images!= 0]
    #flattened_images = [i for i in images if (i >= 10) and (i <= 500)] #in case of trimming the data

    ### using norm function ##
    # mu, sigma = norm.fit(flatten_images)

    ### using scipy stats ###

    hist, bin_edges = np.histogram(flatten_images, bins=1000)
    #hist=hist/np.sum(hist)

    bin_treshold = 20 # threshold for minimum bin population

    n = len(hist)

    #x = np.linspace(10,500,1000)
    x = np.zeros((n),dtype=float) 
    for i in range(n):
        x[i] = (bin_edges[i+1]+bin_edges[i])/2

    
    hist_where = np.where(hist >= bin_treshold)
    x = x[hist_where]
    y = hist[hist_where]

    mu_initial = x[np.argmax(y)]              
    sigma_initial = np.std(flatten_images)
    amp_initial = np.max(y)

    popt,pcov=curve_fit(gaussian,x,y,p0=[amp_initial, mu_initial, sigma_initial])

    mu_fit =  popt[1]
    sigma_fit = popt[2]

    threshold = mu_fit + (5*sigma_fit)

    if inspect_hist:
        x_hist = x
        params = popt
        gauss = gaussian(x_hist, *params)

        fig,ax = plt.subplots()
        ax.hist(flatten_images, bins = 1000)
        ax.plot(x_hist, gauss, 'r', label='Gaussian Fit')
        plt.title('Histogram inspection')
        plt.show()

    return mu_fit, sigma_fit, threshold, popt, x

    ### using an exponential fit ###

def gaussian(x,amp,mu,sigma):
    return amp*np.exp(-(x-mu)**2/2*sigma**2)

def exponential(x, m, t, b):
    return m*np.exp(-(t*x)+b)

def remove_straylight(masked_images):
    masked_array = masked_images.copy()

    #calculate median/mean of each image in array (background) -> to do: test if mean or median works better
    median_per_image = []
    mean_per_image = []
    for i in range(len(masked_array)):
        median_per_image.append(np.median(masked_array[i]))
        mean_per_image.append(np.mean(masked_array[i]))

    #calculate median/mean of new array
    median_new = np.median(median_per_image)
    mean_new = np.mean(median_per_image)

    background_threshold = median_new + 10 #arbitrary number for now -> see what will filter stray light the best

    #remove images above background threshold
    array_coordinates = np.where(median_per_image < background_threshold)

    array_removed_straylight =  masked_array[array_coordinates]
    return array_removed_straylight

