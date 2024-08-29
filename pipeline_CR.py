import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from functions import *
from datetime import datetime
from pathlib import Path
import socket
# # Main loop of entire process
   
MAIN_PATH = Path.cwd()

def main_loop(Images, roll_angle_file, threshold_noise, threshold_cosmics, type_of_visit, generate_plots):

    ### Get images and metadata ###
    images_orig, header_images, metadata_images, nb_images, height_images, width_images = read_images(Images)
    
    # for testing indivisual visits
    #if Images == mainPath / "CH_PR149002_TG001301_TU2024-05-27T12-03-30_SCI_RAW_SubArray_V0300.fits":
    #    print('s')
    
    if nb_images < 10:
        print("This visit has less than 10 images, skipping...")
        return pd.DataFrame()
    
    # visit params
    id =  str(header_images['PROGTYPE']) + '_' + str(header_images['PROG_ID']) + '_' + str(header_images['REQ_ID']) + '_' + str(header_images['VISITCTR'])
    visit_start = header_images['T_STRT_U']
    visit_end = header_images['T_STOP_U']
    n_exp = header_images['NEXP']    
    exp_time = header_images['EXPTIME']
    total_exp_time = header_images['TEXPTIME']
    los_to_sun = metadata_images['LOS_TO_SUN_ANGLE']
    los_to_moon = metadata_images['LOS_TO_MOON_ANGLE']
    los_to_earth = metadata_images['LOS_TO_EARTH_ANGLE']
    
    if type_of_visit == 'science':
        time_images_utc = pd.to_datetime(np.array(metadata_images['UTC_TIME']), format='%Y-%m-%dT%H:%M:%S.%f', utc=True)
        time_images_utc_jd = time_images_utc.to_julian_date()

    # star params
    target_name = header_images['TARGNAME']
    mag_G = header_images['MAG_G']
    ra = header_images['RA_TARG']
    dec = header_images['DEC_TARG']
    
    print(f"{target_name} (OR {id}): {visit_start}, {np.shape(images_orig)[0]} images, stacking order = {n_exp}")
    
    size = np.shape(images_orig[0])[0]
    radius = np.shape(images_orig[0])[0]/2
    edges_circular_mask = create_circular_mask(size, radius)
    
    # Apply circular mask
    images = apply_mask_to_images(images_orig, edges_circular_mask, 0)
    
     ### Subtract the median image ###
    subtracted_median_images = subtract_median_image(images, edges_circular_mask)

    ### Subtract the temporal median image ###
    temporal_median_substracted_images = subtract_temporal_median_image(subtracted_median_images)

    ### Derotate images ###    
    if type_of_visit == 'science':
        derotated_openCV_images = derotate_images(roll_angle_file, temporal_median_substracted_images, nb_images, height_images, width_images)
    elif type_of_visit == 'SAA': 
        # For the SAA, we take attitiue files to get the roll angle as the SCI_COR_Lightcurve is not always available 
        derotated_openCV_images, time_images_utc, time_images_utc_jd, _ = derotate_SAA(roll_angle_file, temporal_median_substracted_images, metadata_images, nb_images, height_images, width_images)
    else:
        raise ("Type of visit must be either 'subarray' or 'imagette'")

    ### Design mask for contaminant ###
    enlarge_mask = True
    inspect_threshold = False
    inspect_mask = False
    inspect_hist = False

     
    # Correct the threshold using the stacking order
    threshold_noise *= np.sqrt(n_exp)
    # print(threshold_noise)
    
    mask, _ = create_contaminant_mask(derotated_openCV_images, edges_circular_mask, type_of_visit, enlarge_mask, inspect_threshold, threshold_noise) # We keep the threshold calculated from the noise above. If the gaussian fit is activated, then the value passed here will be replaced in the function 

    ### Apply mask ###
    masked_images = apply_mask_to_images(derotated_openCV_images, mask, 0)
      
    
    ### Remove images with stray light ###

    straylight_boolean = remove_straylight(masked_images) # straylight images flagges as FALSE, all others as TRUE

    # straylight_only = masked_images[~straylight_boolean] # returns straylight only images
    # images_wo_straylight = masked_images[straylight_boolean] # returns all images without strayligth
    
    # nb_of_removed_images_straylight = nb_images - len(masked_images) # nb_images is the original number of images

    # print(f"{nb_of_removed_images_straylight} images with straylight were removed.")

    ### Detect cosmics ###
    binary_images, nb_cosmics, images_contours, nb_pixels_largest_cosmics = detect_cosmics(masked_images, threshold_cosmics) 
           
    if generate_plots:
        n = 13
        nb_cosmics_plot = nb_cosmics[n]
        title_plot = f"{id}, {target_name}, nexp: {n_exp}, exptime:{exp_time}, Texptime: {total_exp_time} "
        name_plot = f"SAA_visit_{id}_frame_{n}.png" if visit_type == 'SAA' else f"visit_{id}_frame_{n}.png"
        genreate_diagnostic_plots(derotated_openCV_images, images, subtracted_median_images, temporal_median_substracted_images, threshold_noise, threshold_cosmics, mask, masked_images, binary_images, n, nb_cosmics_plot, title_plot, name_plot, show_plot = False)
    
    # Cosmic per cm2
    nb_masked_pixels = np.sum(edges_circular_mask | mask) # number of pixels that are masked (edged + mask)
    fraction_remaining_pixels = (height_images*width_images) - nb_masked_pixels
    pixel_size = 13e-6 # m
    cm2_analysed = fraction_remaining_pixels*((pixel_size*1e2)**2) # cm2
    density_cosmics = nb_cosmics/cm2_analysed/total_exp_time # nb cosmics/cm2/sec
    
    print(f'{nb_masked_pixels} masked pixels')
    
    # Quantize images to take less space
    
    
    flattened_images           = [image.flatten().astype('uint8') for image in images_orig]
    flattened_derotated_images = [image.flatten().astype('uint8') for image in derotated_openCV_images]
    flattened_masked_images    = [image.flatten().astype('uint8') for image in masked_images]
    flattened_binary_images    = [image.flatten().astype('uint8') for image in binary_images]
    
    
    threshold_cosmics = threshold_cosmics*255/(65535*n_exp)
 
   
    flattened_mask = []
    for i in range(len(images)):
        flattened_mask.append(mask.flatten())
        
    latitude = [metadata['LATITUDE'] for metadata in metadata_images]
    longitude = [metadata['LONGITUDE'] for metadata in metadata_images]


    data = pd.DataFrame(data =    {
                                'visit_ID': np.full(nb_images, id),
                                'img_counter': np.arange(nb_images),
                                # 'raw_images': flattened_images,
                                # 'derotated_images': flattened_derotated_images,
                                'masked_images': flattened_masked_images, 
                                #'binary_images': flattened_binary_images,
                                #'mask': flattened_mask,
                                'JD': time_images_utc_jd,
                                'time': time_images_utc,
                                'nb_cosmics' : nb_cosmics.astype(int),
                                'largest_cosmics': nb_pixels_largest_cosmics,
                                'density_cosmics' : density_cosmics,
                                'pix_cosmics': images_contours,
                                'nb_masked_pixels': nb_masked_pixels.astype(int),
                                'im_height': np.full(nb_images, height_images),
                                'im_width': np.full(nb_images, width_images),
                                'threshold_cosmics': np.full(nb_images, threshold_cosmics),
                                'n_exp': np.full(nb_images, n_exp),
                                'exp_time': np.full(nb_images, exp_time),
                                'los_to_sun': np.full(nb_images, los_to_sun),
                                'los_to_moon': np.full(nb_images, los_to_moon),
                                'los_to_earth': np.full(nb_images, los_to_earth),
                                'target_name': np.full(nb_images, target_name),
                                'mag_G': np.full(nb_images, mag_G),
                                'ra': np.full(nb_images, ra),
                                'dec': np.full(nb_images, dec),
                                'LATITUDE': latitude,
                                'LONGITUDE': longitude,
                                'straylight_boolean': straylight_boolean
                                }
    )
    
    data.set_index('JD',drop = True, inplace = True) # set index to JD
    
    return data

 # elif type_of_visit == 'imagette':
        
    #     data = pd.DataFrame(data =    {
    #                                 'visit_ID': np.full(nb_images, id),
    #                                 'img_counter': np.arange(nb_images),
    #                                 #'raw_images': flattened_images,
    #                                 'derotated_raw_images': flattened_derotated_images,
    #                                 'masked_images': flattened_masked_images, 
    #                                 'binary_images': flattened_binary_images, 
    #                                 #'mask': flattened_mask,
    #                                 'JD': time_images_utc_jd,
    #                                 'time': time_images_utc,
    #                                 'nb_cosmics' : nb_cosmics.astype(int),
    #                                 'density_cosmics' : density_cosmics,
    #                                 'pix_cosmics': images_contours,
    #                                 'im_height': np.full(nb_images, height_images),
    #                                 'im_width': np.full(nb_images, width_images),
    #                                 'threshold_cosmics': np.full(nb_images, threshold_cosmics),
    #                                 'n_exp': np.full(nb_images, n_exp),
    #                                 'exp_time': np.full(nb_images, exp_time),
    #                                 # 'los_to_sun': np.full(nb_images, los_to_sun),
    #                                 # 'los_to_moon': np.full(nb_images, los_to_moon),
    #                                 # 'los_to_earth': np.full(nb_images, los_to_earth),
    #                                 'target_name': np.full(nb_images, target_name),
    #                                 'mag_G': np.full(nb_images, mag_G),
    #                                 'ra': np.full(nb_images, ra),
    #                                 'dec': np.full(nb_images, dec)                                    
    #                                 }
    #     )
    
        
        
        
    #     data.set_index('JD',drop = True, inplace = True) # set index to JD

       
    #     # ###############
    #     # # get restituted orbit master file
    #     data_res_orb = pd.read_csv('/Users/alexisheitzmann/Documents/CHEOPS/Code/SAA_monitoring_MC/SAA_ORs_reduction_pipeline/master_orb_res.csv')
    #     data_res_orb.set_index('JD',drop = True, inplace = True) # set index to JD
    #     data_res_orb.sort_index()
    #     data_RES = data_res_orb.copy()
    #     # Use only the part of the restitued around the visit, +/- 1 day to have enough baseline for the interpolation
    #     time_min = data.index.min() - 1
    #     time_max = data.index.max() + 1
    #     data_RES = data_RES[(data_RES.index > time_min) & (data_RES.index < time_max)]
    #     # Interpolate LAT/LON at the time of the images
    #     data = unfold_interp_fold(data_RES, data, order = 3)
        
    #     return data, data_res_orb

    # else:
    #     raise ("Type of visit must be either 'subarray' or 'imagette'")

if __name__ == "__main__":
    
    if "chpscn" in socket.gethostname(): # on compute node
        file_list = MAIN_PATH / "ref_files/filelist.txt"
        image_files_list = []
        roll_angle_files_list = []
        with open(file_list, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "SCI_RAW_SubArray" in line:
                    image_files_list.append(Path(line[:-1]))
                else:
                    roll_angle_files_list.append(Path(line[:-1]))

    else: # on local machine
        visits_folder = MAIN_PATH / "test_visits"
        image_files_list, roll_angle_files_list = get_files_with_substring(visits_folder)

    # Thresholds
    threshold_noise_SAA = 6.5
    threshold_noise_science = 10 # This is now scaled by sqrt(nexp) in the main loop. TBD: replace with a gaussian fit of the noise
    threshold_cosmics = 250

    generate_plots = False

    # For a single visit
    '''
    visit_idx = 5
    all_data, restituted_orbit = main_loop(image_files_list[visit_idx], roll_angle_files_list[visit_idx],  threshold_noise, threshold_cosmics, visit_type, generate_plots)
    '''

    # For all visits
    all_data = pd.DataFrame()
    restituted_orbit = pd.DataFrame()
    for i in range(len(image_files_list)):
        visit = image_files_list[i].stem[3:20]
        if visit.split('_')[0] == "PR340102":
            visit_type = "SAA"
            threshold_noise = threshold_noise_SAA
        else:
            visit_type = "science"
            threshold_noise = threshold_noise_science
        # Check the consistency between files: 
        if not (image_files_list[i].stem.split('_')[:3] == roll_angle_files_list[i].stem.split('_')[:3]):
            print("SubArray don't match the SCI_COR_lightcurve/Attitude file !!!")
            pass
        print(f"########################")
        print(f"Processing visit {i+1} of {len(image_files_list)} ({visit})...")
        df = main_loop(image_files_list[i], roll_angle_files_list[i], threshold_noise, threshold_cosmics, visit_type, generate_plots)
        
        all_data = pd.concat([all_data, df], ignore_index=False)

    all_data = all_data.sort_index()
    start_data = all_data.iloc[0].time.strftime(format = "%Y_%m_%d")
    end_data = all_data.iloc[-1].time.strftime(format = "%Y_%m_%d")
    file_name = "data_" + start_data + "_to_" + end_data + ".pkl"
    save_path = MAIN_PATH / file_name
    all_data.to_pickle(save_path, compression='infer', protocol=5, storage_options=None)
