import numpy as np
import pandas as pd
import socket
import argparse
from functions import *
from datetime import datetime
from pathlib import Path
   
MAIN_PATH = Path.cwd()
TIME_FORMAT_DB = "%Y-%m-%d %H:%M:%S"
INPUT_TIME_FORMATS = [TIME_FORMAT_DB, "%Y-%m-%dT%H:%M:%S"]
INPUT_TIME_FORMATS_STR = "'yyyy-mm-dd hh:mm:ss', 'yyyy-mm-ddThh:mm:ss'"

def validTime(timeStr):
    t = None

    for format in INPUT_TIME_FORMATS:
        try:
            t = datetime.strptime(timeStr, format)
        except ValueError:
            pass
    
    if t is None:
        msg = f"Invalid time '{timeStr}'. Valid formats: {INPUT_TIME_FORMATS_STR}"
        raise argparse.ArgumentTypeError(msg)
    return t

def getArgs():
    """
    Returns the command line arguments.
    """

    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s [-s START] [-e END]"
            ),
        description=(
            "Get the list of visits betweenPrints a section of the monitoring_dashboard for a specific "
            "start and end time."))
    
    parser.add_argument(
        "-s", "--start",
        type=validTime,
        default = '2024-07-15T00:00:00',
        help=(
            f"The start time of the period to export. Valid time formats: "
            f"{INPUT_TIME_FORMATS_STR}. Defaults to 2024-06-15."))

    parser.add_argument(
        "-e", "--end",
        type=validTime,
        default = datetime.now().strftime(TIME_FORMAT_DB), # Defaults to now
        help=(
            f"The end time of the period to export. Valid time formats: "
            f"{INPUT_TIME_FORMATS_STR}. Defaults to the current time."))
    
    # Returns two elements, one containing the argument specified above, and
    # another containing any additional undefined arguments.
    return parser.parse_args()

def database_query(start_time, end_time, copy = True):
    
    import sys
    sys.path.append('/home/astro/heitzman/')
    from pymd import PyMD
    from pymd.model import Visit
    
    # if copy:
    import subprocess 
    # copy the database here
    print('\n Copying db ...')
    command = 'scp chps_ops@chpscn02:/opt/monitoring_dashboard_database/monitoring_dashboard.db .'
    subprocess.run(command, shell=True)
    database= Path.cwd() / "monitoring_dashboard.db"
        
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
            #     #(Visit.exptime < 10) 
            #     Visit.start_time > '{start_time_str}'
            #     & Visit.start_time < '{end_time_str}'
            #     & Visit.visit_counter.isnot(None)
            #     & (
            #          ((Visit.programme_type >= 10) & (Visit.programme_type < 40))
            #        | ((Visit.programme_type >= 40) & (Visit.programme_type < 50)))
            #     )
            # for row in query.all():
            #     print(row.formatted_visit_id)
            
            
    return visit_id_list

if __name__ == "__main__":
    
    # Command line arguments
    args = getArgs()
    
    start = args.start
    end   = args.end
        
    processing_path = Path("/srv/astro/projects/cheops/processing/")

    m4c_prod = processing_path / "chpscn02/opt/monitor4cheops/Operations/"
    m4c_reproc = processing_path / "chpscn03/opt/monitor4cheops/reprocessing02/"
    
    visits_dir        = m4c_prod / "repository/visit/"
    visits_dir_reproc = m4c_reproc / "repository/visit/"

    visits_list = database_query(start,end, copy = True)
    visits_list.sort(reverse = False)

    image_files_list = []
    roll_angle_files_list = []

    # Get paths for all visits of interest
    nb_visits = 0
    start_proc = False
    for visit in visits_list:
        # if visit == "PR100036_TG001501":
        #     start_proc = True
        # if start_proc == False:
        #     continue
        if ("PR100036_TG0015" in visit) or ("PR100036_TG0014" in visit): # Exclude specific visits
            continue 
        visit_dir = visits_dir_reproc / visit[:4] / visit
        if not visit_dir.is_dir(): # Data more recent than 07 June 2023
            visit_dir = visits_dir / visit[:4] / visit
        subarray_found = False
        roll_angle_found = False
        for filename in visit_dir.iterdir():
            if subarray_found and roll_angle_found: # to speed up the process slightly
                break
            elif "SCI_RAW_SubArray" in filename.stem:
                subarray_path = visit_dir / filename
                subarray_found = True
            elif ("COR_Lightcurve-DEFAULT" in filename.stem) and (visit.split('_')[0] != "PR340102"):
                roll_angle_path = visit_dir / filename
                roll_angle_found = True
            elif ("Attitude" in filename.stem) and (visit.split('_')[0] == "PR340102"):
                roll_angle_path = visit_dir / filename
                roll_angle_found = True
            else:
                continue
            
        if subarray_found and roll_angle_found: # Only use visit that have both these files
            print(f"Visit: {visit_dir.name.split('/')[-1]}")
            nb_visits += 1
            image_files_list.append(subarray_path)
            roll_angle_files_list.append(roll_angle_path)
        else:
            continue

    print(f" Recovered {nb_visits} visits.")
    
    image_files_list.sort(reverse = False)
    roll_angle_files_list.sort(reverse = False)
        
    file_list = MAIN_PATH / "ref_files/filelist.txt"

    with open(file_list, "w") as file:
        for im,roll in zip(image_files_list,roll_angle_files_list):
            file.write(f"{im}\n")
            file.write(f"{roll}\n")