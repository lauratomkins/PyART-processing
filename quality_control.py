# -*- coding: utf-8 -*-
"""
Created on Thu Aug 06 12:27:04 2015

Contains various functions for quality control of radar data. Many rely on functions included in
PyART, while others are entirely custom.

@author: thecakeisalie

Version date: 5/23/2019
Daniel Hueholt
North Carolina State University
Undergraduate Research Assistant at Environment Analytics
"""
import pyart
import string
import sys
import time
import numpy as np

def dealias(radar, filename, outpath, name2dealias, new_name, nyquist_vel, 
            skip_along_ray, skip_between_rays, savefile=True):
    """
    DESCRIPTION: Dealiases a specified field using the PyART
        dealiased_region_based function and can save off a separate cfradial 
        file with the new dealiased field.
        
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    filename = A string containing the name of the file.
    outpath = A string that specifies the full path to where the .png images
        will be saved. 
    name2dealias = A string that specifies which field in the radar object to
        dealias.
    new_name = A string specifying the name of the new dealiased field.
    nyquist_value = Numeric interger.
    skip_along_ray = Integer value. Maximum number of filtered gates to skip
        over when joining regions, gaps between region larger than this will 
        not be connected.  Parameters specify the maximum number of filtered 
        gates along a ray. Set to 0 to disable 
        unfolding across filtered gates.
    skip_between_rays = Integer value. Maximum number of filtered gates to skip
        over when joining regions, gaps between region larger than this will 
        not be connected.  Parameters specify the maximum number of filtered 
        gates between rays. Set to 0 to disable 
        unfolding across filtered gates.
    
    OPTIONAL INPUTS:
    savefile = Default set to True. A boolean value. If True, will save a new 
        cfradial file containing the dealiased field. 
    
    OUTPUTS:
    radar = A python object structure that contains radar information with the 
        addition of the new dealiased field.
    """
    
    # Determine which sweeps have no data and extract only the ones that have data 
    if radar.metadata['original_container'] != 'NEXRAD Level II':
        if nyquist_vel != None:
            good = []
            print("Dealiasing in progress!")
            for i in range(radar.nsweeps):
                sweep = radar.get_slice(i)
                if radar.fields[name2dealias]['data'][sweep][0,:].flatten().count() != 0:
                    good.append(i)
            radar = radar.extract_sweeps(good) # For NEXRAD data there are some sweeps which have velocity but not rhohv and vice versa. Need to plot all sweeps
    
    # Dealias and add new dealiased field to radar object 
    if radar.metadata['original_container'] == 'NEXRAD Level II': # NEXRAD has different nyquist velocity for all sweeps. Don't specify velocity and function will automatically detect
        corr_vel = pyart.correct.dealias_region_based(radar,vel_field=name2dealias,skip_along_ray=skip_along_ray,skip_between_rays=skip_between_rays,gatefilter=False,keep_original=False)
        radar.add_field(new_name, corr_vel, True)
    else:
        corr_vel = pyart.correct.dealias_region_based(radar,vel_field=name2dealias,nyquist_vel=nyquist_vel,skip_along_ray=skip_along_ray,skip_between_rays=skip_between_rays,gatefilter=False,keep_original=False)
        radar.add_field(new_name, corr_vel, True)
    print("Dealiasing complete in current file!")
       
    # Save a new file containing the dealiased field
    if savefile == True:
        split_file = filename.split('.')
        filesavename = "%s%s_dealiased.cfradial" % (outpath, split_file[0])
        pyart.io.write_cfradial(filesavename, radar)
            #print("Unable to save dealiased data as CF/Radial! Continuing to save images.")
            
        
    return radar
    
def set2range(radar, field, val_max, val_min):
    """
    DESCRIPTION: Finds instances where a value is out side of the specified 
        bounds and sets the value to the closest bound. Ex. The bounds are
        [-8,16]. A value of -10 will be corrected to -8 and a value of 30 will 
        be corrected to 16.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    field = A string of the field name. Ex. "DBZ"
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
    """
    # Check to see if values are in range
    
    over = (radar.fields[field]['data'].data > val_max)
    under = (radar.fields[field]['data'].data < val_min)
    
    radar.fields[field]['data'].data[over] = val_max
    radar.fields[field]['data'].data[under] = val_min
    
    return radar 

def removeNoiseZ(radar, radar_fieldnames, Z_min, Z_max):
    """
    DESCRIPTION: Removes data across all variables corresponding to noisy reflectivity
        values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = Names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    try:
        over = (radar.fields['reflectivity']['data'].data > Z_max)
        under = (radar.fields['reflectivity']['data'].data < Z_min)
    except KeyError:
        over = (radar.fields['DBZH']['data'].data > Z_max)
        under = (radar.fields['DBZV']['data'].data < Z_min)
    
    # Troubleshooting
    #print(radar_fieldnames)
    #print(radar.fields) #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[over] = None
            radar.fields[field]['data'].data[under] = None
        except NotImplementedError:
            radar.fields[field]['data'][over] = None
            radar.fields[field]['data'][under] = None
    
    return radar

def removeMountainClutter(radar, radar_fieldnames):
    """
    DESCRIPTION: Attempts to kill the friggin mountains!! Removes return that has high reflectivity
    but near-zero velocity. Works surprisingly well in winter storms.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = Names of fields in the radar object
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    Z_max = 25
    
    slow = (radar.fields['dealiased_velocity']['data'].data < 0.2)
    slow2 = (radar.fields['dealiased_velocity']['data'].data > -0.2)
    near_stationary = [slow & slow2]
    highz = (radar.fields['reflectivity']['data'].data > Z_max)
    mountainIndices = [near_stationary & highz]
    shapeOfData = np.array(radar.fields['reflectivity']['data'].data.shape)
    mountainIndices = np.reshape(mountainIndices,[shapeOfData[0],shapeOfData[1]])
    
    # Troubleshooting
    #print radar_fieldnames
    #print radar.fields #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[mountainIndices] = None
        except NotImplementedError:
            radar.fields[field]['data'][mountainIndices] = None #For calculated fields like Rasmussen snow rate
    
    return radar

def removeNoiseZdr(radar, radar_fieldnames, Zdr_min, Zdr_max):
    """
    DESCRIPTION: Removes data across all variables corresponding to noisy Zdr values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    over = (radar.fields['differential_reflectivity']['data'].data > Zdr_max)
    under = (radar.fields['differential_reflectivity']['data'].data < Zdr_min)
    
    # Troubleshooting
    #print radar_fieldnames
    #print radar.fields #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        radar.fields[field]['data'].data[over] = None
        radar.fields[field]['data'].data[under] = None
    
    return radar

def removeNoiseRhoHV(radar, radar_fieldnames, rhohv_min, rhohv_max):
    """
    DESCRIPTION: Removes data across all variables corresponding to low rhoHV
        values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = Names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    try:
        over = (radar.fields['cross_correlation_ratio']['data'].data > rhohv_max)
        under = (radar.fields['cross_correlation_ratio']['data'].data < rhohv_min)
    except KeyError:
        try:
            over = (radar.fields['RHOHV']['data'].data > rhohv_max)
            under = (radar.fields['RHOHV']['data'].data < rhohv_min)
        except KeyError:
            over = (radar.fields['correlation_coefficient']['data'].data > rhohv_max)
            under = (radar.fields['correlation_coefficient']['data'].data < rhohv_min)
    
    # Troubleshooting
    #print radar_fieldnames
    #print radar.fields #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[over] = None
            radar.fields[field]['data'].data[under] = None
        except NotImplementedError:
            radar.fields[field]['data'][over] = None
            radar.fields[field]['data'][under] = None
    
    return radar

def removeNoisePhiDP(radar, radar_fieldnames, PhiDP_min, PhiDP_max):
    """
    DESCRIPTION: Removes data across all variables based on PhiDP values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = Names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    try:
        over = (radar.fields['PHIDP']['data'].data > PhiDP_max)
        under = (radar.fields['PHIDP']['data'].data < PhiDP_min)
    except KeyError:
        over = (radar.fields['specific_differential_phase']['data'].data > PhiDP_max)
        under = (radar.fields['specific_differential_phase']['data'].data < PhiDP_min)
    
    # Troubleshooting
    #print radar_fieldnames
    #print radar.fields #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[over] = None
            radar.fields[field]['data'].data[under] = None
        except TypeError:
            radar.fields[field]['data'][over] = None
            radar.fields[field]['data'][under] = None
    
    return radar

def removeNoiseNCP(radar, radar_fieldnames, ncp_min, ncp_max):
    """
    DESCRIPTION: Removes data across all variables corresponding to low NCP
        values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_fieldnames = names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    over = (radar.fields['normalized_coherent_power']['data'].data > ncp_max)
    under = (radar.fields['normalized_coherent_power']['data'].data < ncp_min)
    
    # Troubleshooting
    #print radar_fieldnames
    #print radar.fields #If you suspect the data is blank, uncomment this to print a sample to the console
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[over] = None
            radar.fields[field]['data'].data[under] = None
        except NotImplementedError:
            radar.fields[field]['data'][over] = None
            radar.fields[field]['data'][under] = None
    
    return radar

def removeNoiseSNR(radar, radar_fieldnames, snr_min, snr_max):
    """
    DESCRIPTION: Removes data across all variables corresponding to low or high SNR
        values.
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radarFieldnames = names of fields in the radar object
    val_max = Numeric value. Sets the maximum bound.
    val_min = Numeric value. Sets the minimum bound.
    
    OUTPUTS:
    radar = The original radar object but with the edited values for the 
        specified field.
        
    """
    # Check to see if values are in range
    within = np.logical_and(radar.fields['snr']['data'].data < snr_max, radar.fields['snr']['data'].data > snr_min)
    
    # Troubleshooting
    # print(radar_fieldnames)
    
    for field in radar_fieldnames:
        try:
            radar.fields[field]['data'].data[~within] = None
        except NotImplementedError:
            radar.fields[field]['data'][~within] = None
        
    
    return radar
    
def PPI_fixfilename(filename): 
    """
    DESCRIPTION: Fixes the inconsistencies in the ROSE file name schemes.
    
    INPUTS:
    filename = The orginal name of the file.
    
    OUTPUTS:
    filename = The corrected file name.
    """                 
    end = len(filename)
    if filename[end-17:end-15] == 's0':
        chopstr = filename[end-17:end-6]
        if chopstr[-1:] == '0':
            longchopstr = filename[end-17:end-5]
            filename = filename.replace(longchopstr, 'SUR_')
        filename = filename.replace(chopstr, 'SUR_')
    return filename
    
def fix_CHILL_PPI_sweep_start_end(radar):
    """
    DESCRIPTION: Fixes the CSU-CHILL PPI sweep start and end indices to omit 
    transitions between sweeps.
    
    INPUTS:
    radar = Python radar object containing CSU-CHILL data
    
    OUTPUTS:
    radar = Radar object with corrected start and end sweep indices.
    """
    radar.sweep_start_ray_index['data'] = [0,382,789,1208,1604,2027]
    radar.sweep_end_ray_index['data'] = [381,788,1176,1567,1993,2428]
    return radar