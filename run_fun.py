# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:54:10 2015

Processes radar data using PyART and custom tools.

@author: thecakeisalie

Version date: 5/23/2019
Daniel Hueholt
North Carolina State University
Undergraduate Research Assistant at Environment Analytics

"""
import numpy as np
import quality_control
import Master_plotter
import pyart
import gc
import sys
import calculated_fields
#import colormap
import time

def parse_filelist(filelist, inpath, outpath, radar_type, fields, ranges, plot_bool, cmaps,
                   colorbar_labels, x_lim, y_lim, scan_strat, dealias_bool, save_cfradial_bool,
                   name2dealias, new_name, nyquist_vel, Z_mask, Zdr_mask, PhiDP_mask, rhoHV_mask,
                   NCP_mask, SNR_mask, Zdr_offset, snow_rate_bool, vdiv_bool, mountain_clutter_bool, 
                   contour_bool, base_field, contour_field, contour_levels, azi_overlay):
    
    # Loop through each file in the list
    length_filelist = np.size(filelist)
        
    for item in range(0,length_filelist):
        # Define the filename
        filename = filelist[item]
        if length_filelist==1:
            filename = filelist
        
        # Make full path to file
        fqfn = inpath + filename
        # Print the full path
        print(fqfn)
        
        # Construct radar object
        if radar_type=='CHILL':
            #CHILL uses a specialized UF format that requires the keys to be designated manually
            radar = pyart.io.read_uf(filename,field_names={
                    'DZ': 'reflectivity',
                    'VE': 'corrected_velocity',
                    'W2': 'spectrum_width',
                    'DR': 'corrected_differential_reflectivity',
                    'RH': 'cross_correlation_ratio',
                    'NC': 'normalized_coherent_power',
                    'DP': 'one_way_differential_phase',
                    'KD': 'two_way_differential_phase',
                        })
        else:
            #Other radar file formats can be determined automatically
            radar = pyart.io.read(fqfn)            
        
        if snow_rate_bool:
            radar = calculated_fields.rasmussen_snow_rate(radar,fields)
            ranges.append((0,1.25))
            cmaps.append('viridis') #or YlGnBu
            colorbar_labels.append('Snow rate (mm/hr)')
      
        # Data quality
        if dealias_bool == False:
            if Z_mask['bool'] == True:
                radar = quality_control.removeNoiseZ(radar,fields,Z_mask['range'][0],Z_mask['range'][1])
            if PhiDP_mask['bool'] == True:
                radar = quality_control.removeNoisePhiDP(radar,fields,PhiDP_mask['range'][0],PhiDP_mask['range'][1])
            if rhoHV_mask['bool'] == True:
                radar = quality_control.removeNoiseRhoHV(radar,fields,rhoHV_mask['range'][0],rhoHV_mask['range'][1])
            if NCP_mask['bool'] == True:
                radar = quality_control.removeNoiseNCP(radar,fields,NCP_mask['range'][0],NCP_mask['range'][1])
            if SNR_mask['bool'] == True:
                radar = quality_control.removeNoiseSNR(radar,fields,SNR_mask['range'][0],SNR_mask['range'][1])
        else:
            # Need to apply masks before dealiasing, but a KeyError occurs due to the mismatch between new_name and name2dealias
            # Replace new_name with name2dealias
            v_ind = fields.index(new_name)
            fields.remove(new_name)
            fields.insert(v_ind,name2dealias)
            
            #despeckler = pyart.correct.despeckle_field(radar,'corrected_velocity',threshold=(-40,40)) # someday
            
            if Z_mask['bool'] == True:
                radar = quality_control.removeNoiseZ(radar,fields,Z_mask['range'][0],Z_mask['range'][1])
            if Zdr_mask['bool'] == True:
                radar = quality_control.removeNoiseZdr(radar,fields,Zdr_mask['range'][0],Zdr_mask['range'][1])
            if PhiDP_mask['bool'] == True:
                radar = quality_control.removeNoisePhiDP(radar,fields,PhiDP_mask['range'][0],PhiDP_mask['range'][1])
            if rhoHV_mask['bool'] == True:
                radar = quality_control.removeNoiseRhoHV(radar,fields,rhoHV_mask['range'][0],rhoHV_mask['range'][1])
            if NCP_mask['bool'] == True:
                radar = quality_control.removeNoiseNCP(radar,fields,NCP_mask['range'][0],NCP_mask['range'][1])
            if SNR_mask['bool'] == True:
                radar = quality_control.removeNoiseSNR(radar,fields,SNR_mask['range'][0],SNR_mask['range'][1])
            
            # Reverse the process
            fields.remove(name2dealias)
            fields.insert(v_ind,new_name)
      
        # Account for Zdr offset
        if Zdr_offset['bool']:
            radar.fields['differential_reflectivity']['data'].data[0:len(radar.fields['differential_reflectivity']['data'].data)] = np.subtract(radar.fields['differential_reflectivity']['data'].data,Zdr_offset['offset'])
        
        # Dealias velocity data
        if dealias_bool == True:
            radar = quality_control.dealias(radar, filename, outpath, name2dealias, new_name, nyquist_vel, 100, 100, save_cfradial_bool)
            gc.collect()
        
        print("Dealiasing complete!") #Dealiasing can take a while, this helps keep the user aware of PyART's progress.
        
        # Add the vertical derivative of dealiased velocity. FOR RHI SCANS ONLY! vdiv cannot be calculated in this way for PPI/sector scans.
        if scan_strat == 'PPI':
            vdiv_bool = False
        if vdiv_bool:
            radar = calculated_fields.velocity_vertical_divergence(radar,fields)
            ranges.append((0,2))
            cmaps.append('inferno')
            colorbar_labels.append('|vDiv| ms^(-1)/gate')
        
        # Remove mountain clutter. Comes late in the process because it relies partly on dealiased velocity.
        if mountain_clutter_bool:
            radar = quality_control.removeMountainClutter(radar,fields)
        
        #pytda.calc_turb_sweep(radar) #someday!!
        
        # Set figure sizes
        if scan_strat == 'RHI':
            #for 0 to 6 km use [40,6]
            #for 0 to 9 km use [42,8]
            figsize = [38,8]#[40,6] #some others that have been useful [40,12]#[14.66, 3.652]#[40, 12]#[49.82, 4]#[43.6, 3.5]#30,4 #25,4
        else:
            figsize = [16,16] #Same settings used for PPI and sector scans # [16.346, 12]
        
        # Create and save plots
        if plot_bool == True:
                Master_plotter.plot(radar, radar_type, filename, outpath, scan_strat, fields, ranges, cmaps, colorbar_labels, figsize, dealias_bool, x_lim, y_lim, contour_bool, base_field, contour_field, contour_levels, azi_overlay)
        else: 
            #Do nothing, other than collect the garbage
            gc.collect()
        
        del radar
        del filename
        del fqfn
        gc.collect()