# -*- coding: utf-8 -*-
"""
DESCRIPTION: Contains functions used to plot radar data that has been processed by the 
rest of the PyART toolkit.

Created on Fri May 30 15:06:17 2014

@author: thecakeisalie

Version date: 5/23/2019
Daniel Hueholt
North Carolina State University
Undergraduate Research Assistant at Environment Analytics
"""

# Import required modules
import numpy as np
import gc
import pyart
from matplotlib import pyplot as plt
from matplotlib import font_manager as ff
import gen_fun
import quality_control
import math
import scipy.ndimage as spyi
import time
import colormap



def contour_overlay(radar, sweepnum, contourField, baseField, ax, total_text, contourValues, scan_strat):
    """
    DESCRIPTION: Overlays contours on a plot of a different data type.
    
    INPUTS:
    radar = A python object structure that contatins radar information. Created by PyART
        in one of the pyart.io.read functions.
    sweepnum = The sweep number of the plot in question.
    contourField = Data key to be contoured.
    baseField = Data key referring to the base plot.
    ax = The plot in question.
    total_text = Text for the plot title.
    contourValues = Numpy array of desired values at which to draw contours
    scan_strat = RHI and PPIs require different approaches for contouring
    
    Written by Daniel Hueholt
    Undergraduate Research Assistant at Environment Analytics
    Last update: 5/22/2019
    """
    # Setup
    dataToContourRaw = radar.get_field(sweepnum,contourField)
    x,y,height = radar.get_gate_x_y_z(sweepnum,edges=False)
    x/=1000.0
    y/=1000.0
    height/=1000.0
    dataCoord = np.sqrt(x**2+y**2)*np.sign(y)
    if np.nanmax(dataCoord) < 0:
        dataCoord = -dataCoord

    dataToContourSmooth = spyi.gaussian_filter(dataToContourRaw,sigma=1.2)
    # Draw contours
    contourColormap = colormap.contourColors()
    if scan_strat=='RHI':
        contours=ax.contour(dataCoord,height,dataToContourSmooth,contourValues,linewidths=1.5,cmap=contourColormap,linestyles='solid',antialiased=True)
    else:
        contours=ax.contour(x,y,dataToContourSmooth,contourValues,linewidths=1.5,cmap=contourColormap,linestyles='solid',antialiased=True)
    contoured_field_is = 'Contoured field is ';
    long_spacer = '     '
    total_text = total_text+long_spacer+contoured_field_is+contourField
    #Make contour labels
    plt.clabel(contours,contourValues,fmt='%r',inline=True,fontsize=20) #%r
    #print("Contour section completed!")
    
    return total_text

   
def plot(radar, radar_type, filename, outpath, scan_strat, fields, ranges, cmaps, 
         colorbar_labels, figsize, dealias_bool, x_lim, y_lim, contour_bool, base_field, contour_field, contour_levels, azi_overlay, axis=None,
         title_flag=False,colorbar_flag = True,):
    """
    DESCRIPTION: Plots polar RHI and PPI data from netCDF on a cartesian grid. 
    
    INPUTS:
    radar = A python object structure that contains radar information. Created
        by PyART in one of the pyart.io.read functions.
    radar_type = A string describing the radar type, e.g. 'CHILL','KASPR', etc.
    filename = A string containing the name of the file.
    outpath = A string that specifies the full path to where the .png images
        will be saved.
    scan_strat = String of either "RHI", "PPI", or "Sector".
    fields = A list of strings specifying which fields to plot. 
        Ex. ["DBZ", "VEL"]
    ranges = A list of tuples specifying the minimum and maximum values for 
        each field in order. The minimum value is the first number and the 
        maximum value is the second number. Ex. [(-10,65), (-30,30)]
    cmaps = A list containing strings or calls of the names of the colorbars used to 
        plot each field respectively. Ex. [LCH, "RdBu_r"]
    colorbar_labels = A list of strings that specify the colorbar labels on 
        each field respectively. Ex. ["Reflectivity (dBZ)", "Velocity (m/s)] 
    figsize = A list of numbers to specify the size of all the figures. The 
        same value is used for all figures. Ex. [14,12]
    dealias_bool = A boolean value where "True" will run the PyART dealiasing 
        algorithm over the velocity data and output a new netCDF file with a 
        dealiased velocity field called "VEL_D", and "False" will not run the 
        PyART dealiasing nor output a new netCDF file.
    x_lim = Array giving x-limits in kilometers
    y_lim = Array giving y-limits in kilometers.
    
    OPTIONAL INPUTS (set to None/False by default):
    axis = Default set to None. Sets whether to plot axis labels or not.
    title_flag = Default set to False. Set to True to plot title.
    colorbar_flag = Default set to True. Set to False to disable the colorbar.
    
    OUTPUTS:
    Plot(s) of RHI/PPI data.
    
    WEIRD QUIRKS:
    metadisp: Logical variable to control whether metatext (title, azimuth, etc.) is included in plots.
    
    """    
    for sweepnum in range(0, np.size(radar.sweep_number['data'])):
        
        if scan_strat == 'RHI':
            azi = gen_fun.get_azimuth(radar, sweepnum)
        else:
            azi = [] #Azimuth only matters for RHI scans
        
        for i in range(len(fields)):
            
            field = fields[i]
            try:
                cmap = cmaps[i]
            except IndexError:
                raise IndexError("Check to make sure number of colormaps is correct!")
                
            try:
                colorbar_label = colorbar_labels[i]
            except IndexError:
                raise IndexError("Check to make sure number of colorbar labels is correct!")
            
            vmin, vmax = ranges[i]                       
            
            # Instantiate PyART radar display object
            display = pyart.graph.RadarDisplay(radar)            
            
            # Clear and close any figures that might be open
            plt.clf()
            plt.close()
            
            # Initiate plot and specify size
            fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111)
            ax.set_facecolor('#CCCCCC') #Controls background color within the radar data display. Can be any Hex color code. Normal value: #CCCCCC (light gray)
                
            try:
#               Set up metadata text to be placed in figure (Currently implemented only for CHILL and KASPR)
                if radar_type=='CHILL':
                    y_text = filename[3:7]
                    m_text = filename[7:9]
                    d_text = filename[9:11]
                    h_text = filename[12:14]
                    min_text = filename[14:16]
                    s_text = filename[16:18]
                elif radar_type=='KASPR':
                    y_text = filename[17:21]
                    m_text = filename[21:23]
                    d_text = filename[23:25]
                    h_text = filename[26:28]
                    min_text = filename[28:30]
                    s_text = filename[30:32]
                elif radar_type=='NEXRAD':
                    y_text = filename[4:8]
                    m_text = filename[8:10]
                    d_text = filename[10:12]
                    h_text = filename[13:15]
                    min_text = filename[15:17]
                    s_text = filename[17:19]
                utc_text = ' UTC'
                spacer = ' '
                colon = ':'
                long_spacer = '     '
                time_text = y_text+spacer+m_text+spacer+d_text+long_spacer+h_text+colon+min_text+colon+s_text+utc_text #YYYY MM DD      HH:MM:SS UTC
                if scan_strat == 'RHI':
                    ang_text = 'azimuth = '
                    a_text = str(round(azi,1)) #Round azimuth to 1 decimal place for title text
                else:
                    ang_text = 'elevation = '
                    a_text = str(round(radar.fixed_angle['data'][sweepnum],2)) #Round tilt angle to 2 decimal places for title text

                degree_sym = u'\N{DEGREE SIGN}'
                angle_text = ang_text+a_text+degree_sym
                if (scan_strat=='Sector') or (scan_strat=='sector'):
                    sect_text = 'Sector PPI'
                    total_text = time_text+long_spacer+angle_text+long_spacer+sect_text
                else:
                    total_text = time_text+long_spacer+angle_text+long_spacer+scan_strat
                
                #caption_dict controls characteristics for all meta text in the title
                # Currently, only size is set here, but this dictionary can be used 
                # for font name, color, weight, etc.
                caption_dict = {'size': 26}
                
                metadisp = True #Logical to display metatext in figure
                    
                if scan_strat == 'RHI':
                    display.plot_rhi(field,sweepnum,vmin=vmin,vmax=vmax,title_flag=title_flag,cmap=cmap,axislabels=(axis, 'AGL (km)'),colorbar_flag=True,colorbar_label=colorbar_label)
                    
                    # Contour overlay code
                    if contour_bool:
                        if field==base_field:
                            total_text = contour_overlay(radar,sweepnum,contour_field,base_field,ax,total_text,contour_levels,scan_strat)
                    
                    display.set_limits(ylim=y_lim)
                    display.set_limits(xlim=x_lim)
                    display.set_aspect_ratio(aspect_ratio=1) #important!!
                    plt.yticks(np.arange(0,y_lim[1]+1,step=1))
                    if metadisp:
                        labeled = 'labeled_' 
                        plt.tight_layout(rect=[0,0.09,1,0.95])
                        plt.figtext(0.35,0.95,total_text,caption_dict) #Title the figure with the metatext
                    else:
                        plt.tight_layout(rect=[0,0.09,1,0.95])                   
                        
                else:
                        display.plot_ppi(field, sweepnum, vmin = vmin, vmax = vmax, title_flag = title_flag, cmap = cmap, axislabels = (axis, "N-S distance (km)"),colorbar_flag=True, colorbar_label = colorbar_label)
                        
                        # Contour overlay code
                        if contour_bool:
                            if field==base_field:
                                total_text = contour_overlay(radar,sweepnum,contour_field,base_field,ax,total_text,contour_levels,scan_strat)
                        
                        ## Sector scan PARTIALLY IMPLEMENTED
                        if scan_strat == 'Sector':
                            #The edges of the sector are calculated using trigonometry
                            azi_lim = np.asarray([min(radar.azimuth['data']),max(radar.azimuth['data'])])
                            sec_angle = np.subtract([90,90],azi_lim)
                            range_lims = np.asarray([max(x_lim),max(y_lim)])
                            sec_rad = np.radians(sec_angle)
                            trig_sec = np.array([math.cos(sec_rad[0]),math.tan(sec_rad[1])])
                            offset = np.multiply(range_lims,trig_sec)
                            
                            #The sector edges are plotted as lines
                            #By default, the lines are solid and use the #105456 UNCW teal color
                            plt.plot([0,offset[0]],[0,range_lims[1]],color='#105456',linewidth=7)
                            plt.plot([0,range_lims[0]],[0,offset[1]],color='#105456',linewidth=7)
                            
                            #The sector edges are then used to create a domain of consistent size that moves with the sector.
                            #This is currently NOT fully functional--it assumes the range is 60km, and only works for certain sectors.
                            if (offset[0]<0) and (60+offset[1]-3>-60):
                                x_lim = [offset[1]-3,60+offset[1]-3]
                            else:
                                x_lim = [0,60]
                            
                            if (offset[1]<0) and (60+offset[1]-3>-60):
                                y_lim = [offset[1]-3,60+offset[1]-3]#[offset[1]-3,75-offset[1]-3]
                            else:
                                y_lim = [0,60]
                            
                            display.set_limits(ylim=y_lim)
                            display.set_limits(xlim=x_lim)
                            display.set_aspect_ratio(aspect_ratio=1)
                        ## ---SECTOR SCAN SETTINGS END HERE---

                        #Overlay RHI azimuth on the PPI scans
                        if scan_strat == 'PPI':
                            if azi_overlay['bool']:
                                max_length = x_lim[1]
                                x_c,y_c = gen_fun.azi_calculator(azi_overlay['azi_lines'],max_length)
                                for iter in range(0,np.size(azi_overlay['azi_lines'])):
                                    plt.plot([0,x_c[iter]],[0,y_c[iter]],color=azi_overlay['color'],linewidth=azi_overlay['linewidth'])
                            
                        display.set_limits(ylim=y_lim)
                        display.set_limits(xlim=x_lim)
                        display.set_aspect_ratio(aspect_ratio=1)
                        
                        if metadisp:
                            labeled = 'labeled_'
                            if radar_type=='NEXRAD' and contour_bool==True:
                                plt.figtext(0.1,0.905,total_text,caption_dict) #Title the figure with the metatext
                            else:
                                plt.figtext(0.17,0.88,total_text,caption_dict) #Title the figure with the metatext
                        plt.tight_layout(rect=[0.05,0,0.95,1])
                            
            except ValueError:
                print("Error in sweep!") #Prevents the plotter from failing silently on a large number of files
                continue
            
            #Control figure text: axis labels, axis tick labels, colorbar tick labels, and colorbar label
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + fig.axes[1].get_yticklabels() + [fig.axes[1].yaxis.label]):
                item.set_fontsize(24)
            #No colorbar version
#            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#                item.set_fontsize(26)
#            for item in ([ax.xaxis.label]):
#                item.set_fontsize(0)
 
            if scan_strat == 'RHI':
                if metadisp==True:
                    if contour_bool==True and field==base_field:
                        save_name = "%s%s%s.azi%d.%s.contour%s.%d.png" %(outpath, labeled, filename, azi, field, contour_field, sweepnum)
                    else:
                        save_name = "%s%s%s.azi%d.%s.%d.png" %(outpath, labeled, filename, azi, field, sweepnum)
                else:
                    if contour_bool==True and field==base_field:
                        save_name = "%s%s.azi%d.%s.contour%s.%d.png" %(outpath, filename, azi, field, contour_field, sweepnum)
                    else:
                        save_name = "%s%s.azi%d.%s.%d.png" %(outpath, filename, azi, field, sweepnum)
            else:
                ang = 'ang'
                a_save = ang+a_text
                if metadisp:
                    if contour_bool == True and field == base_field:
                        save_name = "%s%s%s.%s.contour%s.%d.%s.png" %(outpath, labeled, filename, field, contour_field, sweepnum, a_save)
                    else:
                        save_name = "%s%s%s.%s.%d.%s.png" %(outpath, labeled, filename, field, sweepnum, a_save)
                else:
                    if contour_bool==True and field==base_field:
                        save_name = "%s%s.%s.contour%s.%d.%s.png" %(outpath, filename, field, contour_field, sweepnum, a_save)
                    else:
                        save_name = "%s%s.%s.%d.%s.png" %(outpath, filename, field, sweepnum, a_save)
               
            plt.close('all')
            fig.savefig(save_name)
            del display
            gc.collect()
            
        print ("%s %d" % ('Sweep number', sweepnum))
        del azi
        del fig
        
        gc.collect()
    
