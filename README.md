# PyART Processing

Environment Analytics code written to aid in the processing, analysis, and visualization of radar data through the [ARM PyART](https://github.com/ARM-DOE/pyart) python package. Functions and documentation were originally created by Sara Berry in 2015. Software has been maintained by Daniel Hueholt since 2018.

## Code description
### Environment Analytics PyART processing toolkit
**calculated_fields**: Functions to derive variables such as vertical divergence of horizontal velocity or the Rasmussen snow rate from observed fields.

**colorbars**: Saves example colorbars for the colormaps in colormap. Useful for posters and presentations.

**colormap**: Contains functions controlling colormaps. Includes Matthew Miller's "LCH_spiral" luminance-conserving colormap.

**gen_fun**: Miscellaneous functions for tasks such as parsing a list of files or generating an image filename.

**Master_plotter**: Plots and saves Plan Position Indicator (PPI) or Range Height Indicator (RHI) images for a given field. Also handles contour overlays, file naming, and similar tasks.

**quality_control**: Contains many quality control functions, including the dealiasing manager, noise masks, and mountain removal.

**run_fun**: Manages the processing of radar data using the ARM PyART package and various custom tools.

**start_script**: Sets variables and instantiates a child process in which run_fun can take place.

### Modified PyART files
**nexrad_level2**: Updated to use np.frombuffer instead of np.fromstring, which is now deprecated.

## Sources and Credit
PyART citation:
Helmus, J.J. & Collis, S.M., (2016). The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software. 4(1), p.e25. DOI: [http://doi.org/10.5334/jors.119](http://doi.org/10.5334/jors.119)
