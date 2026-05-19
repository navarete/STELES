# load libraries
import datetime
import lacosmic
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
#import glob
#import pandas as pd

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.stats import sigma_clip, mad_std
from astropy.table import Table
from ccdproc import combine
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.signal import medfilt
#from astropy.stats import , mad_std
#from astropy.convolution import convolve, Gaussian1DKernel
#from astropy.modeling.models import Gaussian2D
from astropy.nddata import CCDData
#from astropy.nddata.utils import Cutout2D
#from astropy.time import Time
#from astropy.visualization import simple_norm
#from astropy.table import Table
#from astropy import units as u
#from ccdproc import combine

#import scipy.signal
#from scipy.interpolate import interp1d
#from scipy.optimize import curve_fit
#from scipy.signal import correlate2d, medfilt



def header2python(str_hdr, pixel=True):
    """
    Parse a string containing [XX:XX,YY:YY] to pixels readable by python.

    Parameter
    ---------
        str_hdr : str
        pixel (bool, default = True): if dealing with pixels, subtract 1
        
    """
    str_hdr = str_hdr.replace('[', '')
    str_hdr = str_hdr.replace(']', '')
    str_hdr = str_hdr.replace(',', ' ')
    str_hdr = str_hdr.replace(':', ' ')
    str_out  = np.array(str_hdr.split(' '), dtype=int)
    
    # if pixel is True, subtract 1.
    if pixel: str_out  -= 1
        
    return str_out

def steles_merge_extensions(file, master_bias=None, overscan_correction=False, gain_per_amplifier=[2.22, 2.27], 
                            remove_astrometry=True, show_results=False, figure_out=None):
    """
    Open a FITS image and join its extensions in a single array.

    Args:

        file (str) : full path of the file for merging
        
        master_bias (str, default = None) : name of the master bias used for correction, if None, do overscan
        
        overscan_correction (bool, default = False) : model overscan and remove it
        
        gain_per_amplifier (list, default = [1.0,1.0]) : gain of each amplifier (in e-/ADU)
        
        remove_astrometry (bool, default = True) : remove astrometric keywords from header
        
        show_results (bool, default = False) : print the overscan fit model and
            the merged file on screen
            
    output:
    
        fits_out (ndarray): the merged array
        
        hdr_out (astropy.io.fits.header): the output header

    """
    hdu = fits.open(file)

    if master_bias is not None:
        # store each bias section into a list
        with fits.open(master_bias) as hdul_bias:
            bias_frames = []
            for i in range(0, 2):
                hdu_i  = hdul_bias[i+1]
                bias_frames.append(hdu_i.data)
            # store result as a numpy array
            bias_frames = np.array(bias_frames)
    
    # get gain per amplifier
    gain = np.array(gain_per_amplifier)
    
    if show_results:
        plt.figure(figsize=(8,8))
        
    # Process each extension
    for i in range(0, 2):
        hdu_i  = hdu[i+1]
        hdr_i  = hdu_i.header
        data_i = hdu_i.data

        if show_results:
            plt.subplot(2,3,2 + 3 * i)
            img = plt.imshow(data_i, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(data_i,99.5))
            plt.title(f"Amplifier #{i+1}")


            plt.subplot(2,3,1 + 3 * i)
            plt.plot(np.sum(data_i[50:4095,:],axis=0)/data_i.shape[0], label='original')
            plt.title(f"Mean profile")
        
        if master_bias is not None:
            data_sub = data_i - bias_frames[i,:,:]
        else:
            data_sub = overscan_subtraction(data_i, header2python(hdr_i['BIASSEC']))
            
        data_sub = data_sub
        data_sub *= gain[i]

        if show_results:
            plt.plot(np.sum(data_sub[50:4095,:],axis=0)/data_sub.shape[0], label='bias sub')
            plt.legend()
            plt.ylim(0,)
            plt.xlim(0,data_sub.shape[1])

            plt.subplot(2,3,3 + 3 * i)
            img2 = plt.imshow(data_sub, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(data_i,99.5)* gain[i])
            if master_bias is not None:
                plt.title(f"Bias correction")
            else:
                plt.title(f"Overscan correction")
            plt.colorbar(img2)
            if i == 1:
                plt.tight_layout()
                plt.show()
        
        # stores the new data using single precision float (doubles the size of the data)
        data_type = 'single'
        
        # extract header information in a readable way
        trim = header2python(hdr_i['TRIMSEC'])
        bias = header2python(hdr_i['BIASSEC'])
        dsec = header2python(hdr_i['DETSEC'])

        if i == 0:
            # extract header information in a readable way
            det_size = header2python(hdr_i['DETSIZE'])
            ccdsum   = header2python(hdr_i['CCDSUM'], pixel=False)

            # Save the area that corresponds to each amplifier
            bin_size = np.array(hdr_i['CCDSUM'].split(' '), dtype=int)

            # adjust for binning
            det_size[1] = det_size[1] / bin_size[0]
            det_size[3] = det_size[3] / bin_size[1] 

            # final image size
            nnx = det_size[1] - det_size[0] + 1
            nny = det_size[3] - det_size[2] + 1

            # create output image
            fits_out = np.empty((nny,nnx), dtype=data_type)

            # define new header
            hdr_out = hdr_i
            
            # update keywords
            hdr_out.set('NAXIS1', value=nnx, comment='length of data axis 1')
            hdr_out.set('NAXIS2', value=nny, comment='length of data axis 2')
            hdr_out.set('BZERO',  value=0,   comment='offset data range to that of unsigned short')
            hdr_out.set('EXTEND', value='F', comment='correct extensions')
            ct = str(datetime.datetime.now())
            
            filename = os.path.basename(file)
            hdr_out.set('ST_INPUT', value=filename,            comment="Original raw file")
            hdr_out.set('ST_MEXEC', value=ct,                  comment='STELES_Merge_Extensions execution time')
            hdr_out.set('ST_MERGE', value=True,                comment='STELES extensions were merged')
            hdr_out.set('ST_OVERS', value=overscan_correction, comment='STELES overscan correction')

        # trim the data
        data_out = data_sub[:,trim[0]:trim[1]+1] # add +1 at the last element to match output from IDL
        
        nx = data_out.shape[1]            
            
        # set DETSEC
        dsec[0:1] =  dsec[0:1] / bin_size[0] 
        dsec[2:3] =  dsec[2:3] / bin_size[1] 

        # add the extension to the output file 
        fits_out[dsec[2]:dsec[3]+1,dsec[0]:dsec[1]+1] = data_out

        
    if show_results:
        plt.figure(figsize=(8,4))
        
        plt.subplot(121)
        plt.axvline(1022, ls=':', lw=1, color='black', alpha=0.25, label='Ampl #1 <-> Ampl #2')
        plt.plot(np.sum(fits_out[50:4095,:],axis=0)/fits_out.shape[0])
        plt.xlim(0,fits_out.shape[1])
        plt.legend(fontsize=9)
        plt.title(f"Mean profile")
        
        plt.subplot(122)
        img = plt.imshow(fits_out, origin='lower', aspect='auto', vmin=0, vmax=np.nanpercentile(fits_out,99.5))
        plt.colorbar(img)
        plt.title("Merged frame")
        plt.tight_layout()
        if figure_out is not None:
            plt.savefig(figure_out, format='png', dpi=300)
        plt.show()

    # remove keywords from the output header
    del hdr_out['NEXTEND']
    del hdr_out['DATASEC']
    del hdr_out['BIASSEC']
    del hdr_out['CCDSEC']
    del hdr_out['CCDSIZE']
    del hdr_out['AMPSEC']
    del hdr_out['TRIMSEC']
    del hdr_out['DETSEC']
    if remove_astrometry:
        del hdr_out['PIXSIZE1']
        del hdr_out['PIXSIZE2']
        del hdr_out['PIXSCAL1']
        del hdr_out['PIXSCAL2']
    
    return fits_out, hdr_out


def overscan_subtraction(data, biassec):
    """
    Subtract row-wise overscan using median statistics.
    data:    2D ndarray (raw frame)
    biassec: tuple/list (x1, x2) overscan columns
    """
    overscan = data[:, biassec[0]+1:biassec[1]-1]
    row_bias = np.median(overscan, axis=1)
    
    return data - row_bias[:, None]


def steles_cosmicray_removal(fits_file, 
                             lacosmic_contrast=3, lacosmic_cr_threshold=10, lacosmic_neighbor_threshold=10,
                             lacosmic_gain=2.36, lacosmic_rdnoise=8.47, lacosmic_maxiter=4, 
                             save_fits=True, show_plot=False, return_cosmicray_mask=False):
    """
      A wrapper for lacosmic (https://lacosmic.readthedocs.io/en/stable/index.html) to be used on STELES data
      input parameters are:
        fits_file (str): full path and file name of the original fits file to be cleaned
        gain (float, default=2.36): gain value for the input data (red channel=2.36, blue channel=2.22, in e-/ADU units)
        rdnoise (float, default=8.47): readnoise of the input data (red channel=8.47, blue channel=6.38, in e- units)
        maxiter (int, default=4): maximum number of iterations of lacosmic().
        save_fits (bool, default=True): save results as a new fits file
        show_plot (bool, default=False): if True, show plot on screen of the input file, cleaned file and cosmic ray mask.
        return_cosmicray_mask (bool, default=False): if True, return the cosmic ray mask
        
        future implementations should replace lacosmic.lacosmic() by lacosmic.remove_cosmics()
        
    """

    img, hdr, channel = steles_read_file(fits_file)

    # if using older version of lacosmic
    try:
        img_clean, cr_mask = lacosmic.lacosmic(img, lacosmic_contrast, lacosmic_cr_threshold, lacosmic_neighbor_threshold,
                                               effective_gain=lacosmic_gain, readnoise=lacosmic_rdnoise, maxiter=lacosmic_maxiter)
    except:
        img_clean, cr_mask = lacosmic.lacosmic(img, lacosmic_contrast, lacosmic_cr_threshold, lacosmic_neighbor_threshold,
                                               effective_gain=lacosmic_gain, readnoise=lacosmic_rdnoise, maxiter=lacosmic_maxiter)

    # number of cleaned pixels
    n_cr = np.sum(cr_mask)
    
    if show_plot:
        
        plt.figure(figsize=(15,10))
        plt.subplot(131)
        if channel == 'blue':
            plt.imshow(img.T, origin='lower', vmin=0, vmax=100)
        else:
            plt.imshow(img, origin='lower', vmin=0, vmax=100)
        plt.subplot(132)
        if channel == 'blue':
            plt.imshow(img_clean.T, origin='lower', vmin=0, vmax=100)
        else:
            plt.imshow(img_clean, origin='lower', vmin=0, vmax=100)
        plt.subplot(133)
        if channel == 'blue':
            plt.imshow(cr_mask.T, origin='lower', cmap='hot')
        else:
            plt.imshow(cr_mask, origin='lower', cmap='hot')
        plt.show()
        
    # define new header
    hdr_out = hdr

    # update keywords
    hdr_out.set('ST_CRAY',  value=True, comment='STELES Cosmic Ray removal')
    hdr_out.set('ST_NCRAY', value=n_cr, comment='number of pixels affected by cosmic rays')
    hdr_out
    
    #avoiding flipping the array
    if channel == 'red':
        img_clean = np.flip(img_clean,1)
    
    if save_fits:
        steles_save_file(img_clean, header=hdr_out, file_out=fits_file.replace(".fits","_cleancr.fits"), quiet=True)
        
    if return_cosmicray_mask:
        return img_clean, hdr_out, cr_mask
    else:
        return img_clean, hdr_out
        
def transpose_frame(ccd):
    """ If working with BLUE channel, transpose the array so the orders increase along the x-axis"""
    
    # Transpose the data
    ccd.data = ccd.data.T
    ccd.data = np.flip(ccd.data,axis=1)

    # Swap WCS / header axis-related keywords if they exist
    header = ccd.header

    # Common FITS keywords that need swapping for a transpose
    for key_prefix in ["NAXIS", "CRPIX", "CRVAL", "CDELT", "CTYPE", "CUNIT", "CD", "PC"]:
        # Handle the ones with two axis numbers
        if key_prefix in ["CD", "PC"]:
            keys_to_swap = []
            for i in [1, 2]:
                for j in [1, 2]:
                    k = f"{key_prefix}{i}_{j}"
                    if k in header:
                        keys_to_swap.append((i, j, header[k]))
            # Swap (1,2) with (2,1)
            for i, j, val in keys_to_swap:
                header[f"{key_prefix}{j}_{i}"] = val
        else:
            k1 = f"{key_prefix}1"
            k2 = f"{key_prefix}2"
            if k1 in header and k2 in header:
                header[k1], header[k2] = header[k2], header[k1]

    # Save transposed CCDData
    return ccd

def load_list_fits_data(directory, file_list, extension='.fits', return_file_names=False):
    """
    Load FITS files from a directory whose filenames start with a given prefix.
    
    Parameters:
        directory (str): Path to the directory containing FITS files.
        prefix (str): Prefix to match filenames (e.g., 'bias', 'flat', 'science').

    Returns:
        list of CCDData: List of loaded CCDData objects.
    """
    files = []
    filepaths = []
    for file in file_list:
        files.append(file+extension)
        filepaths.append(directory+file+extension)
    #files = sorted([f for f in os.listdir(directory) if f.endswith(extension) and f.startswith(prefix)])
    #filepaths = [os.path.join(directory, f) for f in files]
    if return_file_names:
        return [CCDData.read(f, unit='adu') for f in filepaths], files
    else:
        return [CCDData.read(f, unit='adu') for f in filepaths]    
    

def process_frames(ccddata_frames, header=None, sigma_clip=5, interpolate_bad_columns=False, interpolate_bad_columns_sigma=2, interpolate_bad_columns_iterations=10, transpose=False):
    """ interpolate bad columns and combine frames using avsigclip method.
        uses list generated by load_fits_data()
        runs interpol_bad_columns() - not well implemented yet
        then combine the images using AVSIGCLIP method.
    """
    
    # start list to store the reduced images
    reduced_frames = []
    
    # loop into the raw images
    for i, frame in enumerate(ccddata_frames):
        if interpolate_bad_columns:
            frame = interpol_bad_columns(frame, threshold_sigma=interpolate_bad_columns_sigma, iterations=interpolate_bad_columns_iterations, ccddata=True)
        # store the processed frames into a list
        reduced_frames.append(frame)
    
    # combine with average and sigma clipping method
    master_frame = combine(reduced_frames, method='average', sigma_clip=True, scale=None,
                           sigma_clip_low_thresh=sigma_clip, sigma_clip_high_thresh=sigma_clip,
                           sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)
    
    # add header keywords
    master_frame.header.set('ST_IPCOL', value=interpolate_bad_columns, comment='Interpolated bad columns' )
    master_frame.header.set('ST_COMBI', value='avsigclip',             comment='Image combining method'   )
    master_frame.header.set('ST_CSIGC', value=sigma_clip,              comment='Sigma clipping threshold' )
    master_frame.header.set('ST_NCOMB', value=len(reduced_frames),     comment='Number of combined images')
    
    return master_frame         


def steles_read_file(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            image_data   = hdul[0].data
            image_header = hdul[0].header
    except:
        raise ValueError("Check input fits file name and path.")
            
    channel = 'red' if image_header['FPA'] == "RED" else 'blue'   

    # make orders to increase from left to right
    if channel == 'red':
        image_data = np.flip(image_data,axis=1)
    
    return image_data, image_header, channel

def scale_saturated_thar_spectrum(infolder, thar_files, saturation_threshold=65000, output_file=None, return_hdu=False):
    """
    Replaces saturated pixels in a long-exposure FITS image with scaled values 
    from a short-exposure FITS image. Ideal for ThAr lamp spectrum.

    Parameters
    ----------
    infolder : str
        Path to the FITS files.
    thar_files : str
        list of FITS files, generally a short and a long exposure.
    saturation_threshold : float, default=65000
        Pixel value above which pixels are considered saturated.
    output_file : str, optional
        Name of the resulting FITS file. if not provided, will add "_scaled.fits" at the end of the fits file.
    """

    # start list to store data
    data    = []
    headers = []
    texp    = []
    
    # Read data
    for file in thar_files:
        with fits.open(infolder+file) as hdul:
            #data.append(hdul[0].data.astype(np.float32))
            data.append(hdul[0].data.astype(np.float64))
            headers.append(hdul[0].header)
            texp.append(hdul[0].header["AEXPTIME"])
    
    texp = np.array(texp)
    data = np.array(data)
    
    # get index for long and short exposures
    file_long = np.argmax(texp)
    file_short = np.argmin(texp)
    
    # get exposure times
    texp_long  = texp[file_long]
    texp_short = texp[file_short]

    # saves output header as the long exposure
    header_out = headers[file_long]
    
    # retrieve long and short exposures
    short_data = data[file_short,:,:]
    long_data  = data[file_long,:,:]
    
    # Scale short exposure to match long exposure
    scale_factor = texp_long / texp_short
    scaled_short = short_data * scale_factor

    # Identify saturated pixels
    saturated_mask = long_data >= saturation_threshold

    # Replace saturated pixels with scaled short exposure
    corrected_data = np.copy(long_data)
    corrected_data[saturated_mask] = scaled_short[saturated_mask]

    # Save result
    
    if return_hdu:
        # Create primary HDU
        hdu = fits.PrimaryHDU(data=corrected_data, header=header_out)
        return hdu
    else:
        if output_file is None:
            output_file=infolder+thar_files[0].replace(".fits","_scaled.fits")

        fits.writeto(output_file, corrected_data, header_out, overwrite=True)

        print(f"Output saved to {output_file}")
        


def read_master_tracing(trace_data, trace_default_directory, n_orders, poly_order=4, sigma_clip_threshold=5, displ_max_noise=1.5, debug=False, show_plot=False):
    """ Read positions of the master tracing measurements and perform a polynomial fit using fit_trace_stellar_profile() """
    # start an empty list
    trace_poly_coeffs_init = []

    # loop through the orders to create a list of tracing coefficients
    for n in range(n_orders):
    #nn=38
    #for n in range(nn,nn+1): 
        #show_plot=False #False #if n < 35 else True
        trace_coeffs = fit_trace_stellar_profile(trace_data, trace_default_directory + f"trace_xy_{n+1:02d}.txt", 
                                                 order=n+1, poly_order=poly_order, sigma_clip_threshold=sigma_clip_threshold, 
                                                        show_plot=show_plot, displ_max_noise=displ_max_noise, debug=debug)
        # append the tracing coefficients to the list
        trace_poly_coeffs_init.append(trace_coeffs)
        
    return trace_poly_coeffs_init


def read_trace(filename="trace.txt"):
    tbl = Table.read(filename, format="ascii")
    trace_x = np.array(tbl["trace_x"])
    trace_y = np.array(tbl["trace_y"])
    return trace_x, trace_y


def fit_trace_stellar_profile(fits_file, trace_file, order, poly_order=4, sigma_clip_threshold=3, y_range=[500,3500],
                              show_plot=True, displ_max_noise=10, debug=True):
    """
    Trace a stellar profile along the y-axis and fit a polynomial x=f(y).

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    x_guess : float
        Initial guess of the x position at y_center.
    y_center : int
        Central y position (default=2048).
    half_width : int
        Half-width of the extraction box around the current x position.
    step : int
        Step size in y when tracing.
    poly_order : int
        Degree of the polynomial fit.

    Returns
    -------
    y_points : ndarray
        Y positions traced.
    x_points : ndarray
        Measured X positions along the trace.
    poly_coeffs : ndarray
        Polynomial coefficients for x=f(y).
    """

    # Load image
    if isinstance(fits_file, str):
        image = fits.getdata(fits_file)
    else:
        image = np.asarray(fits_file)

    # Load image
    #image = fits.getdata(fits_file)
    ny, nx = image.shape

    # Estimate background noise
    background = np.median(image)
    noise = 1.4826 * np.median(np.abs(image - background))
    
    # Define y-range
    if y_range is None:
        y_min, y_max = 0, ny - 1
    else:
        y_min, y_max = y_range

    x_points, y_points = read_trace(trace_file)

    # apply sigma-clip on datapoints
    clip_mask = ~sigma_clip(x_points, sigma=sigma_clip_threshold, maxiters=5).mask
    #y_points = y_points[mask]
    #x_points = x_points[mask]
    
    # Fit polynomial x=f(y)
    coeffs = np.polyfit(y_points[clip_mask], x_points[clip_mask], deg=poly_order)
    poly = np.poly1d(coeffs)

    res = abs(poly(y_points) - x_points)
    
    # Fit polynomial x=f(y)
    coeffs = np.polyfit(y_points[clip_mask], x_points[clip_mask], deg=poly_order)
    poly = np.poly1d(coeffs)

    # Plot trace
    if show_plot:
        plt.figure(figsize=(10,10))
        plt.imshow(image, origin="lower", cmap="gray", aspect='auto', vmin=0, vmax=displ_max_noise*noise)
        plt.plot(x_points[clip_mask], y_points[clip_mask], '.g', markersize=5, label=f'Measured [{np.sum(clip_mask)}/{len(x_points)}]')
        plt.plot(x_points[~clip_mask], y_points[~clip_mask], 'xr', markersize=5, label=f'Outliers [{np.sum(~clip_mask)}]')
        plt.plot(poly(y_points), y_points, '--', label=f'Trace [ndeg={poly_order}]')
        plt.legend()
        pix_border=10
        plt.xlim(min(x_points)-pix_border,max(x_points)+pix_border)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Order={order}")
        plt.show()

    return coeffs



def refit_trace(trace_data, trace_channel, trace_poly_coeffs, trace_directory, trace_window=20, trace_ndeg=4, sigma_clip_threshold=3., skip_orders=[None], plot_results=True, plot_transpose=True, plot_percentile=[1,99], quiet=True, output_file_prefix=None):
    
    # avoid warning messages
    if quiet:
        warnings.filterwarnings('ignore', message="fit may be poorly conditioned")
    

    fitter     = fitting.LinearLSQFitter()

    refined_trace_poly = []
    for n, trace_poly in enumerate(trace_poly_coeffs):

        poly_guess = np.poly1d(trace_poly)

        trace_db_file = trace_directory + f"trace_xy_{n+1:02d}.txt"

        x_vals, y_vals = read_trace(trace_db_file)

        
        if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):

            refit_trace = True
            
            # --- 2. Prepare arrays for refined trace ---
            x_refined  = np.zeros_like(y_vals, dtype=float)
            x_shift    = np.zeros_like(y_vals, dtype=float)
            
            
            # --- 3. Loop through y positions to refine the trace ---
            for i, y in enumerate(y_vals):

                x_guess = poly_guess(y)

                # Define window range safely to avoid adding positions outside the CCD
                x_min = max(int(x_guess - trace_window), 0)
                x_max = min(int(x_guess + trace_window), trace_data.shape[1]-1)

                # sum the trace over a few pixels in the dispersion direction
                n_pix_sum = 3

                if n_pix_sum > 0:
                    profile = np.sum(trace_data[int(y-n_pix_sum):int(y+n_pix_sum), int(x_min):int(x_max+1)], axis=0)
                else:
                    profile = trace_data[int(y), int(x_min):int(x_max+1)]

                x_window = np.arange(x_min, x_max+1)

                # Smooth the profile to reduce noise / spectral line impact
                profile_smooth = medfilt(profile, kernel_size=3)
                # position of the peak
                x_refined_peak = x_window[np.argmax(profile)]
                peak_max = np.argmax(profile)
                #print("x_refined_peak=",x_refined_peak)
                try:
                    peak_max2, _ = refine_peak_positions(profile_smooth, [peak_max], window=trace_window, 
                                                         plot_fits=False, n_col=1, output=None)
                    x_refined_peak2 = x_window[peak_max]
                except:
                    x_refined_peak2 = x_refined_peak

                x_refined[i] = x_refined_peak2

            # --- 4. Keep only valid positions ---
            y_valid = y_vals#[mask_valid]
            x_valid = x_refined#[mask_valid]

            # --- 5. Sigma-clip outliers relative to first guess ---
            poly_fit = fitter(models.Polynomial1D(degree=trace_ndeg), y_valid, x_valid)
            residuals = x_valid - poly_guess(y_valid)
            mask_clip = sigma_clip(residuals, sigma=sigma_clip_threshold).mask

            x_clean = x_valid[~mask_clip]
            y_clean = y_valid[~mask_clip]

            shift_trace = np.median(x_shift)

            # --- 6. Fit refined polynomial ---
            poly_refined = fitter(models.Polynomial1D(degree=trace_ndeg), y_clean, x_clean)
            poly_coeffs = poly_refined.parameters
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            poly_coeffs = np.flip(poly_coeffs)
            refined_trace_poly.append(poly_coeffs)


        else: # if ( skip_orders[0] is None ) or ( n+1 not in skip_orders ):
            print(f"Skipping tracing refit for order n={n+1}")
            
            refit_trace = False
            
            # I still don't know why, but I need to flip the 'poly_coeffs' array so it will work fine...
            #poly_coeffs = np.flip(poly_guess)
            poly_coeffs = np.array(poly_guess)
            refined_trace_poly.append(poly_coeffs)
        

        dx = np.max(x_vals) - np.min(x_vals)

        # --- 7. Plot result for visual check ---

        if plot_results:

            xmin, xmax = np.min(x_vals) - dx*0.2, np.max(x_vals) + dx*0.2
            if xmin < 0:
                xmin = 0
            if xmax > trace_data.shape[1]:
                xmax = trace_data.shape[1]
                

            y_range=[ 500,1500] if trace_channel == 'blue' else [1000,3000]

            vmin, vmax = np.percentile(trace_data[y_range,int(xmin):int(xmax)], plot_percentile)
                
            if plot_transpose:
                plt.figure(figsize=(14,4))
                plt_img = plt.imshow(np.transpose(trace_data), origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(y_vals, poly_guess(y_vals),   ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(y_vals, poly_refined(y_vals), ls='--', color='red',  lw=1, label='Refined trace')
                plt.scatter(y_vals,              x_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace:
                    plt.scatter(y_valid[~mask_clip], x_valid[~mask_clip], s= 5, color='black', label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(y_valid[mask_clip],  x_valid[mask_clip],  s=10, marker='x', color='red', label=f'Excluded points ({np.sum(mask_clip)})')
                plt.ylim(xmin,xmax)
                
                #plt.xlim(min(y_vals)-50,max(y_vals)+50)
                ymin = max(min(y_vals)-50, 0)
                ymax = min(max(y_vals)+50, trace_data.shape[0])
                
                plt.xlim(ymin,ymax)
                plt.ylabel('X pixel')
                plt.xlabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(f'Order Tracing Refinement (n={n+1})')
                plt.tight_layout()
                if output_file_prefix is not None:
                    plt.savefig(output_file_prefix + f"_n{(n+1):02d}.png", dpi=300)
                plt.show()

            else:
                plt.figure(figsize=(4,8))
                plt_img = plt.imshow(trace_data, origin='lower', aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
                plt.plot(poly_guess(y_vals),   y_vals, ls=':',  color='yellow', lw=1, label='Initial guess')
                if refit_trace:
                    plt.plot(poly_refined(y_vals), y_vals, ls='--', color='red',  lw=1, label='Refined trace')
                plt.scatter(x_vals,              y_vals,              s=10, marker='+', color='blue', label='initial fitting')
                if refit_trace:
                    plt.scatter(x_valid[~mask_clip], y_valid[~mask_clip], s= 5, color='black', label=f'Valid points ({np.sum(~mask_clip)}/{len(mask_clip)})')
                    plt.scatter(x_valid[mask_clip],  y_valid[mask_clip],  s=10, marker='x', color='red', label=f'Excluded points ({np.sum(mask_clip)})')
                plt.xlim(xmin,xmax)
                plt.ylim(min(y_vals)-50,max(y_vals)+50)
                plt.xlabel('X pixel')
                plt.ylabel('Y pixel')
                plt.legend()
                plt.colorbar(plt_img, label='Counts (ADU)', pad=0.03, fraction=0.1)
                plt.title(f'Order Tracing Refinement (n={n+1})')
                plt.tight_layout()
                if output_file_prefix is not None:
                    plt.savefig(output_file_prefix + f"_n{(n+1):02d}.png", dpi=300)
                plt.show()
        
        
    return refined_trace_poly

def save_polyfits_tracing_csv(order, polyfits_x_along_y, output_csv, show_messages=False):
    """
        Read a list contaning the polynimal coefficients and save them as a csv file for future use. 
        The first row contains the order information, and the following rows will store the Polynomial coefficients.
    
    """

    # Convert to string representation of coefficients
    string_array = np.empty_like(polyfits_x_along_y)
    
    if polyfits_x_along_y.ndim > 1:
        
        pixel_array = order
        for i in range(polyfits_x_along_y.shape[0]):   # spatial
            for j in range(polyfits_x_along_y.shape[1]): # spectral lines
                coeffs = polyfits_x_along_y[i, j].parameters  # list of coefficients
                coeffs_txt = np.array2string(coeffs, separator=',')
                coeffs_txt = coeffs_txt#.replace(' ','').replace('[','').replace(']','')
                string_array[i, j] =  coeffs_txt #','.join(map(float, coeffs))
                
        string_array = np.vstack((pixel_array,string_array))
        
    else:
        pixel_array = np.array([1])
        for i in range(polyfits_x_along_y.shape[0]):   # spatial
            coeffs = polyfits_x_along_y[i]#.parameters  # list of coefficients
            coeffs_txt = np.array2string(coeffs, separator=',')
            coeffs_txt = coeffs_txt#.replace(' ','').replace('[','').replace(']','')
            string_array[i] =  coeffs_txt #','.join(map(float, coeffs))

        string_array = np.concatenate((pixel_array,string_array))
        
    # Save to CSV
    np.savetxt(output_csv, string_array, fmt='%s', delimiter=';')
    if show_messages:
        print(f'File {output_csv} was created.')


def read_polyfits_tracing_csv(directory, file_prefix, n_orders):
    """
        Read a CSV file created by save_polyfits_tracing_csv() and returns a numpy array of with tracing polynomial coefficients. 
    
    """
    polyfits = [] 
    
    trace_poly_coeffs = []
    
    for order in np.arange(1,n_orders+1): 
    
        file_csv = directory + f"{file_prefix}{order:02d}.csv"

        # Load string array
        loaded_strings = np.genfromtxt(file_csv, dtype=str, delimiter=';')
        coeff_strings = loaded_strings[1:]
                
        # Convert back to Polynomial1D array
        #polyfits = [] #np.empty_like(np.arange(n_orders), dtype=object)
        #print(polyfits.shape)
        coeffs = []
        for i in range(coeff_strings.shape[0]):
            coeffs_txt = coeff_strings[i]
            #print(i, coeffs_txt)
            #coeffs = list(map(float, coeffs_txt.split(',')))
            coeffs.append(float(coeffs_txt))
        polyfits.append(np.array(coeffs))
        #print(order, coeffs)
        #degree = len(coeff_strings) #- 1
        #print(degree)
        #polyfits[order-1] = Polynomial1D(degree=degree, **{f'c{k}': v for k, v in enumerate(coeffs)})  
        #polyfits.append(Polynomial1D(degree=degree, **{f'c{k}': v for k, v in enumerate(coeffs)}) ) 
        #print("-----")
    #polyfits = polyfits[:-1]
                
    return polyfits


def linearize_trace(fits_file, poly_coeffs, x_left=-8, x_right=12, 
                    y_range=None, flip_axis=False, displ_percent=[1,99], show_plot=True, output_file=None):
    """
    Fast linearization of a curved stellar trace using vectorized interpolation.

    Parameters
    ----------
    fits_file : str or ndarray
        Input FITS file or 2D numpy array.
    poly_coeffs : ndarray
        Polynomial coefficients for x = f(y).
    x_left : int
        Pixels to extract to the left of the trace.
    x_right : int
        Pixels to extract to the right of the trace.
    y_range : tuple (y_min, y_max), optional
        Vertical range of pixels to extract. If None, full range is used.
    output_file : str
        Path to save the linearized FITS file.
    show_plot : bool
        If True, plot the linearized image.

    Returns
    -------
    linearized : ndarray
        Rectified 2D array (width, height) where width = x_right - x_left + 1.
    """

    # Load image
    if isinstance(fits_file, str):
        image = fits.getdata(fits_file)
    else:
        image = np.asarray(fits_file)

    ny, nx = image.shape
    poly = np.poly1d(poly_coeffs)

    # Define y-range
    if y_range is None:
        y_min, y_max = 0, ny - 1
    else:
        y_min, y_max = y_range

    y_min = max(0, y_min)
    y_max = min(ny - 1, y_max)

    # Prepare arrays
    width = x_right - x_left + 1
    height = y_max - y_min + 1

    y_indices = np.arange(y_min, y_max + 1)
    x_offsets = np.arange(x_left, x_right + 1)

    # Compute x centers for all y
    x_centers = poly(y_indices)  # shape: (height,)

    # mask out tracing curves lying outside the image.
    mask = ( x_centers < 0 ) ^ (x_centers > nx )
    
    # Build coordinate grids
    y_grid = np.repeat(y_indices[np.newaxis, :], width, axis=0)  # (width, height)
    x_grid = (x_centers[np.newaxis, :] + x_offsets[:, np.newaxis])  # (width, height)

    # Interpolate in one call
    coords = np.array([y_grid, x_grid])
    linearized = map_coordinates(image, coords, order=1, mode='nearest')

    linearized[:,mask] = 0.0

    if flip_axis:
        linearized = np.flip(linearized,axis=1)
    
    # Save to FITS
    if output_file is not None:
        fits.writeto(output_file, linearized, overwrite=True)
        print(f"Linearized trace saved to {output_file}")
    # Optional plot
    if show_plot:
        plt.figure(figsize=(16, 4))
        
        vmin, vmax = np.percentile(linearized, displ_percent)
        
        img = plt.imshow(linearized, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        plt.axhline(width*0.5, ls='--')
        plt.xlabel("Extraction Width (pixels)")
        plt.ylabel("Y (linearized)")
        plt.title(f"Linearized Trace (y: {y_min}-{y_max})")
        plt.colorbar(img, label="Flux")
        plt.tight_layout()
        plt.show()

    return linearized
