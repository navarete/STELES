import matplotlib.pyplot as plt
import numpy as np
import os


from astropy.io import fits
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

import warnings

def steles_read_file(fits_file):
    try:
        with fits.open(fits_file) as hdul:
            image_data   = hdul[0].data
            image_header = hdul[0].header
    except:
        raise ValueError("Check input fits file name and path.")
    
    # generalize for any fits file
    try:
        channel = 'red' if image_header['FPA'] == "RED" else 'blue'   

        # make orders to increase from left to right
        if channel == 'red':
            image_data = np.flip(image_data,axis=1)
    except:
        channel = None
        
    return image_data, image_header, channel

    
#def save_spectrum_to_fits(spectrum, output, header=None, overwrite=True):
#    """
#    Save 1D spectrum with wavelength axis as a FITS file.
#
#    Parameters:
#        filename : str
#            Output FITS filename.
#        ccddata (CCDData): The extracted spectrum in CCDData format, contaning the wavelength solution in the header.
#    """
#    
#    # Create primary HDU
#    hdu = fits.PrimaryHDU(data=spectrum, header=header) if header is not None else fits.PrimaryHDU(data=spectrum)
#    hdu.writeto(output, overwrite=overwrite)
#    print(f"Saved: {output}")

# wavelength solution
def collapse_spectrum2d(image, row_range=None, return_average=False):
    # read a 3d image, collapse the spatial pixels and returns a 2d frame
    n_orders, n_spat, n_spec = image.shape
    image_out = np.empty((n_orders, n_spec))
    # start the loop
    for order in np.arange(1,n_orders+1):
        if row_range is None:
            image_out[order-1,:] = np.sum(image[order-1,:,:], axis=0)
            n_rows = n_spat
        else:
            image_out[order-1,:] = np.sum(image[order-1,row_range[0]:row_range[1],:], axis=0)
            n_rows = row_range[1]-row_range[0]+1
            
    if return_average:
        image_out /= n_rows
        
    return image_out


# wavelength calibration functions
def gaussian_kernel(x, amp, mu, sigma, offset):
    """ Gaussian kernel to search for arc lines """
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset

def refine_peak_positions(spectrum, initial_guesses, window=10, plot_fits=False, n_col=None, output=None):
    """
    Refine peak positions from initial pixel guesses by fitting Gaussians.
    
    Parameters:
        spectrum (np.ndarray): 1D array of arc lamp flux.
        initial_guesses (list): Approximate pixel locations of arc peaks.
        window (int): Half-width of region to fit around each guess.
        plot_fits (bool): If True, plots the fitted region.

    Returns:
        refined_positions (np.ndarray): Subpixel centroid positions of each arc line.
    """
    refined_positions = []
    peak_fluxes = []
    
    n_peaks = len(initial_guesses)
    if n_col is None:
        n_col = n_peaks if n_peaks <= 4 else 4
    n_row = int(n_peaks / n_col)
    if n_col * n_row < n_peaks:
        n_row += 1
    
    if plot_fits:
        plt.figure(figsize=(n_col * 4, n_row * 3))
    
    for n, guess in enumerate(initial_guesses):
        
        guess = int(guess) # force int
        
        # Define local fitting window        
        x_fit = np.arange(guess - window, guess + window + 1)
        y_fit = spectrum[x_fit]

        # Initial parameters: amplitude, center, width, offset
        amp0 = y_fit.max() - y_fit.min()
        mu0 = guess
        sigma0 = 3.0
        offset0 = y_fit.min()

        try:
            popt, _ = curve_fit(gaussian_kernel, x_fit, y_fit, p0=[amp0, mu0, sigma0, offset0])
            
            # maybe add a max_fwhm_threshold parameter and exclude negative flux_peak values from the results...
            # the data is already cleaned by rms threshold filtering on the fit_wavelength_solution()
            
            flux_peak  = popt[0]
            refined_mu = popt[1]
            fwhm       = abs(popt[2]) * 2.355
            
            refined_positions.append(refined_mu)
            peak_fluxes.append(flux_peak)

            if plot_fits:
                plt.subplot(n_row,n_col,n+1)
                plt.plot(x_fit, y_fit, 'k.')
                plt.plot(x_fit, gaussian_kernel(x_fit, *popt), 'r-', label='Fit')
                plt.axvline(refined_mu, color='blue', linestyle='--', label=r'x$_{peak}$, FWHM='+f'{refined_mu:.2f}, {fwhm:.2f}')
                plt.title(f'Line near x={guess}')
                plt.grid(True)
                if n == 0:
                    plt.ylabel('Flux')
                plt.xlabel('Pixel')
                plt.legend()
                    
        #except RuntimeError:
        except:
            print(f"Fit failed near x={guess}")
            refined_positions.append(np.nan)  # fallback to original guess
            peak_fluxes.append(np.nan)  # fallback to original guess

    if plot_fits:
        plt.suptitle("Refine peak positions for the Arc lamp spectrum",y=1.0)
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()            
        
    return np.array(refined_positions), np.array(peak_fluxes)


def extract_spectrum1d(arc_frame, row_range=(100, 110)):
    """
    Extract 1D spectrum by summing rows across a spatial region.
    
    Parameters:
        arc_frame (CCDData): The 2D arc frame.
        row_range (tuple): Tuple (y1, y2) defining rows to extract.

    Returns:
        1D numpy array: Extracted arc spectrum (flux vs pixel).
    """
    data = arc_frame
    spectrum_1d = np.sum(data[row_range[0]:row_range[1], :], axis=0)
    return spectrum_1d

def wavelength_calibration(x, y, order, x_range=None, ndeg=3, show_plot=True, figsize=(12,5), output=None):

    # start the fitter
    linfitter = LinearLSQFitter()
    spatial_model = Polynomial1D(degree=ndeg)
    fit_wavelength = linfitter(model=spatial_model, x=x, y=y)
    
    wave_xi = fit_wavelength(x)
    
    x_axis = x if x_range is None else np.arange(x_range[0],x_range[1],1)
    wave_x = fit_wavelength(x_axis)
    
    # compute RMS of the fitting
    residuals = y-wave_xi
    rms = np.sqrt(np.sum(abs(residuals)**2)/len(residuals))
    
    if show_plot:
        plt.figure(figsize=figsize)
        
        plt.subplot(2,1,1)
        plt.plot(x,y,'+',label='Measurements')
        plt.plot(x_axis,wave_x,color='red', alpha=0.5, label=f'Model (deg={ndeg})')
        plt.ylabel(r'Wavelength ($\AA$)')
        plt.xlabel('Dispersion axis (pixel)')
        if x_range is not None:
            plt.xlim(x_range)
        plt.legend()
        # plot residuals
        plt.subplot(2,1,2)
        plt.plot(y,residuals,'+',label=fr'RMS={rms:.6f} $\AA$' )
        plt.axhline(0,ls='--',color='grey')
        plt.legend()
        if x_range is not None:
            plt.xlim(min(wave_x),max(wave_x))
        plt.ylabel(r'Residuals ($\AA$)')
        plt.xlabel(r'Wavelength ($\AA$)')
        # make y-axis centered at 0.
        ymax = np.max(abs(residuals))*1.1
        plt.ylim(-ymax,ymax)
        plt.suptitle(f"Wavelength calibration (n={order:02d})", va='bottom')
        plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()
        
    return fit_wavelength, rms

def wavelength_calibration_iter(x, y, order,
                                x_range=None, ndeg=3, n_iter=3, rms_threshold=3.0,
                                show_plot=True, figsize=(12, 5), output=None, return_mask=False):
    
    # Ensure numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Initial mask: all points included
    mask = np.ones(len(x), dtype=bool)
    
    fitter = LinearLSQFitter()
    
    for i in range(n_iter):
        
        # Fit using only current good points
        model = Polynomial1D(degree=ndeg)
        fit_wavelength = fitter(model, x[mask], y[mask])
        
        # Evaluate on all points
        y_fit_all = fit_wavelength(x)
        
        # Residuals
        residuals = y - y_fit_all
        
        # RMS using only current good points
        rms = np.sqrt(np.sum((residuals[mask])**2)/np.sum(mask))
        #print(f"n_iter={i}, rms={rms}, residuals[mask]={residuals[mask]}, residuals[mask]/rms={residuals[mask]/rms}")
              
        # New mask based on threshold
        new_mask = np.abs(residuals) < (rms_threshold * rms)
        
        # Check convergence (no change in mask)
        if np.array_equal(new_mask, mask):
            print(f"Converged at iteration {i+1}")
            break
        
        mask = new_mask
    
    # Final model
    final_fit = fit_wavelength
    
    # Prepare plotting axis
    x_axis = x if x_range is None else np.arange(x_range[0], x_range[1], 1)
    y_axis = final_fit(x_axis)
    
    if show_plot:
        plt.figure(figsize=figsize)
        
        # --- Top panel: fit ---
        plt.subplot(2, 1, 1)
        plt.plot(x[mask], y[mask], '+', label='Used points')
        plt.plot(x[~mask], y[~mask], 'x', label='Rejected points', alpha=0.6)
        plt.plot(x_axis, y_axis, color='red', alpha=0.7,
                 label=f'Polynomial fit (deg={ndeg})')
        
        plt.ylabel(r'Wavelength ($\AA$)')
        plt.xlabel('Dispersion axis (pixel)')
        if x_range is not None:
            plt.xlim(x_range)
        plt.legend()
        
        # --- Bottom panel: residuals ---
        plt.subplot(2, 1, 2)
        plt.plot(y[mask], residuals[mask], '+',
                 label=fr'RMS={rms:.6f} $\AA$')
        plt.plot(y[~mask], residuals[~mask], 'x', alpha=0.6)
        
        plt.axhline(0, ls='--')
        plt.ylabel(r'Residuals ($\AA$)')
        plt.xlabel(r'Wavelength ($\AA$)')
        
        ymax = np.max(np.abs(residuals)) * 1.1
        plt.ylim(-ymax, ymax)
        
        if x_range is not None:
            plt.xlim(min(y_axis), max(y_axis))
        
        plt.legend()
        plt.suptitle(f"Wavelength calibration (n={order:02d})")
        plt.tight_layout()
        
        if output is not None:
            plt.savefig(output, dpi=300)
        
        plt.show()
    
    if return_mask:
        return final_fit, rms, mask
    
    return final_fit, rms

def fit_wavelength_solution(arc_spectrum, ThAr_peak_wavelengths, ThAr_peak_pixels, order,
                            fit_window=11, max_fwhm_threshold=None,
                            wavesolution_ndeg=3, wavesolution_niter=3, wavesolution_rms_threshold=3.0,
                            ref_thar_directory='C:/GoogleDriveLNA/SOAR/Steles/orders/', ref_thar_file='ThAr_atlas_3000_10600_75k',
                            save_plots=False, plot_directory=None, plot_prefix='file', plot_extension='.png', displ_percent_max=99):
    """ a wrapper for performing wavelength solution
    arc_spectrum (np.array): 1d spectrum
    ThAr_peak_wavelengths (np.array): list of peak position wavelengths (in Angstrom)
    ThAr_peak_pixels (np.array): list of peak position (in pixels)
    fit_window (int): set window to search for refined pixel positions (in pixels)
    wavesolution_ndeg (int): polynomial degree for wavelength solution
    save_plots (bool): save plots
    plot_directory (str): directory for saving the plots
    
    returns:
    wavelength_axis (np.array): wavelength axis containing the same number of elements of 'arc_spectrum'
    rms_wavesol (float): RMS of the wavelength solution (in Angstrom)
    
    """
    
    if save_plots:
        output_peak=os.path.join(plot_directory, plot_prefix + "_peak_positions" + plot_extension)
        output_cal =os.path.join(plot_directory, plot_prefix + "_wavelength_solution" + plot_extension)
        output_thar_ref =os.path.join(plot_directory, plot_prefix + "_refThAr" + plot_extension)
    else:
        output_peak=None
        output_cal =None
    
    # read the ThAr spectrum
    #ref_flux, ref_wavelength, l_wave, l_name = read_ref_thar(ref_thar_directory, ref_thar_file)
    ref_flux, ref_wavelength = read_ref_thar_simple(ref_thar_directory, ref_thar_file)
    
    # refine center positions for the HeAr lines used for wavelength calibration
    ThAr_peak_pixels_refined, _ = refine_peak_positions(arc_spectrum, ThAr_peak_pixels,
                                                        window=fit_window,
                                                        plot_fits=True, n_col=5, output=output_peak)

    # 7. wavelength calibration
    #wavesolution_ndeg=3 # DEFAULT
    #fit_wavelength, rms_wavesol = wavelength_calibration(ThAr_peak_pixels_refined, ThAr_peak_wavelengths, order,
    #                                                     x_range=[1,2048], ndeg=wavesolution_ndeg,
    #                                                     show_plot=True, figsize=(15,5), output=output_cal)
    
    fit_wavelength, rms_wavesol = wavelength_calibration_iter(ThAr_peak_pixels_refined, ThAr_peak_wavelengths, order,
                                                              n_iter=wavesolution_niter, rms_threshold=wavesolution_rms_threshold,
                                                              x_range=[1,2048], ndeg=wavesolution_ndeg,
                                                              show_plot=True, figsize=(15,5), output=output_cal)
    
    
    
    # get wavelength axis
    x_axis = np.arange(len(arc_spectrum.data))
    wavelength_axis = fit_wavelength(x_axis)

    # scale atlas with observed ThAr lines
    ymax = np.percentile(arc_spectrum,displ_percent_max)
    idx_wave = ( ref_wavelength > min(wavelength_axis) ) *  ( ref_wavelength < max(wavelength_axis) )
    ymax_ref = np.percentile(ref_flux[idx_wave],displ_percent_max+0.5)
    ref_scalefactor = ymax / ymax_ref

    # compare lines with reference atlas
    plt.figure(figsize=(15,3))
    plt.plot(ref_wavelength,ref_flux*ref_scalefactor, lw=1, color='orange',   alpha=0.75, label='reference ThAr')
    plt.plot(wavelength_axis, arc_spectrum,           lw=1, color='black', alpha=0.5, label=f'STELES (dlambda={abs(wavelength_axis[-2]-wavelength_axis[-1]):.4f}A/pix)')
    txt_label = 'Reference ThAr lines'
    for wave in ThAr_peak_wavelengths:
        plt.axvline(wave,ls='--', color='red', lw=1, label=txt_label)
        txt_label=''
    plt.xlim(min(wavelength_axis),max(wavelength_axis))
    plt.ylim(0,ymax)
    plt.xlabel('Wavelength (A)')
    plt.legend()
    plt.suptitle(f"Reference ThAr spectrum and STELES (n={order:02d})")
    plt.tight_layout()
    plt.savefig(output_thar_ref, format=plot_extension.replace(".",""), dpi=300)
    plt.show()
    
    return wavelength_axis, rms_wavesol, fit_wavelength, ThAr_peak_pixels_refined

def plot_wavesolution_ThAr_atlas(wavelength_axis, arc_spectrum, ThAr_peak_wavelengths, ref_wavelength, ref_flux, 
                                 ref_scalefactor = 0.0250, figsize=(15,3), dpi=300, output=None):
    """ overlay the ThAr spectrum with the reference atlas to check wavelength solution"""

    plt.figure(figsize=(figsize))
    plt.plot(ref_wavelength,ref_flux*ref_scalefactor, lw=1, color='red',   alpha=0.9, label='reference ThAr')
    plt.plot(wavelength_axis, arc_spectrum,           lw=1, color='black', alpha=0.9, 
             label=f'STELES (dlambda={abs(wavelength_axis[1]-wavelength_axis[0]):.4f}A/pix)')
    txt_label = 'Reference ThAr lines'
    for wave in ThAr_peak_wavelengths:
        plt.axvline(wave,ls='--', color='green', label=txt_label)
        txt_label=''
    plt.xlim(min(wavelength_axis),max(wavelength_axis))
    plt.ylim(y_range)
    plt.xlabel('Wavelength (A)')
    plt.legend()
    plt.title(f"ThAr spectrum - order={order}")
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=dpi, format='png')
    plt.show()

def interpolate_wavelength_axis(flux, wavelength, header, dispersion=None, show_plot=False, figsize=(12,4)):
    """
    Create a linear wavelength grid from a non-linear wavelength array.
    """
    wave_min = wavelength.min()
    wave_max = wavelength.max()

    # Estimate the mean delta lambda from original
    deltas = np.diff(wavelength)
    
    if deltas[0] < 0:
        deltas = abs(deltas)
        
    if dispersion is None:
        delta_lambda = np.nanmin(deltas)
        print(f"Dispersion={delta_lambda:.4f} A/pix")
    else:
        delta_lambda = dispersion
        
    wave_linear = np.arange(wave_min, wave_max, delta_lambda)
    
    f_interp = interp1d(wavelength, flux, kind='cubic', fill_value='extrapolate')
    flux_interp = f_interp(wave_linear)
    
    header = header
    
    # Wavelength solution keywords
    crpix1 = 1  # Reference pixel (1-based in FITS)
    crval1 = wave_linear[0]  # Wavelength at reference pixel
    cdelt1 = wave_linear[1] - wave_linear[0]  # Dispersion (assumes linear)
    
    header.set("CRPIX1", value=crpix1, comment='Coordinate reference pixel' )
    header.set("CRVAL1", value=crval1, comment='Coordinate reference value' )
    header.set("CDELT1", value=cdelt1, comment='Coordinate interval' )
    header.set("CTYPE1", value='Wavelength', comment='Axis type' )
    header.set("CUNIT1", value='Angstrom', comment='Axis unit' )
    header.set("ST_WVDSP", value=delta_lambda, comment='Dispersion of the order (A/pix)')
    
    if show_plot:
        plt.figure(figsize=figsize)
        plt.plot(wavelength,  flux.data,   alpha=0.75, label='original')
        plt.plot(wave_linear, flux_interp, alpha=0.75, label='interpolated')
        plt.xlim(np.nanmin(wavelength),np.nanmax(wavelength))
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.ylabel(r'Flux (ADU)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return flux_interp, wave_linear, header


def read_ref_thar_line_list(file_path):
    """
    Reads an ASCII file containing two columns: wavelength and line name.

    Parameters
    ----------
    file_path : str
        Path to the ASCII file.

    Returns
    -------
    wavelengths : ndarray
        Array of central wavelengths (floats).

    line_names : list of str
        List of line names (strings).
    """
    wavelengths = []
    line_names = []

    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines or comment lines
            if line.strip() == "" or line.strip().startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) >= 2:
                wl = float(parts[0])
                name = parts[1]
                wavelengths.append(wl)
                line_names.append(name)

    return np.array(wavelengths), line_names
    
def read_ref_thar(infolder, fits_file):

    # Data from https://noirlab.edu/science/data-services/other/spectral-atlas
    with fits.open(infolder + fits_file + '.fits') as hdul:
        ref_flux   = hdul[0].data
        ref_header = hdul[0].header

    # read the line list
    l_wave, l_name = read_ref_thar_line_list(infolder + fits_file + '.txt')
   
    # Read wavelength calibration info from header (standard keywords)
    crval1 = ref_header.get('CRVAL1')  # Starting wavelength
    cdelt1 = ref_header.get('CDELT1')  # Wavelength increment per pixel
    crpix1 = ref_header.get('CRPIX1', 1)  # Reference pixel (default to 1 if missing)

    # Number of pixels
    npix = ref_flux.shape[0]

    # Compute wavelength array
    pixel_indices = np.arange(1, npix + 1)  # FITS convention: first pixel = 1
    ref_wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

    return ref_flux, ref_wavelength, l_wave, l_name


def read_ref_thar_simple(infolder, fits_file):

    # Data from https://noirlab.edu/science/data-services/other/spectral-atlas
    with fits.open(infolder + fits_file + '.fits') as hdul:
        ref_flux   = hdul[0].data
        ref_header = hdul[0].header
   
    # Read wavelength calibration info from header (standard keywords)
    crval1 = ref_header.get('CRVAL1')  # Starting wavelength
    cdelt1 = ref_header.get('CDELT1')  # Wavelength increment per pixel
    crpix1 = ref_header.get('CRPIX1', 1)  # Reference pixel (default to 1 if missing)

    # Number of pixels
    npix = ref_flux.shape[0]

    # Compute wavelength array
    pixel_indices = np.arange(1, npix + 1)  # FITS convention: first pixel = 1
    ref_wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

    return ref_flux, ref_wavelength

def plot_thar_reference_spectrum(infolder, thar_fits, outfolder, wave_min=3000, wave_max=9400, dwave=200, display_percentile=[0.1,99.5]):
    """
      infolder(str): directory containing the input data
      thar_fits (str): name of the fits file containing the ThAr spectrum
      outfolder (str): directory to store the plots
      wave_min (float, default=3000): minimum wavelength to plot (in Angstrom)
      wave_max (float, default=9400): maximum wavelength to plot (in Angstrom)
      dwave (float, default=200): wavelength interval for each plot (in Angstrom)
      display_percentile(float, default=[0.1,99.5]): percentile range to show the plot

    """

    # read the ThAr spectrum
    ref_flux, ref_wavelength, l_wave, l_name = read_ref_thar(infolder, thar_fits)

    # set number of plots
    n_plot = int( ( wave_max - wave_min ) / dwave )

    # loop into wavelength interval
    for i in np.arange(n_plot):

        # get wavelength interval
        wave_min_n = wave_min + dwave * i
        wave_max_n = wave_min + dwave * (i+1)

        # set x-axis range
        xlim = (wave_min_n-dwave*0.05,wave_max_n+dwave*0.05)

        # set y-axis range
        idx_wave = (ref_wavelength > wave_min_n) * (ref_wavelength < wave_max_n)
        ylim = np.percentile(ref_flux[idx_wave], display_percentile)

        # now plot
        plt.figure(figsize=(15,4))
        plt.plot(ref_wavelength, ref_flux, lw=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('Wavelength (A)')

        # add spectral lines
        plot_thar_lines(l_wave, l_name, wave_min_n, wave_max_n, linestyle='--', linecolor='black', linewidth=0.25, fontsize=6)

        plt.tight_layout()
        plt.savefig(outfolder + f'w_{wave_min_n}_{wave_max_n}.png', format='png', dpi=450)
        plt.show()
    
def plot_thar_lines(l_wave, l_name, wave_min, wave_max, linestyle='--', linecolor='black', linewidth=1, fontcolor='black', fontsize=7):
    
    idx = (l_wave > wave_min ) * (l_wave < wave_max)
        
    n_lines = np.sum(idx)
    print("n_lines=",n_lines)
    
    if n_lines > 0:
    
        p_wave = l_wave[idx]

        for i,w in enumerate(p_wave):
            #p_name = l_name[idx[i]] + f'[{w:.3f}]'
            p_name = f'[{w:.3f}]'
            plt.axvline(w, ls=linestyle, color=linecolor, lw=linewidth)
            plt.text(w, plt.ylim()[1]*1.02, p_name, rotation=90,
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
                color=fontcolor)
    else:
        print("No lines to plot.")
        
# fit spatial distortion along the slit
def fit_spatial_distortion(image, peak_positions_in_pixel, order, window=30, n_pix_step=1, ndeg=2, fitting_sigma_threshold=3.0, fitting_niter=3, 
                           saturation_threshold=50000, sigma_threshold=3., sigma_min= 1.0, sigma_step= 0.5,
                           single_fit=False, n_col=None, output=None):
    """
       Compute spatial distortion along the slit using selected spectral lines in the spectrum.
       image (numpy.ndarray): input 2D spectrum to fit
       peak_positions_in_pixel (numpy.array): list of spectral lines to fit
       window (int, default=30): pixel range to refine the line position
       n_pix_step (int, default=1): step size in pixels to compute the distortion along the slit.
       ndeg (int, default=2): polynomial degree to fit the distortion
       single_fit (boolean, default=False): use all line information to compute a single function (set to True to avoid outliers, specially when saturated lines are included)
       n_col (int, default=None): set number of columns for plotting
       output (str, default, None): name of the output file for the plot. 
    """
    
    # use the 'refined_peak_positions_in_pixel' value to trace the lines along the slit
    n_spat, n_spec = image.shape
    # set a new grid of spatial pixels, avoiding the borders
    spat_rows = np.arange(n_pix_step, n_spat-n_pix_step, n_pix_step)

    # get number of spatial rows
    n_spat_rows = len(spat_rows)
    
    # estimate the RMS from the image using the Median Absolute Deviation (MAD)
    image_rms = np.median(np.absolute(image - np.median(image)))
    if image_rms == 0:
        image_rms = np.nanstd(image)
    
    # now get the refined peak positions for every line and across the rows in 'spat_rows'
    peak_refined_rows = []
    peak_fluxes       = []
    for row in spat_rows:
        # set center and window on y-axis to extract the spectrum
        y_center = row
        row_range = (y_center - 1, y_center + 1 )
        arc_1d = extract_spectrum1d(image, row_range=row_range)
        # refine x_peak positions for each row
        peak_refined_row, peak_flux = refine_peak_positions(arc_1d, peak_positions_in_pixel, window=window, plot_fits=False)
        
        #print(type(peak_positions_in_pixel), type(peak_refined_row))
        #print(row,(peak_positions_in_pixel), (peak_refined_row))
        # update peak information from previous fit - only if it returns a finite value (avoid adding NaN)
        idx_finite = np.isfinite(peak_refined_row)
        peak_positions_in_pixel[idx_finite] = peak_refined_row[idx_finite]
        
        # store the data into a list
        peak_refined_rows.append(peak_refined_row)
        peak_fluxes.append(peak_flux)
        
    # convert list to numpy array
    peak_refined_rows= np.array(peak_refined_rows, dtype='float32')
    peak_fluxes      = np.array(peak_fluxes, dtype='float32')

    # now exclude saturated lines
    n_lines = peak_fluxes.size
    idx_saturated = ( peak_fluxes >= saturation_threshold)
    print(f' Excluding {np.sum(idx_saturated)} saturated measurements (>{saturation_threshold:.1f})')
    
    # --- Adaptive sigma threshold ---
    sigma_current = sigma_threshold

    min_lines_required = max(5, ndeg + 2)

    while True:

        idx_weak = (peak_fluxes <= sigma_current * image_rms)

        # GOOD lines = not saturated AND not weak
        idx_good = (~idx_saturated) & (~idx_weak)

        n_good = np.sum(idx_good)

        if n_good >= min_lines_required:
            break

        sigma_current -= sigma_step

        if sigma_current < sigma_min:
            self.log.warning(
                f"Order {order}: reached minimum sigma threshold ({sigma_min}) "
                f"with only {n_good}/{n_lines} usable lines"
            )
            break
    
    idx_weak = ( peak_fluxes <= sigma_current * image_rms)
    
    idx_out = ( ~( (~idx_saturated) & (~idx_weak) ) )
    
    print(f' Excluding {np.sum(idx_weak)} measurements below {sigma_current:.2f}-sigma (<{sigma_current*image_rms:.2f})')
    print(f'Using {np.sum(~idx_out)}/{n_lines} measurements.')
                        
    # flag saturated/weak measurements to exclude the fitting
    peak_refined_rows_sat  = peak_refined_rows.copy()
    peak_refined_rows_weak = peak_refined_rows.copy()
    peak_refined_rows_sat[idx_saturated] = np.nan
    peak_refined_rows_weak[idx_weak]     = np.nan
    
    # get number of spectral lines to fit
    n_lines = len(peak_positions_in_pixel)
    
    n_plots = n_lines+1
    if n_col is None:
        n_col = n_plots if n_plots <= 4 else 4
    n_row = int(n_plots / n_col)
    if n_col * n_row < n_plots:
        n_row += 1 
    
    # now plot
    plt.figure(figsize=(3*n_col,3*n_row))
    
    
    line_fit   = []
    line_valid = []
    for n in np.arange(n_lines):
        # set (x,y) for each line - we want f(y)=x!
        y = peak_refined_rows[:,n]
        x = spat_rows
        # read saturated and weak measurements
        ysat  = peak_refined_rows_sat[:,n]
        yweak = peak_refined_rows_weak[:,n]
        # if y has "NaN" values, the line is probably saturated, so skip
        #if ( np.sum(np.isnan(ysat)) == 0 ) and ( np.sum(np.isnan(yweak)) == 0 ):
        valid = (~np.isnan(ysat)) & (~np.isnan(yweak))
        
        if np.sum(valid) > max(5 ,ndeg+2):
            x_valid = x[valid]
            y_valid = y[valid]
            
            # start the fitter
            linfitter = LinearLSQFitter()
            spatial_model = Polynomial1D(degree=ndeg)
            #fit_spat = linfitter(model=spatial_model, x=x, y=y)
            fit_spat = linfitter(model=spatial_model, x=x_valid, y=y_valid)
            # store the polynomial fits
            line_fit.append(fit_spat.parameters)
            # flag valid line
            line_valid.append(True)
            
            # get yfit
            yfit = fit_spat(x) #* u.nm
            #plot results
            plt.subplot(n_row,n_col,n+1)
            #plt.plot(yfit,x,color='red',label='Model')
            #plt.plot(y,x,'+',label='Data')
            plt.plot(fit_spat(x),x,color='red',label='Model')
            plt.plot(y_valid,x_valid,'+',label='Data')
            plt.xlabel('Dispersion (pixel)')
            plt.title(f'Line at x={peak_positions_in_pixel[n]:.1f}')
            if n == 0:
                plt.ylabel('Along slit (pixel)')
                
        else:
            plt.subplot(n_row,n_col,n+1)
            plt.plot(y,x,'+',label='Data (no fit)')
            plt.xlabel('Dispersion (pixel)')
            if np.sum(np.isnan(ysat)) > 0:
                plt.title(f'Line at x={peak_positions_in_pixel[n]:.1f} (saturated)')
            elif np.sum(np.isnan(yweak)) > 0:
                plt.title(f'Line at x={peak_positions_in_pixel[n]:.1f} (weak)')
            if n == 0:
                plt.ylabel('Along slit (pixel)')
            # fla valid line
            line_valid.append(False)
    
    # now, get the line displacement along the slit with respect to the center
    delta_peak_refined = peak_refined_rows
    for m in np.arange(n_lines):
        
        # if line was measured:
        if line_valid[m]:
            ref_val = np.nanmedian(peak_refined_rows[:,m])
            delta_peak_refined[:,m] = peak_refined_rows[:,m] - ref_val
        else:
            delta_peak_refined[:,m] = np.nan
            
    # reorganize the arrays
    y = delta_peak_refined.flatten()
    x = spat_rows[:, np.newaxis] * np.ones((spat_rows.shape[0], n_lines))
    x = x.flatten()
    
    # remove any non-finite value
    idx_finite = np.isfinite(y)
    y = y[idx_finite]
    x = x[idx_finite]
    
    # start the fitter
    linfitter = LinearLSQFitter()
    spatial_model = Polynomial1D(degree=ndeg)
    
    if len(x) < (ndeg + 3):
        raise RuntimeError(f"Not enough valid points to fit spatial distortion for order n={order}")
    
    # sigma-clipping based outlier remover
    fitter = fitting.FittingWithOutlierRemoval(linfitter, sigma_clip, sigma=fitting_sigma_threshold, niter=fitting_niter)

    fit_spat_all, mask_outliers = fitter(spatial_model, x, y)
    
    # get yfit
    yfit = fit_spat_all(x)
    #plot results
    plt.subplot(n_row,n_col,n+2)
        

    if np.sum(mask_outliers) == 0:    
        plt.plot(y,x,'+',label='Data')
    else:
        plt.plot(y[~mask_outliers], x[~mask_outliers], '+',              label='Data'    )
        plt.plot(y[mask_outliers],  x[mask_outliers],  'x', color='red', label='Outliers')
        
    plt.plot(yfit, x, color='black', label='Model')
        
    plt.xlabel('Dispersion (pixel)')
    plt.title(f'All lines')

    plt.legend()
    plt.suptitle(f'Spatial distortion fitting along the slit (order={order})', y=1.01)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, dpi=300)
    plt.show()
    
    if single_fit:
        polyfits_x_along_y = []
        for row in spat_rows:
            polyfits_x_along_y.append(fit_spat_all)
        polyfits_x_along_y = np.array(polyfits_x_along_y)
        polyfits_x_along_y = np.reshape(polyfits_x_along_y, (n_spat_rows, -1))
    
    else:
        # now, store the polynomial fits for each line
        polyfits_x_along_y = []
        for row in spat_rows:
            # set (x,y) for each line
            y = row
            # for each line, construct the polynomial mode to evaluate x_peak as a function of the y-axis 
            for coeffs in line_fit:
                # Reconstruct the model
                x_peak_model = models.Polynomial1D(degree=len(coeffs)-1)
                x_peak_model.parameters = coeffs
                polyfits_x_along_y.append(x_peak_model)

        polyfits_x_along_y = np.array(polyfits_x_along_y)
        polyfits_x_along_y = np.reshape(polyfits_x_along_y, (n_spat_rows, -1))

    return spat_rows, polyfits_x_along_y


def read_polyfits_csv(input_csv):
    """
    Read CSV file containing:
    - First line: reference pixel (float)
    - Remaining lines: polynomial coefficients per row

    Returns:
        pixels   : float
        polyfits : array of Polynomial1D objects (shape = n_rows)
    """

    with open(input_csv, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise RuntimeError(f"Invalid CSV format: {input_csv}")

    # --- First line = reference pixel ---
    try:
        pixels = float(lines[0])
    except ValueError:
        raise RuntimeError(f"Could not parse reference pixel in {input_csv}")

    # --- Remaining lines = polynomial coefficients ---
    polyfits = []

    for line in lines[1:]:
        # Remove brackets
        coeffs_txt = line.replace("[", "").replace("]", "")

        # Convert to float list
        try:
            coeffs = [float(x) for x in coeffs_txt.split(",")]
        except ValueError:
            raise RuntimeError(f"Invalid coefficient line: {line}")

        degree = len(coeffs) - 1

        poly = Polynomial1D(
            degree=degree,
            **{f"c{k}": v for k, v in enumerate(coeffs)}
        )

        polyfits.append(poly)

    polyfits = np.array(polyfits, dtype=object)

    return pixels, polyfits


def correct_spatial_distortion(image, y_positions, x_y_fits, reference_peak_positions, order, y_ref=None, 
                               show_plot=True, figsize=(12,12), wave_pixel_reference = [3,11,19], 
                               max_displ_percent = 99, plot_x_limit=(900,1500), output=None):
    """
    Rectify 2D image using a set of x(y) distortion curves at given x positions.
    
    Parameters:
        image : 2D numpy array ()
            The distorted input image [ny, nx].
        y_positions : array-like
            The x positions at which the x(y) functions were measured.
        x_y_fits : list of Polynomial1D models
            The distortion curves (x as a function of y) for each x_position.
        y_ref : float
            Reference y value to align all x(y) to (default: center of image).
        wave_pixel_reference : float (list
            Reference line positions (in pixel) to display over the distorted-corrected images.
    
    Returns:
        rectified_ccddata : 2D numpy array (CCDData)
            The distortion-corrected image.
    """
    ny, nx = image.shape
    if y_ref is None:
        y_ref = ny // 2
    
    # this version of the code only uses one polynomial fit for the entire order, no matter how many lines are stored into 'x_y_fits'.
    # TODO: add all the measurements from 'x_y_fits' to the fit
    if x_y_fits.ndim > 1:
        print("More than one fit to consider...")
        # first, get the center of the dispersion axis
        center_pixel = int(nx // 2)
        # now, find the line closest to the center of the dispersion axis
        idx_center_pixel = np.abs(reference_peak_positions - center_pixel).argmin()
        # set the x_y_fits to the coefficients corresponding to the line closest to the center of the dispersion axis
        x_y_fits = x_y_fits[:,idx_center_pixel]
    
    y_vals = np.arange(ny)
    x_vals = np.arange(nx)

    # Step 1: Evaluate x(y) for each given x_position
    distortion_curves = np.zeros((len(y_positions), ny))  # [n_sample_y, ny]
    
    for i, fit in enumerate(x_y_fits):
        distortion_curves[i, :] = fit(y_vals)

    # Step 2: Interpolate across all x positions to make a full 2D distortion map
    x_shift_map = np.zeros((ny, nx))
    for y in range(ny):
        interp = interp1d(y_positions, distortion_curves[:, y], kind='cubic', fill_value='extrapolate')
        distorted_x = interp(x_vals)
        # Reference: what x value should this trace have at y_ref?
        ref_interp = interp1d(y_positions, [fit(y_ref) for fit in x_y_fits], kind='cubic', fill_value='extrapolate')
        ref_x = ref_interp(x_vals)
        x_shift_map[y, :] = ref_x - distorted_x

    # Step 3: Apply the shift to resample the image
    y_coords, x_coords = np.indices(image.shape)
    corrected_x = x_coords - x_shift_map
    
    rectified_image = map_coordinates(image, [y_coords, corrected_x], order=1, mode='nearest')

    if show_plot:
        xmin=0
        xmax=image.shape[1]

        #fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"hspace": 0.3}, sharex=True, layout='constrained')
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
        #plt.rcParams['figure.constrained_layout.use'] = True

        #print('input_image:',np.nanmin(image),np.nanmax(image),max_displ)
        max_displ = np.percentile(image, max_displ_percent)
        
        ##################################        
        #plt.subplot(4,1,1)
        im_ref = axes[0].imshow(image, aspect='auto', origin='lower', vmin=0, vmax=max_displ)
        
        # draw vertical lines showing the spectral reference positions
        for x in reference_peak_positions:
            axes[0].axvline(x, ls='--', color='black')
        
        # draw horizontal lines showing the spatial reference positions
        for ypp in wave_pixel_reference:
            axes[0].axhline(ypp, ls='--', label=f"y={ypp} (original)")
            
        #plt.xlim(xmin,xmax)
        axes[0].set_xlim(plot_x_limit)
        axes[0].set_title(f"Original 2D spectrum (order={order})")
        
        ##################################        
        #plt.subplot(4,1,2)
        axes[1].imshow(rectified_image, aspect='auto', origin='lower', vmin=0, vmax=max_displ)
        for x in reference_peak_positions:
            axes[1].axvline(x, ls='--', color='black')

            
        for ypp in wave_pixel_reference:
            axes[1].axhline(ypp, ls='--', label=f"y={ypp} (original)")            
            
        #plt.xlim(xmin,xmax)
        axes[1].set_xlim(plot_x_limit)
        axes[1].set_title("Distortion corrected 2D spectrum")

        # Add one colorbar for the first two
        cbar = fig.colorbar(im_ref, ax=[axes[0], axes[1]], location='right', shrink=1.0, fraction=0.05, pad=0.005, aspect=30, label='Flux (ADU)')
        
        ##################################
        #axes[2].subplot(4,1,3)
        y = image
        y_avg = np.nanmean(image,axis=0)
        
        axes[2].plot(np.arange(len(y_avg)),y_avg, color='black', lw=1.5, label=f"Average (original)")
        for ypp in wave_pixel_reference:
            yp = y[ypp,:]
            axes[2].plot(np.arange(len(yp)),yp, lw=1, alpha=0.8, label=f"y={ypp} (original)")
        for x in reference_peak_positions:
            axes[2].axvline(x, ls='--', lw=0.75, color='black')
        axes[2].set_xlim(plot_x_limit)
        axes[2].set_ylim(0,max_displ)
        axes[2].legend()
        axes[2].set_title("Original 1D spectrum")

        ##################################
        #plt.subplot(4,1,4)
        y = rectified_image
        y_avg = np.nanmean(rectified_image,axis=0)
        axes[3].plot(np.arange(len(y_avg)),y_avg, color='black', lw=1.5, label=f"Average (corrected)")
        for ypp in wave_pixel_reference:
            yp = y[ypp,:]
            axes[3].plot(np.arange(len(yp)),yp, lw=1, alpha=0.8, label=f"y={ypp} (corrected)")
        for x in reference_peak_positions:
            axes[3].axvline(x, ls='--', lw=0.75, color='black')
        axes[3].set_xlim(plot_x_limit)
        axes[3].set_ylim(0,max_displ)
        axes[3].legend()
        axes[3].set_title("Distortion corrected 1D spectrum")
        
        #plt.tight_layout()
        if output is not None:
            plt.savefig(output, dpi=300)
        plt.show()

    return rectified_image


def polynomialfit(x, y, nmin=1, nmax=3, debug=False):
    """
    Select the minimum chi squared polynomial fit between nmin and nmax

    :param x: x axis array
    :param y: y axis array
    :param nmin: minimum order to fit
    :param nmax: maximum order to fit

    Note could improve this to weight it by pixel uncertainties
    """
    chis, ps = [], []
    nrange = range(nmin, nmax+1)
    
    for n in nrange:
        p = np.polyfit(x, y, n)
        yfit = np.polyval(p, x)
        
        # get the reduced chi2
        chi_squared = np.sum((y-yfit)**2)
        
        ps.append(p)
        chis.append(chi_squared)
        if debug:
            print(n,chi_squared)
    
    argmin = np.argmin(chis)
    return ps[argmin], nrange[argmin]

def eval_polynomialfit(x, y, nmin=1, nmax=3, chi2_threshold=0.1, debug=False):
    """
    Select the minimum chi squared polynomial fit between nmin and nmax

    :param x: x axis array
    :param y: y axis array
    :param nmin: minimum order to fit
    :param nmax: maximum order to fit

    Note could improve this to weight it by pixel uncertainties
    """
    chis, ps = [], []
    nrange = range(nmin, nmax+1)
    
    for n in nrange:
        p = np.polyfit(x, y, n)
        yfit = np.polyval(p, x)
        
        # evaluate the degrees of freedom of the plot
        dof = len(x) - n - 1
        # get the reduced chi2
        chi_squared = np.sum((y-yfit)**2) / dof
        
        ps.append(p)
        chis.append(chi_squared)
        if debug:
            print(n,chi_squared)
    
    best_degree = None
    best_chi2 = np.inf
    for i,n in enumerate(nrange):
        # Check for the simplest model within the threshold
        
        #if best_degree is not None:
        #    print(i,chis[i], best_chi2, chis[i] < best_chi2, best_chi2 - chis[i], best_chi2 - chis[i] < chi2_threshold)
        
        if chis[i] < best_chi2:
            if best_degree is not None and best_chi2 - chis[i] < chi2_threshold:
                #best_degree = min(best_degree, n)  # Choose simpler model
                best_degree = min(best_degree, i)  # Choose simpler model
            else:
                #best_degree = n
                best_degree = i
                best_chi2 = chis[i]
        #print(i,best_degree)
        
    argmin = best_degree #np.where(nrange == best_degree)[0]
    #print('best_degree=',nrange[argmin])
    return ps[argmin], nrange[argmin]


def fit_quartz(quartz_1d_extracted, nmin=1, nmax=30, sigma_clip_thresh=None, n_cols = 4, output_blaze=None):

    warnings.filterwarnings('ignore', message="overflow encountered in", category=RuntimeWarning)
    # if using older versions of numpy
    try:
        warnings.simplefilter('ignore', np.RankWarning)
    except:
        warnings.simplefilter('ignore', np.exceptions.RankWarning)
    
    # works if you open a ".fits" file
    try:
        with fits.open(quartz_1d_extracted) as hdul:
            quartz1d_data   = hdul[0].data
    # and if you open a numpy.array file
    except:
        quartz1d_data = quartz_1d_extracted
    
    # if multiextension, use first one
    if quartz1d_data.ndim > 2:
        quartz1d_data = quartz1d_data[0,:,:]

    xarr=np.arange(quartz1d_data.shape[1])
    fit = quartz1d_data.copy()
    
    plt.figure(figsize=(18,18))
    
    n_rows = np.round(quartz1d_data.shape[0]/n_cols)
    if n_cols * n_rows < quartz1d_data.shape[0]:
        n_rows += 1
    n_rows = int(n_rows)
    
    for n in range(quartz1d_data.shape[0]):
        yarr = quartz1d_data[n,:]
        #yarr -= np.percentile(yarr,1)
        yarr /= np.percentile(yarr,99)
        idx_finite = np.isfinite(yarr) & ( yarr != 0 ) # second term was added for the first orders that do not completely fill the x-axis
        
        if sigma_clip_thresh is not None:
            # mask outliers at >5 sigma (tune as needed)
            clipped = sigma_clip(yarr, sigma=sigma_clip_thresh, maxiters=3)
            mask = clipped.mask
            idx_fit = ~mask * idx_finite
            if np.sum(idx_fit) == 0:
                idx_fit = idx_finite
        else:
            idx_fit = idx_finite
        
        if np.sum(idx_fit) == 0:
            idx_fit = ~idx_fit
        
        ypfit, n_fit = polynomialfit(xarr[idx_fit], yarr[idx_fit], nmin=nmin, nmax=nmax, debug=False)
        yfit = np.polyval(ypfit, xarr)
        # avoid negative counts
        yfit[yfit <= 0] = 0.001
        # set to "NaN" the blaze at the corners (but not the masked pixels)
        yfit[~idx_finite] = np.nan
        yfit[np.isnan(yfit)] = 1.0
        
        # save the results of the fit as a new array
        fit[n,:] = yfit
        
        # plot order
        plt.subplot(n_rows,n_cols,n+1)
        plt.plot(xarr,yarr, color='black', alpha=0.5)
        plt.plot(xarr, yfit, color='red')
        plt.ylim(0,1.1)
        plt.xlim(0,len(xarr))
        plt.title('order={}, n_fit={}'.format(n+1,n_fit))
        
    plt.tight_layout()
    if output_blaze is not None:
        plt.savefig(output_blaze, dpi=300)
    plt.show()
    
    return fit
     