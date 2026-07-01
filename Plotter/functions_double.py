import logging
import numpy as np
import matplotlib as mpl
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq
from scipy.optimize import curve_fit
import logging
import h5py
import os
import warnings
from sklearn.metrics import r2_score

from scipy.optimize import differential_evolution

logger = logging.getLogger()


def detect_peaks(data, find_dips=True, **kwargs):
    # normalize data:
    # (1) substract offset to zero
    min_data = np.min(data)
    data = data - min_data
    # (2) normalize on 1
    max_data = np.max(data)
    data = data / max_data

    if find_dips:
        data = -data

    peaks, _ = find_peaks(data, **kwargs)

    return peaks

def correct_finder(data, peak_list, WL):
    
    index = np.where(data > WL)[0][0]
    peak_list_upd = np.sort(np.concatenate((peak_list,[index])))
    
    return peak_list_upd

def remove_peaks(data, peak_list, WL):
    
    index = np.where(data > WL)[0][0]
    
    for k in range(len(peak_list)-1):
        if index < peak_list[k]+100:
            if index > peak_list[k]-100:
                peak_list=np.delete(peak_list,k)
    
    return peak_list

def findFitRange(peaks, range0, a_c, a_r):
    
    WL_range = []
    count=0
    
    for r in range(len(peaks)):
        WL_range.append([peaks[r]-range0,peaks[r]+range0])
    
    
    if len(a_c)>0:
        for s in a_c:

            for t in range(len(peaks)):
                if t==s:
                    
                    WL_range[t]=a_r[count]
                    count=count+1
                    break
    
    return WL_range
    
    
def open_h5(path):
    with h5py.File(path, "r") as fh:
        folder = 'datasets'

        meas_type = fh[folder]


        #print("Available datasets:", list(meas_type.keys()))


        #RR_raw = np.array(meas_type['wavelength_scan.S.mean'])
        RR_raw = np.array(meas_type['wavelength_scan.RR.mean'])
        WL = np.array(meas_type['wavelength_scan.wl'])
        RR_raw = RR_raw[:-1]
        WL = WL[:-1]
        sample = str(np.array(meas_type['wavelength_scan.structure_string']))

    return WL, RR_raw, sample

def write_analysis_h5(name, set1, set2, set3, set3b, set4, set4b, set5, set6):
    
    if not os.path.isfile(name):    
        with h5py.File(name, 'w') as newfile:
            newfile.create_dataset('gap', data=np.array([set1]))
            newfile.create_dataset('L', data=np.array([set2]))
            newfile.create_dataset('Qs', data=np.stack((np.ones(len(set3))*np.nan,set3)))
            newfile.create_dataset('sigma_Qs', data=np.stack((np.ones(len(set3b))*np.nan,set3b)))
            newfile.create_dataset('Fs', data=np.stack((np.ones(len(set4))*np.nan,set4)))
            newfile.create_dataset('sigma_Fs', data=np.stack((np.ones(len(set4b))*np.nan,set4b)))
            newfile.create_dataset('ERs', data=np.stack((np.ones(len(set5))*np.nan,set5)))
            newfile.create_dataset('data_filter', data=set6)
            newfile.close()
            out = 'written'
        
    else:
        with h5py.File(name,'r+') as ds:
            
            gap = ds['gap']
            L = ds['L']
            data_filter = ds['data_filter']
            
            gap=np.concatenate([gap,np.array([set1])])
            L=np.concatenate([L,np.array([set2])])
            data_filter=np.vstack((data_filter,set6))
            
            
            cont = [ds['Qs'],ds['sigma_Qs'],ds['Fs'],ds['sigma_Fs'],ds['ERs']]
            cont2 = [set3,set3b,set4,set4b,set5]
            
            for t in range(len(cont)):
                diff = cont[t].shape[1]-len(cont2[t])
                
                if diff<0:
                    stitch = np.ones((cont[t].shape[0],-diff))*np.nan
                    cont_new = np.hstack((cont[t],stitch))
                    cont_f = np.vstack((cont_new,cont2[t]))
                    cont[t] = cont_f
                    
                else:
                    stitch = np.ones(diff)*np.nan
                    cont_new=np.concatenate((cont2[t],stitch))
                    cont_f = np.vstack((cont[t],cont_new))
                    cont[t] = cont_f
            
            
            del ds['gap']
            del ds['L']
            del ds['Qs']
            del ds['sigma_Qs']
            del ds['Fs']
            del ds['sigma_Fs']
            del ds['ERs']
            del ds['data_filter']
            
            ds.create_dataset('gap', data=gap)
            ds.create_dataset('L', data=L)
            ds.create_dataset('Qs', data=cont[0])
            ds.create_dataset('sigma_Qs', data=cont[1])
            ds.create_dataset('Fs', data=cont[2])
            ds.create_dataset('sigma_Fs', data=cont[3])
            ds.create_dataset('ERs', data=cont[4])
            ds.create_dataset('data_filter', data=data_filter)
            ds.close()            
            out='overwritten'
            
    return out
            
            
def normalization(path,WL_data,RR_raw_data):

    with h5py.File(path, "r") as norm_fh:
        
        norm_folder = 'datasets'

        norm_meas_type = norm_fh[ norm_folder ]
        RR_norm = np.array( norm_meas_type['wavelength_scan.S.mean'])
        RR_out=np.zeros(len(RR_raw_data))
        WL_norm =np.array( norm_meas_type['wavelength_scan.wl'])
        
        for i in range(len(WL_data)):
            if(WL_data[i]<1599.8):
                index=np.where(WL_norm>=WL_data[i])[0][0]
                #print(index)
                #print(str(WL_data[i])+', '+str(WL_norm[index]))
                RR_out[i]=abs(RR_raw_data[i])/abs(RR_norm[index])
                #print(str(RR_raw_data[i])+', '+str(RR_norm[index])+', '+str(RR_out[i]))
            else:
                RR_out[i]=abs(RR_raw_data[i])/abs(RR_norm[-1])
    return RR_out

def UnpackAnalysisH5(path):
    with h5py.File(path, "r") as ds:
        
        gap = list(ds['gap'])
        L = list(ds['L'])
        Qs = list(ds['Qs'])[1:]
        sigmaQs = list(ds['sigma_Qs'])[1:]
        Fs = list(ds['Fs'])[1:]
        sigmaFs = list(ds['sigma_Fs'])[1:]
        ERs = list(ds['ERs'])[1:]
        data_filter = list(ds['data_filter'])
        ds.close()

    return gap, L, Qs, sigmaQs, Fs, sigmaFs, ERs, data_filter


# %%Fitting procedure

def Lorentzian_sq(x, amp1, cen1, wid1, b, c):
    gamma = wid1 / 2
    return ((amp1 * gamma / ((x - cen1) ** 2 + gamma ** 2)) + b + c*x)**2



def double_Lorentzian_sq(x, amp1, cen1, wid1, amp2, cen2, wid2, b, c):
    gamma1 = wid1 / 2
    gamma2 = wid2 / 2
    return ((amp1 * gamma1 / ((x - cen1) ** 2 + gamma1 ** 2)) + (amp2 * gamma2 / ((x - cen2) ** 2 + gamma2 ** 2)) + b +c*x)**2


def fitting_Lor(data_x, data_y, guess_):
    bounds = [-np.inf, 0, 0, -np.inf, -np.inf], [0, np.inf, np.inf, np.inf, np.inf]
    par, cov = curve_fit(Lorentzian_sq, data_x, data_y, guess_, maxfev=2000, bounds=bounds)
    QL = par[1] / par[2]
    sigmaQL = np.sqrt(cov[1][1] * (1 / par[2]) ** 2 + cov[2][2] * (par[1] / (par[2]) ** 2) ** 2)
    y_pred = Lorentzian_sq(data_x, *par)
    r2 = r2_score(data_y, y_pred)

    return par, cov, QL, sigmaQL, r2


def fitting_doubleLor(data_x, data_y, guess_):
    # amp1, cen1, wid1, amp2, cen2, wid2, b, c
    bounds = [-np.inf, data_x[int(len(data_x)*0.2)], 0, -np.inf, data_x[0], 0, -np.inf, -np.inf], [0, data_x[int(len(data_x)*0.8)], np.inf, 0, data_x[-1], np.inf, np.inf, np.inf]
    par, cov = curve_fit(double_Lorentzian_sq, data_x, data_y, guess_, maxfev=4000, bounds=bounds)
    y_pred = double_Lorentzian_sq(data_x, *par)
    r2 = r2_score(data_y, y_pred)

    QL1_ = par[1] / par[2]
    sigmaQL1_ = np.sqrt(cov[1][1] * (1 / par[2]) ** 2 + cov[2][2] * (par[1] / (par[2]) ** 2) ** 2)

    QL2_ = par[4] / par[5]
    sigmaQL2_ = np.sqrt(cov[4][4] * (1 / par[5]) ** 2 + cov[5][5] * (par[4] / (par[5]) ** 2) ** 2)
    QL1_ = np.abs(QL1_)
    QL2_ = np.abs(QL2_)
    return par, cov, QL1_, sigmaQL1_, QL2_, sigmaQL2_, r2

def param(data_x, data_y):
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        return np.sum((data_y - Lorentzian_sq(data_x, *parameterTuple)) ** 2)

    def generate_Initial_Parameters():
        # min and max used for bounds
        maxX = max(data_x)
        minX = min(data_x)
        maxY = max(data_y)
        minY = min(data_y)
# amp1, cen1, wid1, amp2, cen2, wid2, b, c

        parameterBounds = []
        parameterBounds.append([-0.1, 0])  # parameter bounds for amp1
        parameterBounds.append([data_x[int(len(data_x)*0.2)], data_x[int(len(data_x)*0.8)]])  # parameter bounds for cen1
        parameterBounds.append([0.0, maxY / 2.0])  # parameter bounds for wid1
        parameterBounds.append([0, 0.1])  # parameter bounds for b
        parameterBounds.append([-0.1, 0.1])  # parameter bounds for c

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    initialParameters = generate_Initial_Parameters()
    return initialParameters

def param_double(data_x, data_y):
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        return np.sum((data_y - double_Lorentzian_sq(data_x, *parameterTuple)) ** 2)

    def generate_Initial_Parameters():
        # min and max used for bounds
        maxX = max(data_x)
        minX = min(data_x)
        maxY = max(data_y)
        minY = min(data_y)
# amp1, cen1, wid1, amp2, cen2, wid2, b, c

        parameterBounds = []
        parameterBounds.append([-0.1, 0])  # parameter bounds for amp1
        parameterBounds.append([data_x[int(len(data_x)*0.2)], data_x[int(len(data_x)*0.8)]])  # parameter bounds for cen1
        parameterBounds.append([0.0, maxY / 2.0])  # parameter bounds for wid1
        parameterBounds.append([-0.1, 0])  # parameter bounds for amp2
        parameterBounds.append([data_x[int(len(data_x)*0.2)], data_x[int(len(data_x)*0.8)]])  # parameter bounds for cen2
        parameterBounds.append([0.0, maxY / 2.0])  # parameter bounds for wid2
        parameterBounds.append([0, 0.1])  # parameter bounds for b
        parameterBounds.append([-0.1, 0.1])  # parameter bounds for c

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    initialParameters = generate_Initial_Parameters()
    return initialParameters

def plot_Fourier_ops(fourier_WL, fourier_RR, WL, RR, RR_filt, RR_filtd, low_f, high_f, threshold, axes_sizes, tick_sizes):
    
    #Fourier transform plot   
    fig2= plt.figure(figsize=(6, 4))
    ax2=fig2.add_subplot()

    ax2.plot(fourier_WL[0:1000],fourier_RR[0:1000])

    ax2.set_xlabel('Time index (x$10^9$) [s]', fontsize=axes_sizes)
    ax2.set_ylabel('Amplitude [-]', fontsize=axes_sizes)
    ax2.tick_params(axis='both', labelsize=tick_sizes)

    fig2.tight_layout()

    left, bottom, width, height = [0.55, 0.4, 0.35, 0.35]
    ax3 = fig2.add_axes([left, bottom, width, height])

    ax3.plot(fourier_WL[np.where(fourier_WL>fourier_WL[1000])],fourier_RR[np.where(fourier_WL>fourier_WL[1000])], color='green')
    ax3.axhspan(-threshold, threshold, color='grey', alpha=0.5)
    
    for i in range(len(low_f)):
        ax2.axvspan(low_f[i], high_f[i], color='grey', alpha=0.5)
        
        
    #fourier transforms
    fig4= plt.figure(figsize=(6, 4))
    ax4=fig4.add_subplot()
    ax4.plot(WL,RR, label='Raw data')
    ax4.plot(WL,RR_filtd, alpha=0.7, label='Filtered out')
    ax4.set_xlabel('Wavelength [nm]', fontsize=axes_sizes)
    ax4.set_ylabel('Relative Reflection [-]', fontsize=axes_sizes)
    ax4.tick_params(axis='both', labelsize=tick_sizes)
    ax4.legend(fontsize=axes_sizes)
    fig4.tight_layout()
    
    #fourier transforms
    fig5= plt.figure(figsize=(6, 4))
    ax5=fig5.add_subplot()
    ax5.plot(WL,RR_filt)
    ax5.set_xlabel('Wavelength [nm]', fontsize=axes_sizes)
    ax5.set_ylabel('Relative Reflection [-]', fontsize=axes_sizes)
    ax5.tick_params(axis='both', labelsize=tick_sizes)
    fig5.tight_layout()
    
    


def fourier_filtering(WL, RR, N_THRESH1, N_THRESH2, baseline):
    # Choose filtering threshold

    t = fftfreq(RR.size, d=(WL[1] - WL[0]))
    RR_fourier_t = rfft(RR)

    
    cut_RR_fourier_t = RR_fourier_t.copy()
    filt_RR_fourier_t = np.zeros(len(RR_fourier_t))
    
    for i in range(len(N_THRESH1)): #remove intervals
        
        idx1 = np.where(t>=N_THRESH1[i])[0][0]
        idx2 = np.where(t>=N_THRESH2[i])[0][0]
        
        cut_RR_fourier_t[idx1:idx2] = 0

    cut_RR_fourier_t[np.where(abs(cut_RR_fourier_t)<baseline)] = 0 #remove baseline coefficients
    
    filt_RR_fourier_t[np.where(cut_RR_fourier_t==0)]=RR_fourier_t[np.where(cut_RR_fourier_t==0)]
    
    RR_above = irfft(cut_RR_fourier_t)
    RR_below = irfft(filt_RR_fourier_t)

    return t, RR_fourier_t, RR_above, RR_below


def Lopriore_plot(f_guess, WL, RR, sample_nr, save_path, counter, f_range):
    sample_nr = sample_nr[2:-1]
    #f_range = [f_guess - dif, f_guess + dif]
    save_plot = 1
    save_fit = 1
    p_fit = 1
    
    # %%Plotting params
    axes_sizes = 12
    tick_sizes = 10
    


    try:
        f_b_idx = np.where(WL >= f_range[0])[0][0]
        f_t_idx = np.where(WL >= f_range[1])[0][0]
        WL_range_raw = WL[f_b_idx:f_t_idx]
        RR_range_raw = RR[f_b_idx:f_t_idx]
    except IndexError:
        print("IndexError")
        return None
    
    #findFitRange(WL_range_raw, RR_range_raw, 3)
    WL_range = WL_range_raw
    RR_range = RR_range_raw
    
    # save data in this directory
    directory = save_path + sample_nr
    if not os.path.exists(directory):
        os.makedirs(directory)
    File_to_save = save_path + sample_nr + f'/'

    #Simple Lorentzian fit
    guess = [-0.0001, f_guess, 0.01, 0,0]
    #Double Lorentzian fit
    guess_double = [-0.0001, f_guess-0.01, 0.005, -0.0001, f_guess, 0.005, 0,0]

    # %%Fitting procedure
    if p_fit:
        try:
            guess = param(WL_range, RR_range)
            
            par1, cov1, QL1, sigmaQL1, r2 = fitting_Lor(WL_range, RR_range, guess)

            guess_double = param_double(WL_range, RR_range)
            
            par2, cov2, QL2, sigmaQL2, QL3, sigmaQL3, r2_double = fitting_doubleLor(WL_range, RR_range, guess_double)

            print(counter, r2_double, r2)

            if np.abs(par2[1]-par2[4]) > 0.6*np.max([par2[2], par2[5]]) and r2_double > r2+0.025: #and abs(par2[0] - par2[3]) < 0.8*(np.max([np.abs(par2[0]), np.abs(par2[3])])):
                print("split")
                r2_tosave = r2_double
                peak_split = True
            else:
                r2_tosave = r2
                peak_split = False
        except RuntimeError:
            print(f"RuntimError at WL = {guess[1]}")
            return None

        print(r2_double, r2)
        fine_WL = np.linspace(f_range[0], f_range[1], 2000)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot()

        ax.plot(WL_range, RR_range, '.',
                markersize=2,
                markeredgecolor=None,
                markerfacecolor=(0, 0, 0, 1),
                linestyle='-',
                linewidth=1,
                label='Data',
                color='tab:blue')

        if not peak_split:
            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par1[0], par1[1], par1[2], par1[3], par1[4]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par1[1], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov1[1][1]), 3))
                          + ', Q$_L$=' + str(int(QL1)) + '$\pm$' + str(int(sigmaQL1)),
                    color='tab:orange')

        if peak_split:
            ax.plot(fine_WL,
                    double_Lorentzian_sq(fine_WL, *par2),
                    linestyle='-',
                    linewidth=2.0,
                    label='Double Lorentzian fit',
                    color='tab:orange')

            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par2[0], par2[1], par2[2], par2[6], par2[7]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par2[1], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov2[1][1]), 3))
                          + ', Q$_L$=' + str(int(QL2)) + '$\pm$' + str(int(sigmaQL2)),
                    color='tab:red',
                    alpha=0.7)
            # amp1, cen1, wid1, amp2, cen2, wid2, b, c
            #  + " R²=" + f'{r2_double}'
            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par2[3], par2[4], par2[5], par2[6], par2[7]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par2[4], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov2[4][4]), 3))
                          + ', Q$_L$=' + str(int(QL3)) + '$\pm$' + str(int(sigmaQL3)),
                    color='tab:purple',
                    alpha=0.7)
        
        fig.show()

        if save_fit:
            filename = f"{File_to_save}{sample_nr}_ParamFile.txt"
            print(filename)
            if not os.path.isfile(filename):
                with open(filename, "w+") as f:
                    f.write(
                        'StartWavelength[nm] EndWavelength[nm] w_r[nm] sigma(w_r)[nm] width[nm] sigma(width)[nm] Q_L[-] sigma(Q_L)[nm] Amplitude[-] sigma(Amp)[-] w_r1[nm] sigma1(w_r)[nm] width1[nm] sigma1(width)[nm] Q_L1[-] sigma1(Q_L)[nm] Amplitude1[-] sigma1(Amp)[-] b[-] c[-] R2\n')

        if not peak_split:
            with open(filename, "a+") as f:
                array = np.c_[f_range[0], 
                              f_range[1], 
                              par1[1], 
                              np.sqrt(cov1[1][1]),
                              par1[2], 
                              np.sqrt(cov1[2][2]),
                              QL1, 
                              sigmaQL1,
                              par1[0], 
                              np.sqrt(cov1[0][0]), 
                              np.NaN, np.NaN, np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN, 
                              par1[3], 
                              par1[4],
                              r2]
                np.savetxt(f, array)
            f.close()

        if peak_split:
            with open(filename, "a+") as f:
                array = np.c_[f_range[0], 
                              f_range[1], 
                              par2[1], 
                              np.sqrt(cov2[1][1]),
                              par2[2], 
                              np.sqrt(cov2[2][2]),
                              QL2, 
                              sigmaQL2,
                              par2[0], 
                              np.sqrt(cov2[0][0]), 
                              par2[4], 
                              np.sqrt(cov2[4][4]),
                              par2[5], 
                              np.sqrt(cov2[5][5]),
                              QL3, 
                              sigmaQL3,
                              par2[3], 
                              np.sqrt(cov2[3][3]), 
                              par2[6], 
                              par2[7],
                              r2_double]
                np.savetxt(f, array)
            f.close()

        plot_id = '_Fit_' + str(int(f_range[0])) + 'to' + str(int(f_range[1])) + 'nm_id'+ str(counter) +'.png'

        ax.tick_params(axis='both', labelsize=tick_sizes)
        ax.tick_params(axis='both', labelsize=tick_sizes)
        ax.set_xlabel('Wavelength', fontsize=axes_sizes)
        ax.set_ylabel('Relative reflection [-]', fontsize=axes_sizes)
        ax.legend(loc='upper left')
        ax.set_title('Peak #'+str(counter)+', R$^2$='+str(np.round(r2_tosave,3)))
        fig.tight_layout()
        
        if r2_tosave > 0.95:
            plt.close()
            
        if save_plot:
            fig.savefig(File_to_save + plot_id, dpi=600)
            
        print(counter)
        
def Pixner_fit(f_guess, WL, RR, sample_nr, save_path, counter, f_range):
    
    sample_nr = sample_nr[2:-1]
    save_plot = 1
    save_fit = 1
    p_fit = 1
    
    # %%Plotting params
    axes_sizes = 12
    tick_sizes = 10
    


    try:
        f_b_idx = np.where(WL >= f_range[0])[0][0]
        f_t_idx = np.where(WL >= f_range[1])[0][0]
        WL_range_raw = WL[f_b_idx:f_t_idx]
        RR_range_raw = RR[f_b_idx:f_t_idx]
    except IndexError:
        print("IndexError")
        return None
    
    #findFitRange(WL_range_raw, RR_range_raw, 3)
    WL_range = WL_range_raw
    RR_range = RR_range_raw
    
    # save data in this directory
    directory = save_path + sample_nr
    if not os.path.exists(directory):
        os.makedirs(directory)
    File_to_save = save_path + sample_nr + f'/'

    #Simple Lorentzian fit
    guess = [-0.0001, f_guess, 0.01, 0,0]
    #Double Lorentzian fit
    guess_double = [-0.0001, f_guess-0.01, 0.005, -0.0001, f_guess, 0.005, 0,0]

    # %%Fitting procedure
    if p_fit:
        try:
            guess = param(WL_range, RR_range)
            
            par1, cov1, QL1, sigmaQL1, r2 = fitting_Lor(WL_range, RR_range, guess)

            guess_double = param_double(WL_range, RR_range)
            
            par2, cov2, QL2, sigmaQL2, QL3, sigmaQL3, r2_double = fitting_doubleLor(WL_range, RR_range, guess_double)

            print(counter, r2_double, r2)

            if np.abs(par2[1]-par2[4]) > 0.6*np.max([par2[2], par2[5]]) and r2_double > r2+0.025: #and abs(par2[0] - par2[3]) < 0.8*(np.max([np.abs(par2[0]), np.abs(par2[3])])):
                print("split")
                r2_tosave = r2_double
                peak_split = True
            else:
                r2_tosave = r2
                peak_split = False
        except RuntimeError:
            print(f"RuntimError at WL = {guess[1]}")
            return None

        print(r2_double, r2)
        fine_WL = np.linspace(f_range[0], f_range[1], 2000)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot()

        ax.plot(WL_range, RR_range, '.',
                markersize=2,
                markeredgecolor=None,
                markerfacecolor=(0, 0, 0, 1),
                linestyle='-',
                linewidth=1,
                label='Data',
                color='tab:blue')

        if not peak_split:
            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par1[0], par1[1], par1[2], par1[3], par1[4]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par1[1], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov1[1][1]), 3))
                          + ', Q$_L$=' + str(int(QL1)) + '$\pm$' + str(int(sigmaQL1)),
                    color='tab:orange')

        if peak_split:
            ax.plot(fine_WL,
                    double_Lorentzian_sq(fine_WL, *par2),
                    linestyle='-',
                    linewidth=2.0,
                    label='Double Lorentzian fit',
                    color='tab:orange')

            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par2[0], par2[1], par2[2], par2[6], par2[7]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par2[1], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov2[1][1]), 3))
                          + ', Q$_L$=' + str(int(QL2)) + '$\pm$' + str(int(sigmaQL2)),
                    color='tab:red',
                    alpha=0.7)
            # amp1, cen1, wid1, amp2, cen2, wid2, b, c
            #  + " R²=" + f'{r2_double}'
            ax.plot(fine_WL,
                    Lorentzian_sq(fine_WL, par2[3], par2[4], par2[5], par2[6], par2[7]),
                    linestyle='-',
                    linewidth=2.0,
                    label='Lorentzian fit: $\omega_r$=' + str(round(par2[4], 3)) + '$\pm$' + str(
                        round(np.sqrt(cov2[4][4]), 3))
                          + ', Q$_L$=' + str(int(QL3)) + '$\pm$' + str(int(sigmaQL3)),
                    color='tab:purple',
                    alpha=0.7)
        
        fig.show()

        if save_fit:
            filename = f"{File_to_save}{sample_nr}_ParamFile.txt"
            print(filename)
            if not os.path.isfile(filename):
                with open(filename, "w+") as f:
                    f.write(
                        'StartWavelength[nm] EndWavelength[nm] w_r[nm] sigma(w_r)[nm] width[nm] sigma(width)[nm] Q_L[-] sigma(Q_L)[nm] Amplitude[-] sigma(Amp)[-] w_r1[nm] sigma1(w_r)[nm] width1[nm] sigma1(width)[nm] Q_L1[-] sigma1(Q_L)[nm] Amplitude1[-] sigma1(Amp)[-] b[-] c[-] R2\n')

        if not peak_split:
            with open(filename, "a+") as f:
                array = np.c_[f_range[0], 
                              f_range[1], 
                              par1[1], 
                              np.sqrt(cov1[1][1]),
                              par1[2], 
                              np.sqrt(cov1[2][2]),
                              QL1, 
                              sigmaQL1,
                              par1[0], 
                              np.sqrt(cov1[0][0]), 
                              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                              par1[3], 
                              par1[4],
                              r2]
                np.savetxt(f, array)
            f.close()

        if peak_split:
            with open(filename, "a+") as f:
                array = np.c_[f_range[0], 
                              f_range[1], 
                              par2[1], 
                              np.sqrt(cov2[1][1]),
                              par2[2], 
                              np.sqrt(cov2[2][2]),
                              QL2, 
                              sigmaQL2,
                              par2[0], 
                              np.sqrt(cov2[0][0]), 
                              par2[4], 
                              np.sqrt(cov2[1][1]),
                              par2[5], 
                              np.sqrt(cov2[2][2]),
                              QL3, 
                              sigmaQL3,
                              par2[3], 
                              np.sqrt(cov2[0][0]), 
                              par2[6], 
                              par2[7],
                              r2_double]
                np.savetxt(f, array)
            f.close()

        plot_id = '_Fit_' + str(int(f_range[0])) + 'to' + str(int(f_range[1])) + 'nm_id'+ str(counter) +'.png'

        ax.tick_params(axis='both', labelsize=tick_sizes)
        ax.tick_params(axis='both', labelsize=tick_sizes)
        ax.set_xlabel('Wavelength', fontsize=axes_sizes)
        ax.set_ylabel('Relative reflection [-]', fontsize=axes_sizes)
        ax.legend(loc='upper left')
        ax.set_title('Peak #'+str(counter)+', R$^2$='+str(np.round(r2_tosave,3)))
        fig.tight_layout()
        
        if r2_tosave > 0.95:
            plt.close()
            
        if save_plot:
            fig.savefig(File_to_save + plot_id, dpi=600)
            
        print(counter)
