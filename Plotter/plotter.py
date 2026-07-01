import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import shutil
from test_callback import *
from functions_double import open_h5, normalization

# Import the necessary peak detection and automated fitting functions
from functions_double import open_h5, normalization, detect_peaks, Pixner_fit

# --- Change working directory to the script's location ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#%%File import

#Unpack the data
measurement_number='50420'
file = '0000' + measurement_number + '-FoPraWavelengthScan.h5'
base_dir = r'/mnt/c/Users/david/Documents/PLOCHA/SKOLA (TUM)/_modules/fopra/113_siliconChip/2026-05-07/16'
path = os.path.join(base_dir, file)
# path = os.path.join(r'/Volumes/eqn/quantumnetworks/Data/2024/RT1/2024-07-18/16/'+file) #for mac
save_path = "Plots/"
WL, RR, sample_nr = open_h5(path)

print("Sample number:", sample_nr)

directory = save_path + sample_nr[2:-1]
if os.path.exists(directory + '_' + measurement_number + '/'):
    shutil.rmtree(directory + '_' + measurement_number + '/')
    
os.makedirs(directory + '_' + measurement_number + '/')

filename = 'plot_' + sample_nr[2:-1] + '.png'
full_save_path = os.path.join((directory + '_' + measurement_number + '/'), filename)


#%%figure    
fig1=plt.figure(num=1, figsize=(10,4))
fig1.clf()
ax1=fig1.add_subplot(111)

ax1.set_title('Structure')
graph, = ax1.plot(WL, RR, '.')
ax1.set_ylabel('Relative reflection [-]')   
ax1.set_xlabel('Wavelength [nm]')

# selectivefitter= SelectiveFitter(graph, 0.7)

timestamp = assign_filename(directory + '_' + measurement_number + '/', '_ParamFile_50405.txt')
selectivefitter= SelectiveFitter(graph, 0.7, timestamp)



fig1.tight_layout()
#ax1.grid()
fig1.savefig(full_save_path,dpi=1200)
print("Full save path:", full_save_path)



#%% Automated Peak Detection and Fitting
print("\n--- Starting Automated Fitting ---")

# 1. Detect peaks (dips) in the data
# find_dips=True inverts the data to find dips instead of peaks.
# You may need to tune 'prominence' and 'distance' based on the noise level.
# bigger prominence => only deeper peaks detected, etc SEE GEMINI FOR DETAIL
peak_indices = detect_peaks(RR, find_dips=True, prominence=0.15, distance=200)

print(f"Found {len(peak_indices)} dips. Proceeding to fit...")

# 2. Loop through each dip, fit it, and save parameters
window_nm = 0.5  # Wavelength window to grab around the center of the dip

for counter, idx in enumerate(peak_indices):
    f_guess = WL[idx]
    f_range = [f_guess - window_nm, f_guess + window_nm]
    
    # Pixner_fit automatically handles:
    # - Cropping the data to f_range
    # - Running Genetic Algorithms to find initial guesses
    # - Fitting single and double Lorentzians
    # - Choosing the best fit based on R^2
    # - Saving individual plots and appending to ParamFile.txt
    print(f"\nFitting peak #{counter} at ~{f_guess:.2f} nm...")
    
    # Inject the measurement number into the string format that Pixner_fit expects
    # so its internal [2:-1] slicing targets your custom folder perfectly.
    modified_sample_nr = f"b'{sample_nr[2:-1]}_{measurement_number}'"
    
    Pixner_fit(f_guess=f_guess, WL=WL, RR=RR, sample_nr=modified_sample_nr, 
               save_path=save_path, counter=counter, f_range=f_range)

print("\n--- Automated Fitting Complete! ---")