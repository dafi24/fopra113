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
import datetime


def assign_filename(direc, stamp):
    e = datetime.datetime.now()

    return direc+'/'+str(e.year)+str(e.month)+str(e.day)+'_'+str(e.hour)+str(e.minute)+stamp

def Lorentzian_sq(x, amp1, cen1, wid1, b, c):
    gamma = wid1 / 2
    return ((amp1 * gamma / ((x - cen1) ** 2 + gamma ** 2)) + b + c*x)**2



def double_Lorentzian_sq(x, amp1, cen1, wid1, amp2, cen2, wid2, b, c):
    gamma1 = wid1 / 2
    gamma2 = wid2 / 2
    return ((amp1 * gamma1 / ((x - cen1) ** 2 + gamma1 ** 2)) + (amp2 * gamma2 / ((x - cen2) ** 2 + gamma2 ** 2)) + b +c*x)**2


def fitting_Lor(data_x, data_y, guess_):
    bounds = [-np.inf, 0, 0, -np.inf, -np.inf], [0, np.inf, np.inf, np.inf, np.inf]
    par_, cov_ = curve_fit(Lorentzian_sq, data_x, data_y, guess_, maxfev=2000, bounds=bounds)
    QL_ = par_[1] / par_[2]
    sigmaQL_ = np.sqrt(cov_[1][1] * (1 / par_[2]) ** 2 + cov_[2][2] * (par_[1] / (par_[2]) ** 2) ** 2)
    y_pred = Lorentzian_sq(data_x, *par_)
    r2_ = r2_score(data_y, y_pred)

    return par_, cov_, QL_, sigmaQL_, r2_


def fitting_doubleLor(data_x, data_y, guess_):
    # amp1, cen1, wid1, amp2, cen2, wid2, b, c
    bounds = [-np.inf, data_x[int(len(data_x)*0.2)], 0, -np.inf, data_x[0], 0, -np.inf, -np.inf], [0, data_x[int(len(data_x)*0.8)], np.inf, 0, data_x[-1], np.inf, np.inf, np.inf]
    par_, cov_ = curve_fit(double_Lorentzian_sq, data_x, data_y, guess_, maxfev=4000, bounds=bounds)
    y_pred = double_Lorentzian_sq(data_x, *par_)
    r2_ = r2_score(data_y, y_pred)

    QL1_ = par_[1] / par_[2]
    sigmaQL1_ = np.sqrt(cov_[1][1] * (1 / par_[2]) ** 2 + cov_[2][2] * (par_[1] / (par_[2]) ** 2) ** 2)

    QL2_ = par_[4] / par_[5]
    sigmaQL2_ = np.sqrt(cov_[4][4] * (1 / par_[5]) ** 2 + cov_[5][5] * (par_[4] / (par_[5]) ** 2) ** 2)
    QL1_ = np.abs(QL1_)
    QL2_ = np.abs(QL2_)
    return par_, cov_, QL1_, sigmaQL1_, QL2_, sigmaQL2_, r2_

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
        
            
            

class SelectiveFitter:
    def __init__(self, graph, r2_thresh, filename) -> None:
        self.graph = graph
        self.xs = np.array(graph.get_xdata())
        self.ys = np.array(graph.get_ydata())
        self.cid = graph.figure.canvas.mpl_connect("key_press_event", self)
        self.r2_thresh = r2_thresh
        self.filename = filename

    def __call__(self, event):
            self.x_lim = event.canvas.figure.axes[0].get_xlim()
            self.y_lim = event.canvas.figure.axes[0].get_ylim()
            
            idx0 = np.where(self.xs>=self.x_lim[0])[0][0]
            idx1 = np.where(self.xs>=self.x_lim[1])[0][0]
            
            selected_x = self.xs[idx0:idx1]
            selected_y = self.ys[idx0:idx1]
            selected_x = selected_x[(selected_y>self.y_lim[0])&(selected_y<self.y_lim[1])]
            selected_y = selected_y[(selected_y>self.y_lim[0])&(selected_y<self.y_lim[1])]
            
            save_fit = False
            
            
            try:
                guess = param(selected_x, selected_y)            
                par1, cov1, QL1, sigmaQL1, r2 = fitting_Lor(selected_x, selected_y, guess)
    
                guess_double = param_double(selected_x, selected_y)            
                par2, cov2, QL2, sigmaQL2, QL3, sigmaQL3, r2_double = fitting_doubleLor(selected_x, selected_y, guess_double)
                
                print(r2_double, r2)
    
                if np.abs(par2[1]-par2[4]) > 0.6*np.max([par2[2], par2[5]]) and r2_double > r2+0.025 and abs(par2[0] - par2[3]) < 0.8*(np.max([np.abs(par2[0]), np.abs(par2[3])])):
                    print("split")
                    r2_tosave = r2_double
                    self.optimizer = double_Lorentzian_sq
                    
                    par = par2
                    cov = cov2
                    print(str(int(QL2))+'+/-'+str(int(sigmaQL2)))
                    print(str(int(QL3))+'+/-'+str(int(sigmaQL3)))
                    peak_split = True
                    
                else:
                    r2_tosave = r2
                    self.optimizer = Lorentzian_sq
                    par = par1
                    cov = cov1
                    print(str(int(QL1))+'+/-'+str(int(sigmaQL1)))
                    peak_split = False
 
            
            except RuntimeError:
                print(f"RuntimError at WL = {guess[1]}")
                return None
            
            
            
            event.canvas.figure.axes[0].plot(np.linspace(selected_x[0], selected_x[-1], 1000),self.optimizer(np.linspace(selected_x[0], selected_x[-1], 1000),*par), label="$Q_l$="+str(int(QL1))+'$\pm$'+str(int(sigmaQL1)), color='tab:orange')
            event.canvas.figure.axes[0].legend()
            

        

# fig, ax = plt.subplots()
# data = np.random.rand(100)
# graph, = ax.plot(data, "o", label='Hello')
# ax.legend()
# selectivefitter= SelectiveFitter(graph, linear)


# plt.show()