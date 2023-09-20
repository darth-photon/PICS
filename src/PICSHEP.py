# from pyfiglet import figlet_format
import time
import math as mtp

import csv
import os
import sys
import warnings

import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from numpy import linalg as LA
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp


# text = figlet_format("PICSHEP", font="starwars")
# print(text)
print("**************************************************************************")
print("Particle Interactions Cascade equation Solver for High Energy Physics")
print()
print("Author: シヴァサンカール ShivaSankar K.A")
print()
print("Affiliation: "+u"北海道大学宇宙理学専攻、大学院理学院, 北海道大学\n Department of CosmoSciences, Graduate School of Science, Hokkaido University")
print()
print("Email: shivasankar.ka@gmail.com")
print()
print("Last update: 2023/03/27")
print("**************************************************************************")
time.sleep(1)


class CES():
    """
    Particle Interactions Cascade equation Solver for High Energy Physics.

    This class provides methods for solving the cascade equation and calculating the number of events and attenuated flux for a given energy range and observation time.

    Attributes:
        e_min (float): The minimum energy.
        e_max (float): The maximum energy.
        N (int): The number of points in the range.
        N_eig (int): The number of eigenvectors.
        model (Model): The model used for calculations.

    Methods:
        set_model(model): Set the model for calculations.
        eigcalc(Energy, num, a, b): Calculate the attenuated flux at the required energy.
        events(e_min, e_max, t_obs, A_range, B_range, N=20, N_eig=20): Calculate the number of events for given energy range and observation time.
        total_events(e_min, e_max, Aval, Bval, t_obs, N_eig): Calculate the total number of events.
        attenuated_flux(e_min, e_max, N_eig, Aval, Bval): Calculate the attenuated flux.
    """
    def __init__(self):
        self.e_min = None
        self.e_max = None
        self.N = None
        self.N_eig = None

    def set_model(self, model) -> None:
        """
        Set the model for calculations.

        Args:
            model (Model): The model used for calculations.
        """
        self.model = model    

    def eigcalc(self, energy: float, num: int, a_mat: float, b_mat: float) -> np.ndarray:
        """
        Calculate the attenuated flux at the required energy E.

        Args:
            Energy (float): The required energy.
            num (int): The number of energy points.
            a (float): Parameter a.
            b (float): Parameter b.

        Returns:
            np.ndarray: The attenuated flux at the required energy E.
        """
        model = self.model
        e = np.logspace(np.log10(model.e_min), np.log10(model.e_max), num, dtype=np.float64)
        delta_e = np.diff(np.log(e)) 
        
        phi_0 = model.flux(e)

        sigma_array = model.xs(e, a_mat, b_mat, 9/b_mat)

        dxs_array = np.triu(model.dxs(e[:, None], e, a_mat, b_mat, 9/b_mat))
        
        rhn = np.zeros((len(e), len(e)))
        i_upper, j_upper = np.triu_indices(len(e), 1)
        rhn[i_upper, j_upper] = delta_e[j_upper - 1] * dxs_array[j_upper, i_upper] * e[j_upper]**1

        # calculating eigenvalues, eigenvectors and solving for the coefficients
        w, v = LA.eig(-np.diag(sigma_array) + rhn)
        ci = LA.solve(v, phi_0)
        phisol = np.dot(v, (ci * np.exp(w)))
        return np.interp(energy, e, phisol)
    
    def events(self, e_min: float, e_max: float, t_obs: float, a_range: list[float], b_range: list[float], n_val: int = 20, n_eig: int = 20) -> None:
        """
        Calculate the number of events for given energy range and observation time.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            t_obs (float): The observation time.
            A_range (List[float]): The range of parameter A.
            B_range (List[float]): The range of parameter B.
            N (int, optional): The number of points in the range. Defaults to 20.
            N_eig (int, optional): The number of eigenvectors. Defaults to 20.
        """
        self.N = n_val
        self.N_eig = n_eig
        model = self.model

        self.e_min = e_min
        self.e_max = e_max
        aval = np.logspace(a_range[0], a_range[1], num=self.N, endpoint=True) 
        bval = np.logspace(b_range[0], b_range[1], num=self.N, endpoint=True)

        steps = 20000
        delta_e = (10**np.log10(self.e_max) - 10**np.log10(self.e_min)) / steps
        enn = np.linspace(10**np.log10(self.e_min), 10**np.log10(self.e_max), steps)

        start_time = time.time()
        if os.path.exists("events_data/events.txt"):
            os.remove("events_data/events.txt")
        print("\n")
        
        with open('events_data/events.txt', mode='a', newline='', encoding='utf-8') as file:
            for i in range(self.N):
                for j in range(self.N):
                    tmp = 0.0
                    tmp = t_obs * np.sum(self.eigcalc(enn, self.N_eig, aval[i], bval[j]) * model.eff_area(enn)) * delta_e
                    file.write(f"{aval[i]} {bval[j]} {tmp}\n")
        end_time = time.time()
        print("\nTime taken: ", end_time - start_time, " seconds\n")

    def total_events(self, e_min: float, e_max: float, a_val: float, b_val: float, t_obs: float, n_eig: int) -> None:
        """
        Calculate the total number of events for a given energy range, observation time, and parameter values.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            Aval (float): The value of parameter A.
            Bval (float): The value of parameter B.
            t_obs (float): The observation time.
            N_eig (int): The number of eigenvectors.
        """
        self.e_min = e_min
        self.e_max = e_max
        model  = self.model
        steps = 100000
        delta_e = (10**np.log10(self.e_max)-10**np.log10(self.e_min))/steps
        enn = np.linspace(10**np.log10(self.e_min),10**np.log10(self.e_max),steps)
        print("No of events: " + str(np.sum(t_obs* self.eigcalc(enn, n_eig, a_val, b_val)*model.eff_area(enn))*delta_e))

    def attenuated_flux(self, e_min: float, e_max: float, n_eig: int, a_val: float, b_val: float) -> None:
        """
        Calculate the attenuated flux for a given energy range, number of eigenvectors, and parameter values.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
            n_eig (int): The number of eigenvectors.
            a_val (float): The value of parameter A.
            b_val (float): The value of parameter B.
        """
        self.e_min = e_min
        self.e_max = e_max
        model = self.model
        enn = np.linspace(10**np.log10(self.e_min),10**np.log10(self.e_max),100)
        phi = self.eigcalc(enn,n_eig, a_val, b_val)
        phi_0 = model.flux(enn)
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
            
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        plt.plot(enn, phi,color ='r',label=r"$\Phi_{att}$")
        plt.plot(enn, phi_0,color ='g',label=r"$\Phi$")
        plt.legend()
        plt.ylabel(r"$\Phi$",fontsize=22)
        plt.xlabel(r"$E\mathrm{~(TeV)}$",fontsize=22)
        plt.show()

    def new_physics(self, x_coords, y_coords):
        """
        Calculate the new physics parameters based on the given x and y coordinates.

        Args:
            x_coords (np.ndarray): The x coordinates.
            y_coords (np.ndarray): The y coordinates.
        """
        model = self.model
        mvsg = np.empty([len(x_coords),2])
        for i, x in enumerate(x_coords):
            # mvsg[i,0] = np.sqrt(dmmass/(y_coords[i]))
            # mvsg[i,1] = mvsg[i,0]**2 * np.sqrt(x_coords[i]/(SigmaChi * 10**3)) * np.sqrt(4*np.pi)
            mvsg[i, 0] = model.m_eqn(model.dm_mass, y_coords[i])
            sigma_chi = model.get_sigma(3 * mvsg[i, 0])
            mvsg[i, 1] = model.g_eqn(mvsg[i, 0], x, sigma_chi)
        return mvsg, model.dm_model, model.dm_mass

    def plot(self, xlim: float, ylim: float, title: str, do_plot: bool, plot_save: bool, event_threshold: float) -> None:
        """
        Plot the data with specified x and y limits, title, and plot options.

        Args:
            xlim (float): The x-axis limits.
            ylim (float): The y-axis limits.
            title (str): The title of the plot.
            do_plot (bool): Whether to display the plot.
            plotsave (bool): Whether to save the plot.
            event_threshold (float): The event threshold for contour plot.
        """
        dat_fin = np.loadtxt("events_data/events.txt", delimiter=" ")
        df = pd.DataFrame(dat_fin, columns = ['Column_A','Column_B','Column_C'])
        xcol, ycol, zcol = 'Column_A', 'Column_B', 'Column_C'
        df = df.sort_values(by=[xcol, ycol])

        xvals = df[xcol].unique()
        yvals = df[ycol].unique()
        zvals = df[zcol].values.reshape(len(xvals), len(yvals)).T
        
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
            
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')

        CP = plt.contour(xvals, yvals, zvals, levels=[event_threshold], colors='r',linestyles='solid')

        x_coords = CP.allsegs[0][0][:,0]
        y_coords = CP.allsegs[0][0][:,1]

        plt.ylabel(r'$B$',fontsize=22)
        plt.xlabel(r'$A$',fontsize=22)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if plot_save is True:
            if os.path.exists("plots/"+title+".pdf"):
                os.remove("plots/"+title+".pdf")
            plt.savefig("plots/"+title+".pdf")
        with open("events_data/AvsB.txt", mode="w", newline="", encoding='utf-8') as csvfile:
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                csvfile.write(f"{x} {y}\n")
        if do_plot is False:
            plt.show(block=False)
        else:
            plt.show()
            
    def plot_mvsg(self, title: str, plot_save: bool) -> None:
        """
        Plot the data with specified x and y limits, title, and plot options.

        Args:
            title (str): The title of the plot.
            plot_save (bool): Whether to save the plot.
        """
        model = self.model
        data = np.loadtxt("events_data/AvsB.txt", delimiter=" ")
        mvsg_dat, model, dmmass = self.new_physics(data[:,0],data[:,1])

        mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
        plt.rcParams['axes.linewidth'] = 2
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2

        fig = plt.figure(figsize=(10,8))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        ax = fig.add_subplot(111)
        ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
        ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
            
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xscale('log')
        plt.yscale('log')
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        
        plt.plot(mvsg_dat[:,0], np.sqrt(mvsg_dat[:,1]))

        plt.ylabel(r"$g_{\nu}$",fontsize=22)
        plt.xlabel(r"$m_{Z'}\mathrm{~(GeV)}$",fontsize=22)
        if plot_save is True:
            if os.path.exists("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt"):
                os.remove("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt")
            with open("events_data/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".txt", mode="w", newline="", encoding='utf-8') as csvfile:
                for i in range(len(mvsg_dat)):
                    csvfile.write(f"{mvsg_dat[i,0]} {np.sqrt(mvsg_dat[i,1])}\n")
                    # writer.writerow([mvsg_dat[i,0], mvsg_dat[i,1]])
            if os.path.exists("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf"):
                os.remove("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf")
            plt.savefig("plots/mvsg_"+title+"_m_dm="+str(dmmass)+"-"+str(model)+".pdf")
        
        print("Thy Bidding is done, My Master \n")
        
class Model():
    """
    This class is used to define the required parameters for the model.
    """
    def __init__(self, path="./"):
        self.modelpath = path
        self.dm_mass = 0.0
        
    def set_model_name(self, model_name: str, model_type: str) -> None:
        """
        Set the model name and type.

        Args:
            model_name (str): The name of the model.
            model_type (str): The type of the model.
        """
        self.model_type = model_type
        self.model_name = model_name
        
        if "Blazar" in self.model_name:
            sys.path.append("../Models/Blazar_U1X")
            from DM_density import blazar_DM_density
            self.sigmacalc = blazar_DM_density()
        elif "AGN" in self.model_name:
            sys.path.append("../Models/AGN_U1X")
            from DM_density import AGN_DM_density
            self.sigmacalc = AGN_DM_density()
        else:
            warnings.warn("Invalid model name: {}".format(self.model_name))
            
    def set_energy_range(self, e_min: float, e_max: float) -> None:
        """
        Set the energy range.

        Args:
            e_min (float): The minimum energy.
            e_max (float): The maximum energy.
        """
        self.e_min = e_min
        self.e_max = e_max

    def set_diff_cross_section(self, diff_cross_section: str) -> None:
        """
        Set the differential cross section function with user defined function.

        Args:
            diff_cross_section (str): The user defined function for the differential cross section.
        """
        self.diff_cross_section_eqn = diff_cross_section
        self.dxs = eval(self.diff_cross_section_eqn)

    def set_cross_section(self, cross_section: str) -> None:
        """
        Set the cross section function with user provided data.

        Args:
            cross_section (str): The user defined function for the cross section.
        """
        self.cross_section_eqn = cross_section
        self.xs = eval(self.cross_section_eqn)

    def set_eff_area_data(self, effective_area: str) -> None:
        """
        Set the effective interaction area function with user provided data.

        Args:
            effective_area (str): The file name of the effective area data.
        """
        if os.path.exists(self.modelpath + effective_area):
            data = np.genfromtxt(self.modelpath + effective_area, delimiter=',')
            self.eff_area_data = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value='extrapolate')
            self.eff_area = lambda E: self.eff_area_data(E)[()]
        else:
            warnings.warn("Effective area file not found: {}".format(self.modelpath + effective_area))
    
    def set_eff_area_func(self, effective_area: str) -> None:
        """
        Set the effective interaction area function with user defined function.

        Args:
            effective_area (str): The user defined function for the effective area.
        """
        self.eff_area_eqn = effective_area
        self.eff_area = lambda E: eval(self.eff_area_eqn)
        
    def set_flux_data(self, flux: str) -> None:
        """
        Set the flux function with user provided data.

        Args:
            flux (str): The file name of the flux data.
        """
        if os.path.exists(self.modelpath + flux):
            data = np.genfromtxt(self.modelpath + flux, delimiter=',')
            self.flux_data = interp1d(data[:, 0], data[:, 1], kind='linear', fill_value="extrapolate")
            self.flux = lambda E: self.flux_data(E)[()]
        else:
            warnings.warn("Flux file not found: {}".format(self.modelpath + "input/flux.csv"))
    
    def set_flux_func(self, flux: str) -> None:
        """
        Set the flux function with user defined function.

        Args:
            flux (str): The user defined function for the flux.
        """
        self.flux_eqn = flux
        self.flux = lambda E: eval(self.flux_eqn)

    def set_np_parameterization(self, m: str, g: str) -> None:
        """
        Set the new physics mass and coupling relation with user provided parameterizations.

        Args:
            m (str): The user defined function for the mass.
            g (str): The user defined function for the coupling relation.
        """
        self.m_eqn = lambda dm_mass, B: eval(m)
        self.g_eqn = lambda mzp, A, Sigma: eval(g)
        print("Parameterization of new physics parameters initialized:\n m=" + m + "\n g=" + g + "\n")
    
    def set_dm_model_info(self, model_type: str, model_mass: float) -> None:
        """
        Set the dark matter model information.

        Args:
            model_type (str): The type of the model.
            model_mass (float): The mass of the model.
        """
        self.dm_model = model_type
        self.dm_mass = model_mass

        # Blazar
        # self.models = [["CIA",7.19725*10**25,1.0],["CIIA",5.78693*10**21,1.0],["CIB",7.48421*10**26,0.48],["CIIB",1.64899*10**24,0.73]]
        # AGN
        self.models = [["CIA",2.42932*10**28,1.0],["CIIA",3.62965*10**23,1.0],["CIB",9.51132*10**27,0.48],["CIIB",4.94941*10**23,0.73]]
        [(A := self.models[i][1], B := self.models[i][2]) for i in range(len(self.models)) if self.dm_model == self.models[i][0]]
        A = float(A)
        B = float(B)
        # self.SigmaChi = 10**A * (self.dm_mass)**(1-B) * (1.98* 10**-14)**2 * 10**-3
        self.SigmaChi = A * (1.98* 10**-14)**2
        
    def get_sigma(self, m_chi: float) -> float:
        """
        Calculate the sigma value for a given dark matter mass.

        Args:
            m_chi (float): The dark matter mass.

        Returns:
            float: The calculated sigma value.
        """
        alpha = {
            "CIA": 7/3,
            "CIIA": 7/3,
            "CIB": 3/2,
            "CIIB": 3/2
        }[self.model_type]
        sigma_val = {
            "CIA": 10**-8,
            "CIIA": 3,
            "CIB": 10**-8,
            "CIIB": 3
        }[self.model_type]

        sigma_calc = self.sigmacalc.Sigma(alpha=alpha, r=10**3, m_chi=m_chi, sigma=sigma_val) 
        return sigma_calc * 3.086*(10**18) * (1.98* 10**-14)**2  # conversion factors for pc to cm and GeV to cm
