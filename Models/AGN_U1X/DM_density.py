import numpy as np
from scipy.integrate import quad
from numba import njit
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class AGN_DM_density():
    def __init__(self) -> None:
        self.tBH = 10**9 * 365 * 24 * 3600
        self.RS = 2.97 * 10**-6  # pc
        self.ri = 4 * self.RS
        self.RS = 4.84814 * 10**-6 * 0.1986  # pc
        self.rs = 13 * 10**3  # pc
        self.Rho_s = 0.35  # GeV/cm^3
        self.Rsp = 0.7 * 10**3  # pc
        self.rh = 0.65  # pc
        self.ri = 4 * self.RS
   
    def f(self, alpha, r):
        return r**(-alpha) * (r**3/(3 - alpha) + 12*self.RS* r**2/(alpha - 2) - 48*self.RS**2 * r/(alpha - 1) + 64*self.RS**3/alpha)
    
    def rho_c(self, m_chi, sigma):
        return m_chi/(sigma * 10**-26 * self.tBH)  # GeV/cm^3

    def Num(self, alpha):
        return 10**7*(1.988 * 10**30 * 5.62 * 10**26)/(4 * np.pi * (self.f(alpha, self.rh) - self.f(alpha, self.ri)))

    def rhoN(self, alpha):
        if alpha == 3/2:
            return self.Num(alpha) / self.rh**(3/2) * 1/(3.086 * 10**18)**3
        elif alpha == 7/3:
            return self.Num(alpha) / self.Rsp**(7/3) * 1/(3.086 * 10**18)**3
  
    def rhoNp(self, alpha):
        return self.Num(alpha) * (self.rh**(5/6)/self.Rsp**(7/3)) * 1/(3.086 * 10**18)**3
  
    def rho_alpha(self, alpha, r):
        if alpha == 3/2:
            if self.ri <= r <= self.rh:
                return self.rhoN(3/2) * (1 - 4*self.RS/r)**3 * (self.rh/r)**(3/2)
            elif r > self.rh:
                return self.rhoNp(3/2) * (self.Rsp/r)**(7/3)
        elif alpha == 7/3:
            if r >= self.ri:
                return self.rhoN(7/3) * (1 - 4*self.RS/r)**3 * (self.Rsp/r)**(7/3)
            else:
                return 0
        return 0

    def rhoNFW(self, r):
        return self.Rho_s * (r/self.rs)**-1 * (1 + r/self.rs)**-2

    def rho_chi(self, alpha, r, m_chi, sigma):
        if r <= self.ri:
            return 0
        elif self.ri < r <= self.Rsp:
            return self.rho_alpha(alpha, r) * self.rho_c(m_chi, sigma) / (self.rho_c(m_chi, sigma) + self.rho_alpha(alpha, r))
        elif r > self.Rsp:
            return self.rhoNFW(r) * self.rho_c(m_chi, sigma) / (self.rho_c(m_chi, sigma) + self.rhoNFW(r))
    
    def Sigma(self, alpha, r, m_chi, sigma):
        rmin = self.RS*30
        rmax = 10**4
        # x = np.logspace(np.log10(rmin), np.log10(rmax), 20000)
        # dx = np.diff(x)
        # dat = np.zeros((len(x)))
        # for i in range(len(dx)):
        #     dat[i] = self.rho_chi(alpha, x[i], m_chi, sigma) * dx[i]
        # result = np.sum(dat)
        x = np.logspace(np.log10(rmin),np.log10(r), num=5000)
        dx = np.diff(x)
        dat = np.zeros((len(dx)))
        for i in range(len(dx)):
            dat[i] = self.rho_chi(alpha, x[i], m_chi, sigma) * dx[i]
        result = np.sum(dat)
        # result, _ = quad(lambda x: self.rho_chi(alpha, x, m_chi, sigma), rmin, r, epsabs=1e-12, epsrel=1e-12)
        return result
       
# sigmacalc = blazar_DM_density()
# x = np.logspace(np.log10(0.00002958), 4, 5000)
# rho_chi_values = np.zeros(len(x))
# rho_chi_values1 = np.zeros(len(x))
# for i in range(len(x)):
#     rho_chi_values[i]  = sigmacalc.rho_chi(7/3, x[i], 10**-6, 10**-8)
#     rho_chi_values1[i] = sigmacalc.rho_chi(3/2, x[i], 10**-6, 10**-8)
# plt.plot(x, rho_chi_values, color='r')
# plt.plot(x, rho_chi_values1, color='b')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('r')
# plt.ylabel(r'$\rho_\chi$')
# plt.show()

####
# sigmacalc = AGN_DM_density()
# m = np.logspace(-5, 3, 50)
# rho_chi_values = np.zeros(len(m))
# rho_chi_values1 = np.zeros(len(m))
# rho_chi_values2 = np.zeros(len(m))
# rho_chi_values3 = np.zeros(len(m))
# for i in range(len(m)):
#     rho_chi_values[i]  = sigmacalc.Sigma(7/3, m[i], 10**-6, 10**-8) *  3.086*(10**18)
#     rho_chi_values1[i] = sigmacalc.Sigma(7/3, m[i], 10**-6, 3) *  3.086*(10**18) 
#     rho_chi_values2[i]  = sigmacalc.Sigma(3/2, m[i], 10**-6, 10**-8) *  3.086*(10**18)
#     rho_chi_values3[i]  = sigmacalc.Sigma(3/2, m[i], 10**-6, 3) *  3.086*(10**18)

# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams.update({'font.size': 16})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['axes.linewidth'] = 2
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot()
# ax1.set_facecolor('white')      
# plt.plot(m, rho_chi_values, color='r', label="CIA")
# plt.plot(m, rho_chi_values1, color='r', label="CIIA", linestyle='dashed')
# plt.plot(m, rho_chi_values2, color='b', label="CIB")
# plt.plot(m, rho_chi_values3, color='b', label="CIIB", linestyle='dashed')

# ax1.tick_params(which='major',direction='in',width=2,length=7,top=True,right=True, pad=7)
# ax1.tick_params(which='minor',direction='in',width=1,length=5,top=True,right=True)

# ax1.set_xlabel("r [pc]")
# ax1.set_ylabel(r"$\Sigma(r) [\mathrm{GeV/cm^2}]$")

# ax1.set_yscale('log')
# ax1.set_xscale('log')

# ax1.set_xlim([10**-6,10**3])
# # ax1.set_ylim([10**0.0,10**20.0])

# ax1.set_xticks(10**np.arange(-6.0,3.1, 2))
# # ax1.set_yticks(10**np.arange(0.0,20.1, 4))

# legend =ax1.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, bbox_to_anchor=(0.9,0.2), ncol=2, borderaxespad=0, prop={'size': 13})
# legend.get_frame().set_facecolor('white')

# ax1.text(10**0.1, 10**18.5, r'$m_{DM} = 1\;\mathrm{keV}$', ha='center', va='center',
#                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),fontsize = 15)

# #############################
# # Show the plot
# plt.tight_layout()
# plt.savefig("DM_AGN_r.pdf",dpi=500)
# plt.show()

#####
sigmacalc = AGN_DM_density()
m = np.logspace(-6, 3, 50)
rho_chi_values = np.zeros(len(m))
rho_chi_values1 = np.zeros(len(m))
rho_chi_values2 = np.zeros(len(m))
rho_chi_values3 = np.zeros(len(m))
for i in range(len(m)):
    rho_chi_values[i]  = sigmacalc.Sigma(7/3, 10**3, m[i], 10**-8) *  3.086*(10**18)
    rho_chi_values1[i] = sigmacalc.Sigma(7/3, 10**3, m[i], 3) *  3.086*(10**18) 
    rho_chi_values2[i]  = sigmacalc.Sigma(3/2,10**3, m[i],10**-8) *  3.086*(10**18)
    rho_chi_values3[i]  = sigmacalc.Sigma(3/2,10**3, m[i], 3) *  3.086*(10**18)

plt.rcParams['axes.linewidth'] = 2
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 2
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot()
ax1.set_facecolor('white')      
plt.plot(m, rho_chi_values/m, color='r', label="CIA")
plt.plot(m, rho_chi_values1/m, color='r', label="CIIA", linestyle='dashed')
plt.plot(m, rho_chi_values2/m, color='b', label="CIB")
plt.plot(m, rho_chi_values3/m, color='b', label="CIIB", linestyle='dashed')

ax1.tick_params(which='major',direction='in',width=2,length=7,top=True,right=True, pad=7)
ax1.tick_params(which='minor',direction='in',width=1,length=5,top=True,right=True)

ax1.set_xlabel("r [pc]")
ax1.set_ylabel(r"$\frac{\Sigma(r)}{m_{DM}} [\mathrm{cm^{-2}}]$")

ax1.set_yscale('log')
ax1.set_xscale('log')

ax1.set_xlim([10**-6,10**3])
# ax1.set_ylim([10**0.0,10**20.0])

ax1.set_xticks(10**np.arange(-6.0,3.1, 2))
# ax1.set_yticks(10**np.arange(0.0,20.1, 4))

legend =ax1.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, bbox_to_anchor=(0.55,0.8), ncol=2, borderaxespad=0, prop={'size': 13})
legend.get_frame().set_facecolor('white')

# ax1.text(10**0.1, 10**18.5, r'$m_{DM} = 1\;\mathrm{keV}$', ha='center', va='center',
#                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),fontsize = 15)

#############################
# Show the plot
plt.tight_layout()
plt.savefig("DM_AGN_mdm.pdf",dpi=500)
plt.show()
