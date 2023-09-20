import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp

data1 = np.loadtxt("/Users/shivasankark.a/Documents/Research_work/Tools/PICSHEP/Models/Blazar_U1X/events_data/mvsg_U(1)X_xH=2_mchi=0.01_m_dm=0.01-BM1.csv",delimiter=",")
data2 = np.loadtxt("/Users/shivasankark.a/Documents/Research_work/Tools/PICSHEP/Models/Blazar_U1X/events_data/mvsg_U(1)X_xH=1_mchi=0.01_m_dm=0.01-BM1.csv",delimiter=",")
data3 = np.loadtxt("/Users/shivasankark.a/Documents/Research_work/Tools/PICSHEP/Models/Blazar_U1X/events_data/mvsg_U(1)X_xH=0_mchi=0.01_m_dm=0.01-BM1.csv",delimiter=",")
data4 = np.loadtxt("/Users/shivasankark.a/Documents/Research_work/Tools/PICSHEP/Models/Blazar_U1X/events_data/mvsg_U(1)X_xH=-1_mchi=0.01_m_dm=0.01-BM1.csv",delimiter=",")
# data5 = np.loadtxt("/Users/shivasankark.a/Documents/Research_work/Tools/PICSHEP/Models/Blazar_U1X/events_data/mvsg_U(1)X_xH=-2_mchi=0.01_m_dm=0.01-BM1.csv",delimiter=",")

plt.rcParams['axes.linewidth'] = 2
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 2
fig = plt.figure(figsize=(8,6))
fig.tight_layout()
plt.subplots_adjust(wspace=0.35)
ax = fig.add_subplot(111)
ax.set_facecolor('white')

ax.tick_params(which='major',direction='in',width=2,length=10,top=True,right=True, pad=7)
ax.tick_params(which='minor',direction='in',width=1,length=7,top=True,right=True)
        
plt.plot(data1[:,0], np.sqrt(data1[:,1]), color='b', label=r"$x_H = 2$", linestyle='dashed')
plt.plot(data2[:,0], np.sqrt(data2[:,1]), color='b', label=r"$x_H = 1$", linestyle='dashdot')    
plt.plot(data3[:,0], np.sqrt(data3[:,1]), color='r', label=r"$x_H = 0$")
plt.plot(data4[:,0], np.sqrt(data4[:,1]), color='g', label=r"$x_H = -1$", linestyle='dashdot')
# plt.plot(data5[:,0], np.sqrt(data5[:,1]), color='g', label=r"$x_H = -2$", linestyle='dashed')

ax.set_xlim([min(data1[:,0]),10**1])
ax.set_ylim([10**-2.5,max(data1[:,1])])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xscale('log')
plt.yscale('log')
# plt.legend(loc='lower right')

legend = plt.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, loc='lower right')
legend.get_frame().set_facecolor('white')

plt.ylabel(r"$g_{\nu}$",fontsize=20)
plt.xlabel(r"$m_{Z^{\prime}} \mathrm{~(GeV)}$",fontsize=20)

# plt.savefig("mvsg_U(1)X.pdf") 
plt.show()