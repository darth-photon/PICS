import sys
src_path = "../../src/"
sys.path.append(src_path)
from PICSHEP import *
CES = ces()
model = Model(path="./")
model.model_name(model_name="AGN",model_type="CIA")

F0 = 13.22
F1 = 1.498
F2 = -0.00167
F3 = 4.119

# charge = [4.0,2.0,1.0,0.0,-1.0]
# dm_modelz = [{'mod':'BM1','sigval':5.74*10**31},{'mod':'BM2','sigval':2.47*10**28},{'mod':'BM3','sigval':9.11*10**26},{'mod':'BM1p','sigval':9.417*10**28},{'mod':'BM2p','sigval':1.00*10**28},{'mod':'BM3p','sigval':1.12*10**27}]

charge = [4.0,2.0,1.0,0.0,-1.0]
# modelz = ['Dirac','Majorana-II','Complex']

# charge = [2.0]
modelz = ['Majorana-II']

for mod in modelz:
    for xH in charge:
        if mod=='Dirac':
                #xchi = 2.0
                xchi = 1000.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")
        
        elif mod=='Majorana-I':
                xchi = 1.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")

        elif mod=='Majorana-II':
                xchi = 5.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Fermionic cross section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(1 + E**2/x**2)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Ev,A,B,mchi: "+str(QX)+"*(2 * B * Ev**2 * ((4 * B * Ev * (2 * B * Ev + 1) + 1) / (4 * B * Ev**2 + 2 * Ev + mchi) + 1 / (2 * Ev + mchi)) - (2 * B * Ev + 1) * np.log(4 * B * Ev**2 + 2 * Ev + mchi) + (2 * B * Ev + 1) * np.log(2 * Ev + mchi))*A/(4 * B**3 * Ev**2)")

        elif mod=='Complex':
                xchi = 1000.0
                xv = -1/2*(xH) - 1.0
                QX = xchi**2 * xv**2
                #Complex scalar section
                model.set_diff_cross_section(diff_cross_section="lambda E,x,a,b,dm_mass: "+str(QX)+"* a*(2*E/x)* 1/((1+2*b*(x-E))**2)")
                model.set_cross_section(cross_section="lambda Evp,A,B,m_chi: "+str(QX)+"* -2*A*Evp**2*m_chi**2*(-((-((2*B*Evp*m_chi)/(4*B*Evp**2+2*Evp+m_chi)) - np.log(4*B*Evp**2+2*Evp+m_chi) + np.log(2*Evp+m_chi))/(4*B**2*Evp**3*m_chi**2))-1/(2*B*Evp**2*m_chi**2))")

        model.set_flux_func(Flux="4.9032578920279493e-11 * (E)**-3.196")

        model.set_eff_area_data(Effective_area="input/eff_area.csv")

        # model.set_NP_parameterization(m="np.sqrt(dm_mass/(B))",
        #                         g="(mzp**2) * np.sqrt(A/(Sigma*10**3)) * np.sqrt(8*np.pi)"
        #                         )
        model.set_NP_parameterization(m="3/(B)",
                                      g="(mzp**2) * np.sqrt(A/(Sigma)) * np.sqrt(8*np.pi)"
                                      )

        model.set_energy_range(Emin=1.5,Emax=15)

        model.DM_model_info(modeltype="CIA",modelmass=10**-3)

        CES.set_model(model=model)

        # CES.attenuated_flux(Emin=1.5, Emax=15, N_eig=50,Aval=0.5,Bval=0.1)

        # for normal charge, Arange = [-3,4], Brange = [-5,4]
        # for xchi=100, Arange = [-6,1], Brange = [-5,4]
        # For Majorana II, Arange = [-2,6]
        # for rest models, Arange = [-7.5,1] 
        CES.events(Emin=1.5,
                Emax=15, 
                t_obs= 3186.105475562*24*3600, 
                A_range=[-2,6], 
                B_range=[-1,6], 
                N=50, 
                N_eig=20, 
                method="numpy")

        CES.plot(xlim=[10**-2,10**6],
                 ylim=[10**-1,10**6],
                title="AvsB_U(1)X_xH="+str(xH)+"_"+mod+"_xchi="+str(xchi),
                do_plot = False,
                plotsave=True,
                event_threshold=8.1)

        CES.plotmvsg(title="U(1)X_xH="+str(xH)+"_"+mod+"_xchi="+str(xchi), plotsave=True)