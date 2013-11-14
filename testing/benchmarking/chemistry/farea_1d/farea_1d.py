# plots cation concentration along x at last time step 
# benchmark: compares to pflotran simulation results
# author: S.Molins - Oct. 2013

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# ----------- AMANZI + ALQUIMIA -----------------------------------------------------------------

def GetXY_Amanzi(path,root,time,comp):

    # open amanzi concentration and mesh files
    dataname = os.path.join(path,root+"_data.h5")
    amanzi_file = h5py.File(dataname,'r')
    meshname = os.path.join(path,root+"_mesh.h5")
    amanzi_mesh = h5py.File(meshname,'r')

    # extract cell coordinates
    y = np.array(amanzi_mesh['0']['Mesh']["Nodes"][0:len(amanzi_mesh['0']['Mesh']["Nodes"])/4,0])
    # y = np.array(amanzi_mesh['Mesh']["Nodes"][0:len(amanzi_mesh['Mesh']["Nodes"])/4,0]) # old style

    # center of cell
    x_amanzi_alquimia  = np.diff(y)/2+y[0:-1]

    # extract concentration array
    c_amanzi_alquimia = np.array(amanzi_file[comp][time]).flatten()
    amanzi_file.close()
    amanzi_mesh.close()
    
    return (x_amanzi_alquimia, c_amanzi_alquimia)

# ----------- PFLOTRAN STANDALONE ------------------------------------------------------------

def GetXY_PFloTran(path,root,time,comp):

    # read pflotran data
    filename = os.path.join(path,"ascem-2012-1d-"+root+".h5")
    pfdata = h5py.File(filename,'r')

    # extract coordinates
    y = np.array(pfdata['Coordinates']['X [m]'])
    x_pflotran = np.diff(y)/2+y[0:-1]

    # extract concentrations
    c_pflotran = np.array(pfdata[time][comp]).flatten()
#    c_pflotran = c_pflotran.flatten()
    pfdata.close()

    return (x_pflotran, c_pflotran)

if __name__ == "__main__":

    import os
    import run_amanzi_chem
    import numpy as np

    # root name for problem
    root = "farea-full"

    # components and minerals
    components = ['H+', 'Al+++', 'Ca++', 'Cl-', 'Fe+++', 'CO2(aq)', 'K+', 'Mg++', 'Na+', 'SiO2(aq)', 'SO4--', 'Tritium', 'NO3-', 'UO2++']
    minerals =['Quartz', 'Goethite', 'Kaolinite', 'Schoepite', 'Gibbsite', 'Jurbanite', 'Basaluminite', 'Opal']

    # amanzi output
    amanzi_totc_templ = "total_component_concentration.cell.Component {0} conc"
    amanzi_totc = [amanzi_totc_templ.format(x) for x in range(len(components))]

    amanzi_sorb_templ = "total_sorbed.cell.{0}"
    amanzi_sorb = [amanzi_sorb_templ.format(x) for x in range(len(components))]

    amanzi_vf_templ = "mineral_volume_fractions.cell.{0} vol frac"
    amanzi_vf = [amanzi_vf_templ.format(x) for x in minerals]

    # pflotran output
    pflotran_totc_templ = "Total_{0} [M]"
    pflotran_totc = [pflotran_totc_templ.format(x) for x in components]

    pflotran_sorb_templ = "Total_Sorbed_{0} [mol_m^3]"
    pflotran_sorb = [pflotran_sorb_templ.format(x) for x in components]

    pflotran_vf_templ = "{0}_VF"
    pflotran_vf = [pflotran_vf_templ.format(x) for x in minerals]

    # hardwired time / add or remove here
    timespflo = ['Time:  5.00000E+01 y']
    timesama  = ['71']
    
    # hardwired selected components / add or remove here
    search = ['Ca++','Mg++','K+','Cl-'] # ['UO2++'] # ['Ca++','Mg++','K+','Cl-'] # ['Ca++', 'Mg++','SiO2(aq)']
    index = [components.index(comp) for comp in search]

    # hardwired selected minerals / add or remove here
    searchm = ['Kaolinite'] #, 'Goethite', 'Kaolinite', 'Schoepite'] # ['Goethite'] #['Goethite', 'Kaolinite', 'Gibbsite']
    indexm = [minerals.index(min) for min in searchm] 

    # pflotran selected output
    totcpflo = [pflotran_totc[i] for i in index]
    sorbpflo = [pflotran_sorb[i] for i in index]
    vfpflo   = [pflotran_vf[i] for i in indexm]

    # amanzi selected output
    totcama = [amanzi_totc[i] for i in index]
    sorbama = [amanzi_sorb[i] for i in index]
    vfama   = [amanzi_vf[i] for i in indexm]

    # start with pflotran results
    path_to_pflotran = "pflotran"

    # tot concentration
    u_pflotran = [[[] for x in range(len(totcpflo))] for x in range(len(timespflo))]
    for i, time in enumerate(timespflo):
       for j, comp in enumerate(totcpflo):          
          x_pflotran, c_pflotran = GetXY_PFloTran(path_to_pflotran,root,time,comp)
          u_pflotran[i][j] = c_pflotran

    # sorbed concentration    
    v_pflotran = [[[] for x in range(len(sorbpflo))] for x in range(len(timespflo))]
    for i, time in enumerate(timespflo):
       for j, sorb in enumerate(sorbpflo):
          x_pflotran, c_pflotran = GetXY_PFloTran(path_to_pflotran,root,time,sorb)
          v_pflotran[i][j] = c_pflotran

    # mineral volume fraction
    w_pflotran = [[[] for x in range(len(vfpflo))] for x in range(len(timespflo))]
    for i, time in enumerate(timespflo):
       for j, vf in enumerate(vfpflo):
          x_pflotran, c_pflotran = GetXY_PFloTran(path_to_pflotran,root,time,vf)
          w_pflotran[i][j] = c_pflotran

    CWD = os.getcwd()
    local_path = "" 
    
    try:
        # Amanzi native chemistry
        input_filename = os.path.join("amanzi-u-1d-"+root+".xml")
        path_to_amanzi = "amanzi-native-output"
        run_amanzi_chem.run_amanzi_chem("../"+input_filename,run_path=path_to_amanzi,chemfiles=["ascem-2012-"+root+".bgd"])

        # tot conc
        u_amanzi_native = [[[] for x in range(len(totcama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, comp in enumerate(totcama):
              x_amanzi_native, c_amanzi_native = GetXY_Amanzi(path_to_amanzi,'farea-1d',time,comp)
              u_amanzi_native[i][j] = c_amanzi_native

        # sorb conc
        v_amanzi_native = [[[] for x in range(len(sorbama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, sorb in enumerate(sorbama):
              x_amanzi_native, c_amanzi_native = GetXY_Amanzi(path_to_amanzi,'farea-1d',time,sorb)
              v_amanzi_native[i][j] = c_amanzi_native

        # mineral volume fraction
        w_amanzi_native = [[[] for x in range(len(vfama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, vf in enumerate(vfama):
              x_amanzi_native, c_amanzi_native = GetXY_Amanzi(path_to_amanzi,'farea-1d',time,vf)
              w_amanzi_native[i][j] = c_amanzi_native

    except:

        pass


    try:  
        # Amanzi-Alquimia
        input_filename = os.path.join("amanzi-u-1d-"+root+"-alq.xml")
        path_to_amanzi = "amanzi-alquimia-output"
        run_amanzi_chem.run_amanzi_chem("../"+input_filename,run_path=path_to_amanzi,chemfiles=["ascem-2012-1d-"+root+".in","ascem-2012-"+root+".dat"])

        # tot concentration
        u_amanzi_alquimia = [[[] for x in range(len(totcama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, comp in enumerate(totcama):
              x_amanzi_alquimia, c_amanzi_alquimia = GetXY_Amanzi(path_to_amanzi,"farea-1d",time,comp)
              u_amanzi_alquimia[i][j] = c_amanzi_alquimia  

        # sorbed concentration
        v_amanzi_alquimia = [[[] for x in range(len(sorbama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, sorb in enumerate(sorbama):
              x_amanzi_alquimia, c_amanzi_alquimia = GetXY_Amanzi(path_to_amanzi,"farea-1d",time,sorb)
              v_amanzi_alquimia[i][j] = c_amanzi_alquimia

        # mineral volume fraction
        w_amanzi_alquimia = [[[] for x in range(len(vfama))] for x in range(len(timesama))]
        for i, time in enumerate(timesama):
           for j, vf in enumerate(vfama):
              x_amanzi_alquimia, c_amanzi_alquimia = GetXY_Amanzi(path_to_amanzi,"farea-1d",time,vf)
              w_amanzi_alquimia[i][j] = c_amanzi_alquimia

        alq = True

    except:

        alq = False

    # initialize subplots
    fig, ax = plt.subplots(3,sharex=True,figsize=(8,15))
#    bx =[None,]*3
#    bx[0] = ax[0].twinx()
#    bx[2] = ax[2].twinx()

    colors= ['r','b','m','g'] # components
    colors2= ['c','k','g','y'] # components
    styles = ['-','--','x'] # codes
    codes = ['Amanzi+Alquimia(PFloTran)','Amanzi Native Chemistry','PFloTran'] + [None,]*9

    # lines on axes
    # ax[0],b[0] ---> Aqueous concentrations
    # ax[1],b[1] ---> Sorbed concentrations

    # for i, time in enumerate(times):
    i = 0
    for j, comp in enumerate(search):
 
            if alq:
                   ax[0].plot(x_amanzi_alquimia, u_amanzi_alquimia[i][j],color=colors[j],linestyle=styles[0],linewidth=2)
            ax[0].plot(x_amanzi_native, u_amanzi_native[i][j],color=colors[j],linestyle=styles[1],linewidth=2,label=comp)
            ax[0].plot(x_pflotran, u_pflotran[i][j],color=colors[j],linestyle='None',marker=styles[2],linewidth=2)
 
            if alq:
                   ax[1].plot(x_amanzi_alquimia, v_amanzi_alquimia[i][j],color=colors[j],linestyle=styles[0],linewidth=2,label=codes[j*len(styles)])
            ax[1].plot(x_amanzi_native, v_amanzi_native[i][j],color=colors[j],linestyle=styles[1],linewidth=2,label=codes[j*len(styles)+1]) #label=comp)
            ax[1].plot(x_pflotran, v_pflotran[i][j],color=colors[j],linestyle='None',marker=styles[2],linewidth=2,label=codes[j*len(styles)+2])

    # ax[2],b[2] ---> Mineral Volume Fractions

    # for i, time in enumerate(times):
    i = 0
    for j, vf in enumerate(searchm):

            if alq:
                   ax[2].plot(x_amanzi_alquimia, w_amanzi_alquimia[i][j],color=colors2[j],linestyle=styles[0],linewidth=2) #label=codes[j*len(styles)])
            ax[2].plot(x_amanzi_native, w_amanzi_native[i][j],color=colors2[j],linestyle=styles[1],linewidth=2,label=vf) #label=codes[j*len(styles)+1])
            ax[2].plot(x_pflotran, w_pflotran[i][j],color=colors2[j],linestyle='None',marker=styles[2],linewidth=2) #label=codes[j*len(styles)+2])

    # axes
    ax[2].set_xlabel("Distance (m)",fontsize=15)

    ax[0].set_ylabel("Total Concentration [mol/L]",fontsize=15)
    ax[1].set_ylabel("Total Sorbed Concent. [mol/m3]",fontsize=15)
    ax[2].set_ylabel("Mineral Volume Fraction [m3/m3]",fontsize=15)

    # plot adjustments
    plt.subplots_adjust(left=0.10,bottom=0.05,right=0.90,top=0.90)
    ax[0].legend(loc='center right',fontsize=15)
    ax[1].legend(loc='center right',fontsize=15)
    ax[2].legend(loc='center right',fontsize=15)

#    ax[2].set_ylim(bottom=-0.01)
#    bx[2].set_ylim(bottom=-2.0e-6)

    plt.suptitle("Amanzi 1D "+root.title()+" Benchmark at 50 years",x=0.57,fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    #pyplot.show()
    #plt.savefig(root+"_1d.png",format="png")
    #plt.close()

    #finally:
    #    pass 
