# plots tracer concentration along x at last time step 
# benchmark: compares to pflotran simulation results
# author: S.Molins - Sept. 2013

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
    c_amanzi_alquimia = np.array(amanzi_file[comp][time])
    amanzi_file.close()
    amanzi_mesh.close()
    
    return (x_amanzi_alquimia, c_amanzi_alquimia)

def GetXY_AmanziS(path,root,time,comp):
    try:
        import fsnapshot
        fsnok = True
    except:
        fsnok = False

    plotfile = os.path.join(path,root)
    if os.path.isdir(plotfile) & fsnok:
        (nx, ny, nz) = fsnapshot.fplotfile_get_size(plotfile)
        x = np.zeros( (nx), dtype=np.float64)
        y = np.zeros( (nx), dtype=np.float64)
        (y, x, npts, err) = fsnapshot.fplotfile_get_data_1d(plotfile, comp, y, x)
    else:
        x = np.zeros( (0), dtype=np.float64)
        y = np.zeros( (0), dtype=np.float64)
    
    return (x, y)

# ----------- PFLOTRAN STANDALONE ------------------------------------------------------------

def GetXY_PFloTran(path,root,time,comp):

    # read pflotran data
    filename = os.path.join(path,"1d-"+root+".h5")
    pfdata = h5py.File(filename,'r')

    # extract coordinates
    y = np.array(pfdata['Coordinates']['X [m]'])
    x_pflotran = np.diff(y)/2+y[0:-1]

    # extract concentrations
    c_pflotran = np.array(pfdata[time][comp])
    c_pflotran = c_pflotran.flatten()
    pfdata.close()

    return (x_pflotran, c_pflotran)

# ------------- CRUNCHFLOW ------------------------------------------------------------------
def GetXY_CrunchFlow(path,root,cf_file,comp,ignore):

    # read CrunchFlow data
    filename = os.path.join(path,cf_file)
    f = open(filename,'r')
    lines = f.readlines()
    f.close()

    # ignore couple of lines
    for i in range(ignore):
      lines.pop(0)

    # extract data x0, x1, ..., xN-1 per line, keep only two columns
    xv=[]
    yv=[] 
    for line in lines:
      xv = xv + [float(line.split()[0])]
      yv = yv + [float(line.split()[comp+1])]
    
    xv = np.array(xv)
    yv = np.array(yv)

    return (xv, yv)

if __name__ == "__main__":

    import os
    import run_amanzi_chem
    import numpy as np

    # root name for problem
    root = "tracer"

    # pflotran
    path_to_pflotran = "pflotran"

     # hardwired for 1d-calcite: time and comp
    time = 'Time:  5.00000E+01 y'
    comp = 'Total_'+root.title()+' [M]'

    x_pflotran, c_pflotran = GetXY_PFloTran(path_to_pflotran,root,time,comp)    
    
    # CrunchFlow: hardwired for calcite_1d_CF.in: time and comp
    times_CF = 'totcon5.out'
    comp = 0
    ignore = 4

    # crunchflow GIMRT
    path_to_crunchflow = "crunchflow/gimrt"
    x_crunchflow, c_crunchflow = GetXY_CrunchFlow(path_to_crunchflow,root,times_CF,comp,ignore)

    # crunchflow OS3D
    path_to_crunchflow = "crunchflow/os3d"
    x_crunchOS3D, c_crunchOS3D = GetXY_CrunchFlow(path_to_crunchflow,root,times_CF,comp,ignore)
    
    CWD = os.getcwd()
    local_path = "" 

# run+read Amanzi native chemistry simulation
    try:
        time = '71'
        comp = 'total_component_concentration.cell.Tracer' # conc'

        # Amanzi native chemistry
        input_filename = os.path.join("amanzi-u-1d-"+root+".xml")
        path_to_amanzi = "amanzi-output"
        run_amanzi_chem.run_amanzi_chem("../"+input_filename,run_path=path_to_amanzi)

        x_amanzi_native, c_amanzi_native = GetXY_Amanzi(path_to_amanzi,root,time,comp)
        native = len(x_amanzi_native)

    except:
        native = len(x_amanzi_native)

# run+read Amanzi-Alquimia-PFloTran simulation
    try:
        comp = 'total_component_concentration.cell.Tracer conc'
        input_filename = os.path.join("amanzi-u-1d-"+root+"-alq-pflo.xml")
        path_to_amanzi = "amanzi-alquimia-output"
        run_amanzi_chem.run_amanzi_chem("../"+input_filename,run_path=path_to_amanzi,chemfiles=["1d-"+root+".in",root+".dat"])
        x_amanzi_alquimia, c_amanzi_alquimia = GetXY_Amanzi(path_to_amanzi,root,time,comp)
        alq = len(x_amanzi_alquimia)

    except:
        alq = 0

# run+read Amanzi-Alquimia-CrunchFlow simulation
    try:
        comp = 'total_component_concentration.cell.Tracer conc'
        input_filename = os.path.join("amanzi-u-1d-"+root+"-alq-crunch.xml")
        path_to_amanzi = "amanzi-alquimia-crunch-output"
        run_amanzi_chem.run_amanzi_chem("../"+input_filename,run_path=path_to_amanzi,chemfiles=["1d-"+root+"-crunch.in",root+".dbs"])
        x_amanzi_alquimia_crunch, c_amanzi_alquimia_crunch = GetXY_Amanzi(path_to_amanzi,root,time,comp)
        alq_crunch = len(x_amanzi_alquimia_crunch)

    except:
        alq_crunch = 0


    # amanziS data
    
    # +pflotran
    try:
        input_filename = os.path.join("amanzi-s-1d-tracer-alq-pflo.xml")
        path_to_amanziS = "struct_amanzi-output-pflo"
        run_amanzi_chem.run_amanzi_chem(input_filename,run_path=path_to_amanziS,chemfiles=None)
        root_amanziS = "plt00501"
        compS = "Tracer_Aqueous_Concentration"
        x_amanziS, c_amanziS = GetXY_AmanziS(path_to_amanziS,root_amanziS,time,compS)
        struct = len(x_amanziS)
    except:
        struct = 0

    # +crunchflow
    try:
        # import pdb; pdb.set_trace()
        input_filename = os.path.join("amanzi-s-1d-tracer-alq-crunch.xml")
        path_to_amanziS = "struct_amanzi-output-crunch"
        run_amanzi_chem.run_amanzi_chem(input_filename,run_path=path_to_amanziS,chemfiles=None)
        root_amanziS = "plt00501"
        compS = "Tracer_Aqueous_Concentration"
        x_amanziS_crunch, c_amanziS_crunch = GetXY_AmanziS(path_to_amanziS,root_amanziS,time,compS)
        struct_c = len(x_amanziS_crunch)
    except:
        struct_c = 0

# plotting --------------------------------------------------------

# subplots
    fig, ax = plt.subplots()

# pflotran
    ax.plot(x_pflotran, c_pflotran,'m-',label='PFloTran',linewidth=2)

# crunchflow
    ax.plot(x_crunchflow, c_crunchflow,'m--',label='CrunchFlow GIMRT',linewidth=2)
    ax.plot(x_crunchOS3D, c_crunchOS3D,'m*',label='CrunchFlow OS3D',linewidth=2) 

# unstruct amanzi native chem
    if native>0:
        ax.plot(x_amanzi_native, c_amanzi_native,'rx',label='Amanzi',linewidth=2)

# unstruct amanzi alquimia + pflotran
    if alq>0:
        ax.plot(x_amanzi_alquimia, c_amanzi_alquimia,'r-',label='Amanzi-Alq(PFT)',linewidth=2)

# unstruct amanzi alquimia + pflotran
    if alq_crunch>0:
        ax.plot(x_amanzi_alquimia_crunch, c_amanzi_alquimia_crunch,'r*',label='Amanzi-Alq(CF)',linewidth=2)

# struct amanzi alquimia + pflotran
    if (struct>0):
        sam = ax.plot(x_amanziS, c_amanziS,'g-',label='AmanziS+Alq(PFT)',linewidth=2)     

# struct amanzi alquimia + crunchflow
    if (struct_c>0):
        samc = ax.plot(x_amanziS_crunch, c_amanziS_crunch,'g*',label='AmanziS+Alq(CF)',linewidth=2)     

# figure look
    # axes
    ax.set_xlabel("Distance (m)",fontsize=20)
    ax.set_ylabel("Total "+root.title()+" concentration [mol/L]",fontsize=20)

    # plot adjustments
    plt.subplots_adjust(left=0.20,bottom=0.15,right=0.95,top=0.90)
    plt.legend(loc='upper right',fontsize=13)
    plt.suptitle("Amanzi 1D "+root.title()+" Benchmark at 50 years",x=0.57,fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    #pyplot.show()
    plt.savefig(root+"_1d.png",format="png")
#    plt.close()

#    finally:
#        pass 
