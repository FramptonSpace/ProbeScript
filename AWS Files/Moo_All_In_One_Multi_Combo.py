#####TODO: Do everything below for the science and the relay probe, make it so number of relay probes can change...
#####TODO: Compare max comm depth with position list, add to max depth for loop

###### LOAD TOOLS AND PREAMBLE ########
import os
import sys
import datetime

import numpy as np
import math

from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import *
from scipy import integrate



import pymoo
from pymoo.problems.functional import FunctionalProblem
from pymoo.visualization.scatter import Scatter
import pymoo.core.result
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.util import default_number_to_text
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.population import Population

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization


import yaml
import time
import json

def preamble():
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(''))
    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(parent_dir)

    # Now you can import modules from the parent directory
preamble()

#returns the value of log(a+b), makes adding logs easier
def logsum(a,b):
    value = math.log(a)+math.log(1+(b/a))
    return(value)

def PlesaRead(filename):
    tempfile = 'InputFiles/thermochemical_{}_profiles_min_avg_max.txt'.format(filename)
    xcord = []
    Tmin = []
    Tavg = []
    Tmax = []
    with open(tempfile, 'r') as file:
        for i, line in enumerate(file):
            if i < 12:
                continue
            line = line.rstrip("\n")
            data = line.split('\t')
            xcord.append(float(data[0])*1000)
            Tmin.append(float(data[1]))
            Tavg.append(float(data[2]))
            Tmax.append(float(data[3]))
    
    Salinity = []
    tempfile = 'InputFiles/thermochemical_{}_profile.txt'.format(filename)
    with open(tempfile, 'r') as file:
        for i, line in enumerate(file):
            if i < 12:
                continue
            line = line.rstrip("\n")
            data = line.split('\t')
            Salinity.append(float(data[3])*10000)
            

    #Returns Salinity in ppm 
    return(xcord,Tmin,Tavg,Tmax,Salinity)
    
def Fileread(filename,depthlimit):
    
    #Gives directory of file to be opened, then opens it
    f = open('InputFiles/thermochemical_{}_profile.txt'.format(filename),'r')
    #[x:y] tells python to read from the x line to the y line, blank means "to the beginning/end" so here we're reading from line 15 to the end of the document
    lines = f.readlines()[12:]
    #create a bunch of empty lists that we can populate with our data
    xcord = []
    ycord = []
    Temp = []
    Rho = []
    SaltConc = []
    #Loop to add each new data point in the list
    for line in lines:
        #The way this text file is written, each new line is denoted by a "\n" character, the rstrip tells us to remove "\n" from every line
        line = line.rstrip("\n")
        #Each new data point is seperated by a tab, so we split each line into a new data point whenever python sees a tab
        data = line.split('\t')
        #We then add the first (0th) element to the first empty list, and then the 2nd to the 2nd etc...
        xcord.append(float(data[0]))
        Temp.append(float(data[1]))
        Rho.append(float(data[2]))
        SaltConc.append(float(data[3]))
        
    #At the end of the loop we then return the complete lists 
    for i in range(len(xcord)):
        xcord[i] = xcord[i] - 1520.7998175999999
        xcord[i] = xcord[i]*1000
    
    #print('xcord:', xcord)
    T_interp = interp1d(xcord,Temp) 
    xcord = range(0,(depthlimit+ 1))
    #print('xcord1: ',xcord)
    #print('x:', xcord, 'y:', ycord,'Temp:', Temp,'Density:',Rho, 'Salt:',SaltConc,'TempInterp',T_interp)
    return(xcord,ycord,Temp,Rho,SaltConc,T_interp)

def Fileread2D(filename):
    
    #Gives directory of file to be opened, then opens it
    f = open('/home/ec2-user/AWS Files/AWS Files/InputFiles/thermochemical_{}_data.txt'.format(filename),'r')
    #[x:y] tells python to read from the x line to the y line, blank means "to the beginning/end" so here we're reading from line 15 to the end of the document
    lines = f.readlines()[15:]
    #create a bunch of empty lists that we can populate with our data
    xcord = []
    ycord = []
    Temp = []
    Rho = []
    SaltConc = []
    #Loop to add each new data point in the list
    for line in lines:
        #The way this text file is written, each new line is denoted by a "\n" character, the rstrip tells us to remove "\n" from every line
        line = line.rstrip("\n")
        #Each new data point is seperated by a tab, so we split each line into a new data point whenever python sees a tab
        data = line.split('\t')
        #We then add the first (0th) element to the first empty list, and then the 2nd to the 2nd etc...
        xcord.append(float(data[0]))
        ycord.append(float(data[1]))
        Temp.append(float(data[3]))
        Rho.append(float(data[4]))
        SaltConc.append(float(data[5]))
        
    #At the end of the loop we then return the complete lists 
    #for i in range(len(xcord)):
    #    xcord[i] = xcord[i] - 1520.7998175999999
    #    xcord[i] = xcord[i]*1000
    
    #print('xcord:', xcord)
    #T_interp = CubicSpline(xcord,Temp) 
    #xcord = range(0,(depthlimit+ 1))
    #print('xcord1: ',xcord)
    #print('x:', xcord, 'y:', ycord,'Temp:', Temp,'Density:',Rho, 'Salt:',SaltConc,'TempInterp',T_interp)
    return(xcord,ycord,Temp,Rho,SaltConc)

def BryantRead(filename):
    filename = 'InputFiles/Bryant{}.txt'.format(filename)
    xcord = []
    Temp = []
    #for line in lines:
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                # Skip the first line
                continue
            line = line.rstrip("\n")
            data = line.split('\t')
            xcord.append(float(data[0]))
            Temp.append(float(data[1]))
    

    return(xcord,Temp)

#Interpolate the plesa data using a cubicspline model
def PlesaInterp(filename):
    xcord, tmin, tavg, tmax, Salinity = PlesaRead(filename)
    tmininterp = CubicSpline(xcord,tmin)
    tavginterp = CubicSpline(xcord,tavg)
    tmaxinterp = CubicSpline(xcord,tmax)
    SalInterp = CubicSpline(xcord,Salinity)
    return(xcord, tmininterp,tavginterp,tmaxinterp, SalInterp)

def BryantInterp(filename):
    filename = 'InputFiles/Bryant{}.txt'.format(filename)
    xcord = []
    Temp = []
    #for line in lines:
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                # Skip the first line
                continue
            else:
                line = line.rstrip("\n")
                data = line.split('\t')
                xcord.append(float(data[0]))
                Temp.append(float(data[1]))
    
        xcord.reverse()
        Temp.reverse()
    tavginterp = CubicSpline(xcord,Temp)
    return(xcord,tavginterp)

#loads the pressure from the calculated list, based on pressure equation
def PressureLoad():
    pressurefilepath = 'InputFiles/PressureTable.csv'
    MeltingFilePath = 'InputFiles/MeltTempTable.csv'
    Pressuredata = []
    MeltingPointdata = []

    Pressuredata = np.genfromtxt(pressurefilepath, delimiter=',')
    x_value = Pressuredata[1:,0]
    drho_5_Mpa = Pressuredata[1:,2]
    drho_11_Mpa= Pressuredata[1:,3]
    drho_23_Mpa= Pressuredata[1:,4]
    drho_46_Mpa= Pressuredata[1:,5]

    MeltingPointdata = np.genfromtxt(MeltingFilePath, delimiter=',')
    P_MPa = MeltingPointdata[:,0]
    Temp = MeltingPointdata[:,1]

    drho_5_Pressure = CubicSpline(x_value,drho_5_Mpa)
    drho_11_Pressure = CubicSpline(x_value,drho_11_Mpa)
    drho_23_Pressure = CubicSpline(x_value,drho_23_Mpa)
    drho_46_Pressure = CubicSpline(x_value,drho_46_Mpa)
    T_Melt = CubicSpline(P_MPa,Temp)

    return(drho_5_Pressure,drho_11_Pressure,drho_23_Pressure,drho_46_Pressure,T_Melt)
    
#TODO: Change formatting of this section? I think that importing things like this is bad
#TODO: Write code to measure attenuation of signal through the surface, start by defining what the surface signal is and at what power.


#Loads all the data needed for a link budget from a .json file, example format is seen in LinkInput.json
def CommsLoad(filename):
    with open('Configs/{}.json'.format(filename),'r') as file:
        data = json.load(file)
        LinkData =[]
        for key, value_dict in data.items():
            # Check if the value is a dictionary and contains a "value" key
            if isinstance(value_dict, dict) and "value" in value_dict:
                # Extract and print the "value" entry
                value_entry = value_dict["value"]
                LinkData.append(value_entry)
            else:
                print(f"Invalid entry for {key}")

    return(LinkData)


def GaindBCalc(effic,wavelength_m,Area_m2):
    #efficiency between 0-1
    #Aread of appature
    
    Gain_dB = 10*math.log((4*(math.pi)*effic*Area_m2)/(wavelength_m**2),10)
    return(Gain_dB)
#Most Basic Link Budget, takes all the variables of the .json file and returns output power. Verified with online tool
def PowerReceived(Power_Tx_dBm ,G_Tx_dBi ,L_Tx_dB ,PathLoss ,G_Rx_dBi ,L_Rx_dB):
    P_received = Power_Tx_dBm + G_Tx_dBi + L_Tx_dB -PathLoss + G_Rx_dBi + L_Rx_dB
    return(P_received)

#Returns boolean of if positive margin has been reached or not
def PositiveLinkCheck(filename,P_target):
    if PowerReceived(filename) >= P_target:
        return(True)
    else:
        return(False)
    
#For BryantValidation, multiploes rather than using dB  
def SNRCalc(P_tx,eff_tx,G_tx,Loss,effic_rx,G_rx,k_boltz,T_sys,BW):
    #print('Loss: ', Loss)
    SNR_factor = (P_tx*eff_tx*G_tx*effic_rx*G_rx)/(k_boltz*T_sys*BW)
    
    SNR = SNR_factor*Loss
    return(SNR)

#Free space path loss in dB
def FreeLoss(Freq,x_range,i):
    a =  20 * math.log10(x_range)
    b = 20 * math.log10(Freq)
    c = 20 * math.log10(3e8)
    d= 20*math.log10(4 * math.pi)
    if x_range == 0:
        fspl_dB = 0
    elif i ==1:
        fspl_dB = d-a-b-c
    elif i ==0:
        fspl_dB = d-c-b
    
    return(fspl_dB)

#For Bryant Validation Case, returns ratio, NOT Db
def FreeLossBryant(Freq,x_range):
    #Use speed of light in ICE, this correction factor validates Bryant
    fsl_factor = (3e8/(math.pi*Freq*x_range*4))**2
    return(fsl_factor)
#calculates the conductivity and emisivity of a medium, equation taken from Bryant 02
def CondAndEmiss(Temp,i):
#Temp in K
# i 0 if pure water
# i 1 for 13 ppm NaCl
# Only valid for region around 1Ghz
    Temp_c = Temp-273.15

    #dE = 3.1884+(0.0091*(Temp_c))
    dE = 3.1884+(0.00091*(Temp_c))
    if i ==0:
        #ddE = 10**(-3.0129+0.0123*(Temp_c))
        ddE = 10**(-3.0129+(0.0123*(Temp_c)))
        #print('ddE = ', ddE)
    else:
        #ddE = 10**(-2.398+(0.0299*(Temp_c)))
        ddE = 10**(-2.398+(0.0299*(Temp_c)))
    return(dE,ddE)

#Adjust the con and emiss based on the salinity of the ice at your location in ppm
def ConAndEmissSal(Temp,Salinity_ppm):
    #print('Temp:', Temp)
    ###limit for temperature if outside temp range

    Temp_c = Temp-273.15
    #dE = 3.1884+(0.0091*(Temp_c))
    dE = 3.1884+(0.00091*(Temp_c))
    #print('dE: ',dE)
    dde_pure = 10**(-3.0129+(0.0123*(Temp_c)))
    dde_13ppm = 10**(-2.398+(0.0299*(Temp_c)))
    Delta_dde = dde_13ppm - dde_pure
    #print('Dela dde: ', Delta_dde)
    #print('Salinity: ', Salinity_ppm)
    ddE = abs(Delta_dde)*Salinity_ppm
    #print('ddE: ', ddE)
    return(dE,ddE)

#Calculate attenuation length from con and emiss: https://iopscience.iop.org/article/10.1088/1742-6596/81/1/012003/pdf
def AttenuationLength(Temp,Salinity,freq):
    Salinity *= 10000
    dE,ddE = ConAndEmissSal(Temp,Salinity)
    
    tandelta = ddE/dE
    L = (3*10**8)/(np.pi*np.sqrt(dE)*tandelta*freq)
    return(L)

#Take a distance and factors of the medium to calculate added path loss (N.B. Free space loss needs to be added to this)
def PathLossFactor(Temp,Freq,x_range,i):
#Freq in Hz
#x_range in m
#i as in CondandEmiss index
    dE, ddE = CondAndEmiss(Temp,i)
    lamb = ((3e8)/(Freq))
    P_ratio = math.exp(-2*x_range*((2*math.pi*ddE)/(lamb*dE)))
    PathLossdB = 10 * math.log10(P_ratio)
    return(PathLossdB)

#For Bryant Validation, returns ratio, NOT dB
def PathLossFactorBryant(Temp,Freq,x_range,c,salinity_ppm):
#Freq in Hz
#x_range in m
#i as in CondandEmiss index
    dE, ddE = ConAndEmissSal(Temp,salinity_ppm)
    if salinity_ppm >= 55000:
        #Adjustment for high salinity where salinity dependence of ddE begins to break down. Future work.
        ddE *= 0.5
    #print('ddE2: ', ddE)
    #Use speed of light in ICE, this correction factor validates Bryant
    #with correct boltzman and same correction in fspl
    #i = 1 c = 0.9e8, 0 1.8e8. 
    #with incorrect boltzman (e20):
    #i = 1 c = , 0 2.7e8


    #with correct boltzman and correction ONLY here:
    #brittle
    #i = 1 c=0.8 i=0 c= 1.5e8 (speed of light in water!)
    #for convecting:
    #0.67 below is exact
    #i 1 c= 0.67 i=0 c = 1.15-1.11
    #So there's a correction factor that is dependent on temperature profile...

    #if i == 1:
    #    c = 0.67e8
    #if i == 0:
    #    c = 1.13e8
    #print(dE)
    #print(ddE)
    lamb = ((c)/(Freq))
    P_ratio = math.exp(-2*float(x_range)*((2*math.pi*ddE)/(lamb*dE)))
    #print('P_ratio: ', P_ratio)
    #PathLossRatio = 10 * math.log10(P_ratio)
    return(P_ratio)

#Takes the profile filename and two points in the ice, and returns the total loss from free space and material
def LossThroughIce(filename,x_0,x_1,Freq,i):
    #x_0 and x_1 need to be in km, and are converted here.
    xcord, tmin, tavg, tmax = PlesaInterp(filename)
    t_min_mean = (tmin(x_0)+tmin(x_1))/2
    t_avg_mean = (tavg(x_0)-tavg(x_1))/2
    t_max_mean = (tmax(x_0)-tmax(x_1))/2
    x_range = abs(x_0-x_1)*1000
    fspl= FreeLoss(Freq,x_range,1)
    ice_loss_min = PathLossFactor(t_min_mean,Freq,x_range,i) + fspl
    ice_loss_avg = PathLossFactor(t_avg_mean,Freq,x_range,i) + fspl
    ice_loss_max = PathLossFactor(t_max_mean,Freq,x_range,i) + fspl
    return(ice_loss_min,ice_loss_avg,ice_loss_max) 

def LossThroughIceBryant(tavg,x_probe,x_current,Freq,c,salinity_interp):
    #xcord,tavg,= BryantInterp(filename)

    x_range = abs(x_current-x_probe)
    #########TESTBED################
    
    #Calculate the range of values
    r_list = np.linspace(x_current, x_probe, int(x_range / 10))

    #Calculate tavg for each value in r_list
    temp_list = tavg(r_list / 1000)
    sal_list = salinity_interp(r_list / 1000)
    #Calculate the mean of temp_list
    t_avg_mean = np.mean(temp_list)
    salinity_ppm = np.mean(sal_list)

    ####################

    #t_avg_mean = (tavg(x_probe/1000)+tavg(x_current/1000))/2
    #print('t_avg_mean: ', t_avg_mean)
    
    #print('x_range: ', x_range)
    fspl = (c/(math.pi*Freq*x_range*4))**2

    ice_loss_avg = PathLossFactorBryant(t_avg_mean,Freq,x_range,c,salinity_ppm)
    ice_loss_avg *= fspl
    return(ice_loss_avg) 

def LossThroughIceBryant2(tavg,x_probe,x_current,Freq,c,salinity_interp):
    #xcord,tavg,= BryantInterp(filename)

    x_range = abs(x_current-x_probe)
    #########TESTBED################
    
    #Calculate the range of values
    r_list = np.linspace(x_probe, x_current, int(x_range / 10))
    #print(r_list)
    #Calculate tavg for each value in r_list
    temp_list = tavg(r_list / 1000)
    sal_list = salinity_interp(r_list / 1000)
    #Calculate the mean of temp_list
    t_avg_mean = np.mean(temp_list)
    salinity_ppm = np.mean(sal_list)
    #print(salinity_ppm)
    ####################

    #t_avg_mean = (tavg(x_probe/1000)+tavg(x_current/1000))/2
    #print('t_avg_mean: ', t_avg_mean)
    
    #print('x_range: ', x_range)
    fspl = (c/(math.pi*Freq*x_range*4))**2

    ice_loss_avg = PathLossFactorBryant(t_avg_mean,Freq,x_range,c,salinity_ppm)
    ice_loss_avg *= fspl
    return(ice_loss_avg) 

def CommAbove(tavg,x_probe,x_surface,Freq,c,salinity_interp,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,target_SNR):
    #xcord,tavg,= BryantInterp(filename)

    x_list = np.linspace(x_probe+100,x_surface,100)
    for x_current in x_list:
        x_range = abs(x_current-x_probe)
    
        #Calculate the range of values
        r_list = np.linspace(x_probe, x_current, int(x_range / 10))
        #print(r_list)
        #Calculate tavg for each value in r_list
        temp_list = tavg(r_list)
        sal_list = salinity_interp(r_list)
        #Calculate the mean of temp_list
        t_avg_mean = np.mean(temp_list)
        salinity_ppm = np.mean(sal_list)
    #print(salinity_ppm)
    ####################

    #t_avg_mean = (tavg(x_probe/1000)+tavg(x_current/1000))/2
    #print('t_avg_mean: ', t_avg_mean)
    
    #print('x_range: ', x_range)
        fspl = (c/(math.pi*Freq*x_range*4))**2

        ice_loss_avg = PathLossFactorBryant(t_avg_mean,Freq,x_range,c,salinity_ppm)
        ice_loss_avg *= fspl
    
        SNR = (P_tx_Sci*eff_tx_Sci*G_tx_Sci*eff_rx_Sci*G_rx_Sci)/(k_boltz*T_sys_Sci*BW)
    #print('SNR: ',SNR)
        SNR *= ice_loss_avg
        SNR_db = 10*math.log(SNR)
        if SNR_db < target_SNR:
            return(x_current)
        elif x_current >= x_surface:
            return(x_surface)
        
#R = data rate, B = bandwidth, returns ShannonLimit
def ShanLimit(R,B):
    Ri = R/(2*B)
    Shan = ((2**(2*Ri))-1)/(2*Ri)
    return(Shan)

#The distance at which a positive comms budget can no longer be maintained
def SurfaceToProbeLink(EuropaModel, config_file_path, depth):
    with open(config_file_path.format('ScienceProbe.json')) as ScienceProbeFile :
        ScienceProbeConfig= json.load(ScienceProbeFile)
    

    surface =1560.8
    pathlossmin,pathloss,pathlossmax = LossThroughIce(EuropaModel, 
                                                      surface, 
                                                      depth, 
                                                      ScienceProbeConfig["Communication Frequency"]["value"],
                                                        1)                                                                         
    return(pathlossmin,pathloss,pathlossmax)

#Given a max depth and an interval, run through the dB loss through the ice and return a list
# of losses at those depths                              
def SurfaceToProbeLossList(EuropaModel,config_file_path,max_depth,interval):
    values = np.arange(max_depth, 1560.8-interval, interval)

    minlosslist = []
    losslist= []
    maxlosslist = []
    for x in values:
       pathlossmin,pathloss,pathlossmax = SurfaceToProbeLink(EuropaModel, 
                                                             config_file_path,
                                                             x)
       minlosslist.append(pathlossmin)
       losslist.append(pathloss)
       maxlosslist.append(pathlossmax)

    return(values,losslist)

###Take the position of a probe, and do the link budget UPWARDS, return the distance that the SNR can be achieved
###Never return a value higher than the surface
def MaxCommsUpward():
    x=10
    return(x)


with open('AWS Files/AWS Files/Configs/EuropaPhysical.json') as EuropaConstantFile:
    Europa_constants = json.load(EuropaConstantFile)
    rho_w = Europa_constants["rho_w"]["value"]
    g_europa = Europa_constants["g_europa"]["value"]
    L = Europa_constants["L"]["value"]
    c_pw = Europa_constants["c_pw"]["value"]
    alpha_w = Europa_constants["k_w"]["value"]

profile_list = ['40km_Drho5_1D','40km_Drho11_1D','40km_Drho23_1D','40km_Drho46_1D']

def rho_i_calc(temp):
    rho_i = 933.31+(0.037978*temp)-((3.6274e-4)*temp**2)
    return(rho_i)


#This mu_w is NOT the ice temperature, it's the temperature of the water. Put the melt water temperature in here
#TODO: FIRST ESTIMATE, assume melt water is at 0 degrees.
def mu_w_calc(temp):
    mu_w = (1.4147*10**(-4))*((temp/226.8)-1)**(-1.5914)
    return(mu_w)

def c_pi_calc(temp):
    temp_ratio = temp/273.16
    a = temp_ratio**3
    b = (1.843*10**5) + (1.6357*10**8)*(temp_ratio**2) + (3.5519*10**9)*temp_ratio**6
    c = 1 + (1.667*10**2)*(temp_ratio**2) + (6.465*10**4)*(temp_ratio**4)+(1.6935*10**6)*(temp_ratio**8)
    
    c_pi = (a*b)/c
    return(c_pi)

#Calculates L*, the reduced latent heat of melting
def L_star_calc(Ice_temp,T_m,T_i):
    c_pi = c_pi_calc(Ice_temp)
    L_star = L + c_pi*(T_m-T_i)
    return(L_star)

#take the depth (e.g. 1560.8)and the drho profile and 
#return the temperature of melting
#calculates the melting point of ice at pressure by comparing
#the pressure at a given depth, with the melting point in table 
#melting table from 
#https://edisciplinas.usp.br/pluginfile.php/4557662/mod_resource/content/1/CRC%20Handbook%20of%20Chemistry%20and%20Physics%2095th%20Edition.pdf
#def MeltingPoint(filename,depth):
def T_melt_calc(depth,EuropaProfile):
    
    pressure_list =[drho_5_P,drho_11_P,drho_23_P,drho_46_P,T_Melt] = PressureLoad()
    pressure_profile = pressure_list[profile_list.index(EuropaProfile)]
    P_Mpa = pressure_profile(depth)
    Melt_Point = T_Melt(P_Mpa)
    Melt_Point_K = Melt_Point +273.15
    return(P_Mpa,Melt_Point_K)


#alpha w =
#rho_i = ice density
#rho_w = water density
#V = velocity
#R = probe radius
#mu_w = viscosity of melt water
#F_star = corrected contact force
#g = gravity acceleration 
#l = length of probe
#probe_mass in kg

def Efficiency_calc(ice_temp,rho_i,V_max,R,l,probe_mass,Tm):

    mu_w = mu_w_calc(Tm)
    F_star1 = (g_europa*probe_mass)
    F_star2 = (math.pi*R**2)*rho_w*g_europa*l

    F_star = F_star1-F_star2
    
    if F_star <= 0:
        F_star = 1e-20
    D1 = (1/(20*alpha_w))
    D2 = (((rho_i/rho_w)*V_max*R)**(4/3))
    D3 = (((3*np.pi*mu_w)/(2*F_star))**(1/3))
   
    #D =(1/(20*alpha_w))*(((rho_i/rho_w)*V*R)**(4/3))*(((3*np.pi*mu_w)/(2*F_star))**(1/3))
    D = D1*D2*D3
    gamma = (1-(3*D))/((7*D)+1)
    return(gamma)

#Qdot = Heat flow to probe head
#R = probe radius
#rho_i = ice density
#L = latent heat of melting
#c_pi = heat capacity of ice
#Ti = Ice temperature
#Tm = Ice Melting Temperature

def VelocityCurrent4(Probe_list,Ice_temp,rho_i,Tm):
    Qdot = Probe_list[0]
    R = Probe_list[1]
    l = Probe_list[2]

    mass = Probe_list[3]
    c_pi = c_pi_calc(Ice_temp)
    A = math.pi*R**2
    L_star = L+c_pi*(Tm-Ice_temp)
    q_dot = Qdot/A
    V_max = q_dot/(rho_i*L_star)
    Efficiency = Efficiency_calc(Ice_temp,rho_i,V_max,R,l,mass,Tm)
    ###On Test, efficiency of 0.6 seems to work... Check the inputs for the above efficiency calc...
    V_current = Efficiency*V_max*0.7
    return(V_current)

def VelocityCurrent3(Probe_list,Ice_temp,rho_i,Tm):
    Qdot = Probe_list[0]
    R = Probe_list[1]
    l = Probe_list[2]
    mass = Probe_list[3]
    rho_i = rho_i_calc(Ice_temp)
    c_pi = c_pi_calc(Ice_temp)
    A = math.pi*R**2
    V_max = (Qdot)/(A*rho_i*(L+c_pi*(Tm-Ice_temp)))
    Efficiency = Efficiency_calc(Ice_temp,rho_i,V_max,R,l,mass)
    V_current = Efficiency*V_max
    return(V_current)

def VelocityCurrent2(Qdot,R,l,mass,Ice_temp,rho_i,Tm):
    c_pi = c_pi_calc(Ice_temp)
    A = math.pi*R**2
    V_max = (Qdot)/(A*rho_i*(L+c_pi*(Tm-Ice_temp)))
    Efficiency = Efficiency_calc(Ice_temp,V_max,R,l,mass)
    V_current = Efficiency*V_max
    return(V_current)


def VelocityCurrent(Qdot, R,l, Ice_temp,Tm,):
    c_pi = c_pi_calc(Ice_temp)
    rho_i = rho_i_calc(Ice_temp)
    A=math.pi*R**2
    V_max = (Qdot)/(A*rho_i*(L+c_pi*(Tm-Ice_temp)))
    Efficiency = Efficiency_calc(Ice_temp,V_max,R,l)
    V_current = Efficiency*V_max
    return(V_current)

def phase_interface_Q(Q_h,gamma):
    Q_c = Q_h-(1-gamma)*Q_h
    return(Q_c)

#TODO, calculate refreezing length as described in this paper"
#https://pdf.sciencedirectassets.com/272593/1-s2.0-S0019103518X0013X/1-s2.0-S0019103518301568/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEN7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQD%2B5Cor556X8ho%2FqDXFQen4A9zmdpL3I8%2BZZ%2FrnZraqXwIhALkkBuzQ2GpoYK67yVxaSKmokynX6NFjFA12syRmS0fbKrMFCEcQBRoMMDU5MDAzNTQ2ODY1Igwb%2Bo3yJMT2AvuxmiAqkAVkFPYBH4fAVqjqTlSYHl8QfqScTlAr1GU6PJHMPb%2BI%2FsHWmw0UcaZozCELy7%2FJoRn4TaxTr1swQgp7PhAkVZ7sVcQtymp1KQQJ68G4F43S4Eh3CpnagT%2FD8UDl%2BFi5zWyYW64ciV6nUD8j%2FkJQ2d%2BzufYLXwKBGqNiW8zBMa5ugpWZpCpAYfC%2FJdGIhUi90Khmv2OBolPfiMONcWHyJd7LBErNsq8P0SV5tCqwFTTTrHyRls59%2BErMsjZJ4tjfoLntHnOtkad1uOdvs7UbOxCFImBir545oPUNOqcWmBpoNTcxqPRurQjy8inWklLkJdCOom7OO%2Fp42xawNESBcibbYoBJImbDFEqavRNZ%2BNa%2FhTzAubgX1v4WP7HruoR4evZkA0XBEkquiAg1LVonscHAAy3TjxWC%2Fulgaey0tkcWrXyVd9EAJf1BiWXrToc0WSxh6dGJMcpf9d4XpkrY%2FkY%2F1H4tkF7HHkhJlrDsvesNZdguITPTJvdjOt3d3seHYJopYcitLUDjJ5PRTD6%2FA6XuYk0I7idsrYnn8HE5wyXSZmib8Ngb5mPTtdMEFoUTp%2BSZSwYfB7aE8vwz9DX%2BC%2BPeqPmWgh64s33ICbIPT44VLNYGMFOWQQUYRv811STbsBbnL4uwoCmRJSup82aeiCbclKktMC4FktBUZ6czGqqXZEUSqEdQ1tGMQ%2BH5SJhHpbYaxP9HjTa%2FbUnaRPkMaDC2CaA9mgCVjhyhq3v6EzKjeqtkcWGPDqZN%2FS3o1XbOq8vC0y6GMFNF1lZPpIIycIer1D4RsbfIVM5iU%2BaoKIaYOjTg0lEpp5jkaY7PeUaniDrmUBlYJhK4SWQClncbYipa2VraIKC8AkCWMb0xTcS4BDCnjLKrBjqwAX9rz2IP%2FLtWDUzI6pfJW%2B8m6pDpmVgoRyrhbnlcrdbyZZ85UzScnVwEZpBbUR4uOR9DiiZKmDwwWHY2lne1EKhavjO1MAuOoiavLcxw7GS8t0SUHACRVorgt2%2FQAhsXhTVKIVaWnWJ9yZxulLGESTt1kgYhslUpV6j1G6kuudNI%2B2Xf8kwh%2FkC2bxgO1n1VXb1P3J2dzmwrGAQdyLYY8x%2B9G%2BrUFJRyI4kZvkFa4zMU&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231203T141610Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2HIAOE5M%2F20231203%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=11767592643608738cda08e83ff8689d8b30f748fc196e7be8522ec32beefd85&hash=0ad487b2240bd9db62ea1558c6c511e0074e2dfcc9d8756c15e1b4ed60b04578&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0019103518301568&tid=spdf-8e01a7de-4c55-4d3b-b670-23e516bc05b1&sid=8ecf3e28536723440449a99-95fe9463f83bgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1d065653050455035256&rr=82fc6d09ca6edceb&cc=gb
#Be careful, she uses L* as refreezing length :(
def RefreezeCalc(ice_temp,V,R,l,Q_h):
    #These next two values are approximations, and only true for 5 × 104 s/m2 < σ < 108 s/m2
    #and assume Ql = Qe
    n = 932
    d = 0.762
    gamma = Efficiency_calc(ice_temp,V,R,l)
    Q_c = phase_interface_Q(Q_h,gamma)

    #l_refreeze = (((Q_c*(V**(d-1))*R**(2*(d-1)))/n*(T_m-T_s))*((1-gamma)/gamma))**(1/d)

    #return(l_refreeze)
    return(gamma)

###Return a boolean, is the refreezing length less than the length of the probe?
def RefreezeLengthBool(ice_temp,V,R,l,Q_h,ice_melt_temp,gamma):
        #These next two values are approximations, and only true for 5 × 104 s/m2 < σ < 108 s/m2
    #and assume Ql = Qe
    n = 932
    d = 0.762
    Q_c = phase_interface_Q(Q_h,gamma)
    #check if 100 is right... is melt_temp in K?
    l_refreeze = ((((Q_c*(V**(d-1))*R**(2*(d-1)))/n*(ice_melt_temp-100))*((1-gamma)/gamma))**(1/d))/1000
    #print(l_refreeze)
    if l_refreeze <= l:
        return(True)
    else:
        return(False)
    
#calculates the maximum length to overcome buoyancy forces, limit on probe design
def BouyancyLengthLimit(m,R,rho_w):
    L_max = m/(math.pi*(R**2)*rho_w)
    return(L_max)

#Take a probe and a Europa Profile, return the velocity and acceleration interpolations
def VelocityProfile(ProbeFile,EuropaProfile):
    with open('Configs/Probe Profiles/{}.json'.format(ProbeFile)) as ProbeParameters:
        Params = json.load(ProbeParameters)
        R = Params["Probe Radius"]["value"]
        l = Params["Probe Length"]["value"]
        probe_mass = Params["Probe Mass"]["value"]
        Qdot = Params["Qdot"]["value"]
    
    A=math.pi*R**2
    xrange,Ice_t_min,Ice_t_avg,Ice_t_max = PlesaInterp(EuropaProfile)
    
    Europa_Range = np.arange(1520.8,1560.8,0.01)
    Velocity_Profile_min_t = [] 
    Velocity_Profile_avg_t = []
    Velocity_Profile_max_t = []
    for x in Europa_Range:
        c_pi_min = c_pi_calc(Ice_t_min(x))
        c_pi_avg = c_pi_calc(Ice_t_avg(x))
        c_pi_max = c_pi_calc(Ice_t_max(x))
        rho_i_min = rho_i_calc(Ice_t_min(x))
        rho_i_avg = rho_i_calc(Ice_t_avg(x))
        rho_i_max = rho_i_calc(Ice_t_max(x))
        #depth = 1560.8-x
        P_Ma, Tm_K = T_melt_calc(x,EuropaProfile)
        
   
        V_max_min_temp = (Qdot)/(A*rho_i_min*(L+c_pi_min*(Tm_K-Ice_t_min(x))))
        V_max_avg_temp = (Qdot)/(A*rho_i_avg*(L+c_pi_avg*(Tm_K-Ice_t_avg(x))))
        V_max_max_temp = (Qdot)/(A*rho_i_max*(L+c_pi_max*(Tm_K-Ice_t_max(x))))
        
        test = Ice_t_min(x)
        test2 = Ice_t_avg(x)
        test3 = Ice_t_max(x)

        Efficiency_min_temp = Efficiency_calc(Ice_t_min(x),V_max_min_temp,R,l,probe_mass)
        Efficiency_avg_temp = Efficiency_calc(Ice_t_avg(x),V_max_avg_temp,R,l,probe_mass)
        Efficiency_max_temp = Efficiency_calc(Ice_t_max(x),V_max_max_temp,R,l,probe_mass)

        V_current_min_temp = Efficiency_min_temp*V_max_min_temp
        V_current_avg_temp = Efficiency_avg_temp*V_max_avg_temp
        V_current_max_temp = Efficiency_max_temp*V_max_max_temp

        Velocity_Profile_min_t.append(V_current_min_temp)
        Velocity_Profile_avg_t.append(V_current_avg_temp)
        Velocity_Profile_max_t.append(V_current_max_temp)

    #returns velocity in m/s
    return(Europa_Range,Velocity_Profile_min_t,Velocity_Profile_avg_t,Velocity_Profile_max_t)

#TODO: Figure out integration to calculate transit time, not quite there with this one
#give depth to nearest 0.01
def TransitTime(ProbeFile,EuropaProfile,initial_position,final_position,melt_angle):
    #Give the depth in km, convert to europa radius here:
    #distance = 1560.8-depth
    
    x_range,velocity_min_t,velocity_avg_t,velocity_max_t = VelocityProfile(ProbeFile,EuropaProfile)
    #convert x_range to m
    melt_angle_rad = (melt_angle*math.pi)/180
    x_range = [value * 1000 for value in x_range]
    velocity_min_t = [value * -1*np.sin(melt_angle_rad) for value in velocity_min_t]
    velocity_max_t = [value * -1*np.sin(melt_angle_rad) for value in velocity_max_t]
    velocity_avg_t = [value * -1*np.sin(melt_angle_rad) for value in velocity_avg_t]
    velocity_min_t_interp = interp1d(x_range, velocity_min_t)
    velocity_avg_t_interp = interp1d(x_range, velocity_avg_t)
    velocity_max_t_interp = interp1d(x_range, velocity_max_t)

    #slowest_time = trapz(velocity_min_t, x_range)
    #print("x_range =", x_range)
    #print("velocity =", velocity_min_t)

    x_i_km = x_range[-1]
    x_f_km = final_position*1000
    time_taken_min_t, _ = integrate.quad(lambda x:-1 / velocity_min_t_interp(x), x_f_km, x_i_km)
    time_taken_avg_t, _ = integrate.quad(lambda x:-1 / velocity_avg_t_interp(x), x_f_km, x_i_km)
    time_taken_max_t, _ = integrate.quad(lambda x:-1 / velocity_max_t_interp(x), x_f_km, x_i_km)

    print("time taken days at min t = ", time_taken_min_t/86400)
    print("time taken days at max t = ", time_taken_max_t/86400)


def MeltPointWater():
    PressureList = [0.000612,0.1,1,2,5,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
    PressureList = np.array(PressureList)*10**6
    MeltPointList = [0.01,0.0026,-0.64,-0.14,-0.37,-0.75,-1.14,-1.54,-2.36,-3.21,-4.09,-5.00,-5.94,-6.91,-7.91,-8.94,-11.09,-13.35,-15.73,-18.22,-20.83]
    MeltPointList =np.array(MeltPointList)+273.15
    Spline = CubicSpline(PressureList,MeltPointList)
    return(Spline)

#Vel_test = VelocityCurrent4([20000,0.1,3.7,440],269.224,924.6,269.15)


#####TODO: Do everything below for the science and the relay probe, make it so number of relay probes can change...
#####TODO: Compare max comm depth with position list, add to max depth for loop

###### LOAD TOOLS AND PREAMBLE ########

#####Objects to stay in this file
class DepthCalculation:
    def check_condition(value):
        if value < 0:
            print("Condition met: value is negative. Exiting the script.")
            sys.exit()

    def CommAbove4(self,tavg,x_probe,x_surface,Freq,c_ice,salinity_interp,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,target_SNR,temp0,salt0):
        #xcord,tavg,= BryantInterp(filename)

        #x_list = np.linspace(x_probe+0.001,x_surface,1000)
        #seperation = x_list[-1]-x_list[-2]
        #print(temp_profile_list[0](x_probe/1000))
        #print(salt_profile_list[0](x_probe/1000))
        icelossfactor = PathLossFactorBryant(temp0(x_probe/1000),Frequency,1,c_ice,salt0(x_probe/1000))

        SNR_preamble = (P_tx_Sci*eff_tx_Sci*G_tx_Sci*eff_rx_Sci*G_rx_Sci)/(k_boltz*T_sys_Sci*BW)
        #print(SNR_preamble)
        for x_current in range(int(x_probe)+1,int(x_surface)+1):
            temp_current = tavg(x_current/1000)
            sal_current = salinity_interp(x_current/1000)
            x_range = x_current-x_probe
            #print('x_range: ',x_range)
            fspl = (3e8/(math.pi*Freq*(x_range*1000)*4))**2
            loss_without_ice = 10*np.log(SNR_preamble)+10*np.log(fspl)
            ice_noise = 20*np.log(icelossfactor)
            #print('fspl: ', fspl)
            #print('iceloss: ', icelossfactor)
            total_loss_add = loss_without_ice + ice_noise
            #print('add: ', total_loss_add)
            total_loss = fspl*icelossfactor
            
            #print('SNR: ',SNR)
            SNR = loss_without_ice + ice_noise
            
            if SNR <= 0:
                SNR_db = 100000
            else:
                SNR_db = 10*math.log(SNR)
                #print('snr_db: ',SNR_db)

            if SNR_db < target_SNR:
                return(x_current)
            elif int(x_current) >= int(x_surface):
                return(x_surface)
            else:
                icelossfactor *= PathLossFactorBryant(temp_current,Frequency,1,c_ice,sal_current)



    def __init__(self, **kwargs):
        pass

    def SingleDepth(self,mission_configs,z_temp,z_rho,z_salt,index_list, x_grid,
                Frequency,BW,Target_SNR,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,T_sys_Sci,
                Power_to_Melt_Sci,Melt_eff_Sci,R_Science,l_Science,mass_Science):
        def CommAbove4(tavg,x_probe,x_surface,Freq,c_ice,salinity_interp,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,target_SNR,temp0,salt0):
            #xcord,tavg,= BryantInterp(filename)

            #x_list = np.linspace(x_probe+0.001,x_surface,1000)
            #seperation = x_list[-1]-x_list[-2]
            #print(temp_profile_list[0](x_probe/1000))
            #print(salt_profile_list[0](x_probe/1000))
            icelossfactor = PathLossFactorBryant(temp0(x_probe/1000),Frequency,1,c_ice,salt0(x_probe/1000))

            SNR_preamble = (P_tx_Sci*eff_tx_Sci*G_tx_Sci*eff_rx_Sci*G_rx_Sci)/(k_boltz*T_sys_Sci*BW)
            #print(SNR_preamble)
            for x_current in range(int(x_probe)+1,int(x_surface)+1):
                temp_current = tavg(x_current/1000)
                sal_current = salinity_interp(x_current/1000)
                x_range = x_current-x_probe
                #print('x_range: ',x_range)
                fspl = (3e8/(math.pi*Freq*(x_range*1000)*4))**2
                loss_without_ice = 10*np.log(SNR_preamble)+10*np.log(fspl)
                ice_noise = 20*np.log(icelossfactor)
                #print('fspl: ', fspl)
                #print('iceloss: ', icelossfactor)
                total_loss_add = loss_without_ice + ice_noise
                #print('add: ', total_loss_add)
                total_loss = fspl*icelossfactor
                
                #print('SNR: ',SNR)
                SNR = loss_without_ice + ice_noise
                
                if SNR <= 0:
                    SNR_db = 100000
                else:
                    SNR_db = 10*math.log(SNR)
                    #print('snr_db: ',SNR_db)

                if SNR_db < target_SNR:
                    return(x_current)
                elif int(x_current) >= int(x_surface):
                    return(x_surface)
                else:
                    icelossfactor *= PathLossFactorBryant(temp_current,Frequency,1,c_ice,sal_current)


        
        Qdot_Science = Power_to_Melt_Sci*Melt_eff_Sci
        Science_Probe_Profile = [Qdot_Science,R_Science,l_Science,mass_Science]
        


###objects to go in setupfile
        slices =  mission_configs['slices']
        timestep_s = mission_configs['timestep_s']
        n_value = mission_configs['n_value']
        filename = mission_configs['filename']
        k_boltz = mission_configs['k_boltz']
        Mission_Time_Limit_d =  mission_configs['Mission_Time_Limit_d']
        Mission_Time_Limit_s = Mission_Time_Limit_d*86400
        ##### Make each slice a 1D interpolation called by SliceList[i]
        TempSliceList = []
        SaltSliceList = []
        RhoSliceList = []
        xSliceList = []
        for i in range(len(index_list)):

            list_index = index_list[i]

            x_slice = x_grid[list_index][~np.isnan(x_grid[list_index])]

            #print('test4:', x_slice)

            temp1 = z_temp[list_index][~np.isnan(z_temp[list_index])]
            temp1 = temp1.reshape(-1)

            saltconc1 = z_salt[list_index][~np.isnan(z_salt[list_index])]
            saltconc1 = saltconc1.reshape(-1)

            rho1 = z_rho[list_index][~np.isnan(z_salt[list_index])]
            rho1 = rho1.reshape(-1)
            #print(temp1_x_rev)
            #Reversing because cubicspline NEEDS increasing sequence
            temp_slice = CubicSpline(x_slice,temp1)
            salt_slice = CubicSpline(x_slice,saltconc1)
            rho_slice = CubicSpline(x_slice,rho1)
            TempSliceList.append(temp_slice)
            SaltSliceList.append(salt_slice)
            RhoSliceList.append(rho_slice)
            xSliceList.append(x_slice)

        little_indexes = range(1,slices)
        little_indexes = little_indexes[n_value-1::n_value]

        temp_profile_list = []
        salt_profile_list = []
        rho_profile_list = []
        x_profile_list = []
        for index in little_indexes:

            temp_profile_list.append(TempSliceList[index])
            salt_profile_list.append(SaltSliceList[index])
            rho_profile_list.append(RhoSliceList[index])
            x_profile_list.append(xSliceList[index])

        flatx = [value for sublist in x_profile_list for value in sublist]

        # Find the minimum value in the flattened lis
        x_abs_min = min(flatx)
        x_abs_max = max(flatx)

        ######Comm Height Calculation
        c_ice=1.6e8
        Max_Comm_Height_list_Sci = []
        Max_Comm_Height_list_Rel = []
        x_range_list = []
        x_surface_list = []
        x_ice_water_list = []
        #for profile_index in range(len(little_indexes)):
        for profile_index in range(len(little_indexes)):
        
            x_range = [x * 1000 for x in x_profile_list[profile_index]]
            x_range_list.append(x_range)

            x_ice_water = min(x_range)
            x_surface = max(x_range)

            x_surface_list.append(x_surface)
            x_ice_water_list.append(x_ice_water)

        #initiate start point
            Sci_Max_Comm_List = []
            

        #Go through x positions and calculate max height above which a probe can communicate
        #Returns surface if it can communicate back to surface

            comm_range_list = range(int(x_ice_water), int(x_surface),100)
            #print(len(comm_range_list))
            comm_data_list = []
            for i in comm_range_list:
                Probe_position = i
                #print(Probe_position)
                comm_data_list.append(Probe_position)
                #print('Probe Depth: ', Probe_position)
                x_current = Probe_position
                x_current += 100
                Sci_calc =CommAbove4(temp_profile_list[profile_index],Probe_position,x_surface,Frequency,c_ice,salt_profile_list[profile_index],P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,Target_SNR,temp_profile_list[0],salt_profile_list[0])
                
                Sci_Max_Comm_List.append(Sci_calc)
                
                #print(g)       
                #Probe_position +=1
            #print(Max_Comm_List)
            #print(Maxx_Comm_List[-1])
            Sci_Max_Comm_spline = CubicSpline(comm_data_list,Sci_Max_Comm_List)
            

            Max_Comm_Height_list_Sci.append(Sci_Max_Comm_spline)
            

            ####Max_Comm_Height takes the position of a probe x, and returns the position x that the probe is able to communicate to! GOOD job :)
            
            #Max_Comm_Height_list_Sci.append(CubicSpline(comm_range_list,Max_Comm_List))

        ####Calculate the pressure on a point of water and therefore the melt temperature
        #### Go down through the water from the surface, and caluclate the mass of ice above the point
        ####do what needs to be done eith the Haynes calculation reference
        #### return cubic spline of melt temperatures.

        #Melting temp of water  value: 273.152519 p = 101325 Pa

        Mpw = MeltPointWater()
        Mp_w_ScenarioSpline_list = []

        for profile_index in range(len(little_indexes)):
            Pressure_range_list = range(int(min(x_range_list[profile_index])),int(max(x_range_list[profile_index])),500)
            rho_for_pressure_list = []
            Pressure_list = []
            
            MeltPointList =[]
            for i in Pressure_range_list:
                Density_current = rho_profile_list[profile_index](i/1000)
                rho_for_pressure_list.append(Density_current)
                Mean_density_Above = np.mean(rho_for_pressure_list)
                #print(Mean_density_Above)
                ice_above = abs(i-max(x_range_list[profile_index]))
                #print(ice_above)
                Pressure = (Mean_density_Above*ice_above*1.3)
                MeltPointList.append(Mpw(Pressure))


            #### MP_w_Scenario takes the current position in m and returns the melting point of water for that pressure.
            Mp_w_ScenarioSpline = CubicSpline(Pressure_range_list,MeltPointList)
            Mp_w_ScenarioSpline_list.append(Mp_w_ScenarioSpline )

            ###Initisialise Velocity of Probes at the surface and t=0

            Temp_surface = temp_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Salinity_surface = salt_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Rho_surface = rho_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Tm_surface = Mp_w_ScenarioSpline_list[profile_index](x_surface_list[profile_index])

            V_0_Science = VelocityCurrent4(Science_Probe_Profile,Temp_surface,Rho_surface,Tm_surface)

            def check_condition(value):
                if value <= 0:
                    print("V_0 = zero")
                    return (True)
                else:
                    return (False)

            #if check_condition()        
            
            V_0_Science_mh = V_0_Science*3600
            V_0_Science_kmd = V_0_Science*86.4

    

        ####For Science&Relay: Go through each depth and caculate the velocity of the probes

        Science_Velocity_profiles = []

        Temp_profiles = []
        Melt_point_profiles = []
        for profile_index in range(len(little_indexes)):
            Science_Velocity_List = []
            
            TempList =[]
            for x_position in x_range_list[profile_index]:
                Tm_current= Mp_w_ScenarioSpline_list[profile_index](x_position)
                Temp_current = temp_profile_list[profile_index](x_position/1000)
                TempList.append(Temp_current)
                Rho_current = rho_profile_list[profile_index](x_position/1000)
                Science_V_Current = VelocityCurrent4(Science_Probe_Profile,Temp_current,Rho_current,Tm_current)
                Sci_CRF = RefreezeLengthBool(temp_profile_list[profile_index](x_position/1000),Science_V_Current,R_Science,l_Science,Qdot_Science,Mp_w_ScenarioSpline_list[profile_index](x_position),1-Melt_eff_Sci)
                if Sci_CRF == True:
                    Science_Velocity_List.append(0)
                else:
                    Science_Velocity_List.append(Science_V_Current*86.4)
                #print(Science_V_Current)
                
            
            
            Science_Vel_splined = CubicSpline(x_range_list[profile_index],Science_Velocity_List)
            
            Temp_splined = CubicSpline(x_range_list[profile_index],TempList)
            
            
            Science_Velocity_profiles.append(Science_Vel_splined)
            
            Temp_profiles.append(Temp_splined)


        ####Calculate the transition of the probes over time ######
        timeline = range(0,Mission_Time_Limit_s,timestep_s)

        Science_Descent_profiles = []
        
        Transit_times_s = []
        Max_Depth_Reached = []
        Science_Total_Power_list = []
        Science_Energy_per_m_list = []
        

        Break_Reasons = []

        for profile_index in range(len(little_indexes)):
            Science_v = Science_Velocity_profiles[profile_index]
            
            x_0 = max(x_range_list[profile_index])
            
            Science_Transit_list = [0]
            
            Sci_Max_Comm_check = Max_Comm_Height_list_Sci[profile_index]
            
            
            Science_x = x_0 
            Relay_x = x_0
             
            
            Science_v_current = Science_v(Science_x)*(1000 / (24 * 3600))
            

            for t in timeline:
                
                Science_v_current = Science_v(Science_x)*(1000 / (24 * 3600))
                #print(Science_v_current)
                Science_x -= Science_v_current*timestep_s
                Science_Transit_list.append(Science_x-x_0)

                #print('test: ', Sci_Max_Comm_check(Science_x)-x_0)
                
                #Sci_a = RefreezeLengthBool(temp_profile_list[profile_index](Science_x/1000),Science_v_current,R_Science,l_Science,Qdot_Science,Mp_w_ScenarioSpline_list[profile_index](Science_x),1-Melt_eff_Sci)
                #print(Science_x/1000)
                #print(Science_x)
                #print(Sci_a)
                if int(Science_x) < int(min(x_range_list[profile_index])):
                    Transit_times_s.append(t)
                    Max_Depth_Reached.append(min(x_range_list[profile_index])-x_0)

                    Science_Total_Power_list.append(Qdot_Science*t)
                    Science_Energy_per_m_list.append(-Qdot_Science*t/(min(x_range_list[profile_index])-x_0))
                    

                    Break_Reasons.append('Water Reached')
                    #print('a')
                    break

                elif t == timeline[-1]:
                    Transit_times_s.append(t)
                    Max_Depth_Reached.append(Science_x-x_0)

                    Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                    Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                    

                    Break_Reasons.append('Mission Limit')
                    #print('b')
                    break

                elif Science_v_current <= 0:
                    Transit_times_s.append(t)
                    Max_Depth_Reached.append(Science_x-x_0)
                    Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                    Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                    Break_Reasons.append('Sci Frozen')
                    #print('e')
                    break 

                elif int(Sci_Max_Comm_check(Science_x)) < int(x_0):
                    Transit_times_s.append(t)
                    Max_Depth_Reached.append(Science_x-x_0)
                    Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                    Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                    Break_Reasons.append('Science Comms')
                    #print('d')
                    break

            
            #print(Science_Transit_list)
            Science_Descent_profiles.append(Science_Transit_list)

        ####Returns min, as depth is a negative number from surface


        Transit_time_mean = np.mean(Transit_times_s)
        Transit_time_sd = np.std(Transit_times_s)
        Transit_time_min = (min(Transit_times_s))
        Transit_time_max = (max(Transit_times_s))
        percentiles = np.percentile(Transit_times_s, [25,50,75])
        Transit_time_25 = (percentiles[0])
        Transit_time_50 = (percentiles[1])
        Transit_time_75 = (percentiles[2])

        Science_Total_Power_mean = np.mean(Science_Total_Power_list)
        Science_Total_Power_sd = np.std(Science_Total_Power_list)
        Science_Total_Power_min = (min(Science_Total_Power_list))
        Science_Total_Power_max = (max(Science_Total_Power_list))
        percentiles = np.percentile(Science_Total_Power_list, [25,50,75])
        Science_Total_Power_25 = (percentiles[0])
        Science_Total_Power_50 = (percentiles[1])
        Science_Total_Power_75 = (percentiles[2])

        Max_Depth_mean = np.mean(Max_Depth_Reached)
        Max_Depth_sd = np.std(Max_Depth_Reached)
        Max_Depth_min = (min(Max_Depth_Reached))
        Max_Depth_max = (max(Max_Depth_Reached))
        percentiles = np.percentile(Max_Depth_Reached, [25,50,75])
        Max_Depth_25 = (percentiles[0])
        Max_Depth_50 = (percentiles[1])
        Max_Depth_75 = (percentiles[2])

        #Max_data = np.multiply(Frequency,Transit_times_mea)

        #print('Max_depth: ', Max_Depth_mean)
        #print('Transit Time: ', Transit_time_mean)
        return [Max_Depth_mean, np.mean(Transit_time_mean)]
    
    def TwoDepth(self,mission_configs,z_temp,z_rho,z_salt,index_list, x_grid,
                Frequency,BW,Target_SNR,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,T_sys_Sci,
                P_tx_Rel,eff_tx_Rel,G_tx_Rel,eff_rx_Rel,G_rx_Rel,T_sys_Rel,
                Power_to_Melt_Sci,Melt_eff_Sci,R_Science,l_Science,mass_Science,
                Power_to_Melt_Rel,Melt_eff_Relay,R_Relay,l_Relay,mass_Relay):
        
        def CommAbove4(tavg,x_probe,x_surface,Freq,c_ice,salinity_interp,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,target_SNR,temp0,salt0):
            #xcord,tavg,= BryantInterp(filename)

            #x_list = np.linspace(x_probe+0.001,x_surface,1000)
            #seperation = x_list[-1]-x_list[-2]
            #print(temp_profile_list[0](x_probe/1000))
            #print(salt_profile_list[0](x_probe/1000))
            icelossfactor = PathLossFactorBryant(temp0(x_probe/1000),Frequency,1,c_ice,salt0(x_probe/1000))

            SNR_preamble = (P_tx_Sci*eff_tx_Sci*G_tx_Sci*eff_rx_Sci*G_rx_Sci)/(k_boltz*T_sys_Sci*BW)
            #print(SNR_preamble)
            for x_current in range(int(x_probe)+1,int(x_surface)+1):
                temp_current = tavg(x_current/1000)
                sal_current = salinity_interp(x_current/1000)
                x_range = x_current-x_probe
                #print('x_range: ',x_range)
                fspl = (3e8/(math.pi*Freq*(x_range*1000)*4))**2
                loss_without_ice = 10*np.log(SNR_preamble)+10*np.log(fspl)
                ice_noise = 20*np.log(icelossfactor)
                #print('fspl: ', fspl)
                #print('iceloss: ', icelossfactor)
                total_loss_add = loss_without_ice + ice_noise
                #print('add: ', total_loss_add)
                total_loss = fspl*icelossfactor
                
                #print('SNR: ',SNR)
                SNR = loss_without_ice + ice_noise
                
                if SNR <= 0:
                    SNR_db = 100000
                else:
                    SNR_db = 10*math.log(SNR)
                    #print('snr_db: ',SNR_db)

                if SNR_db < target_SNR:
                    return(x_current)
                elif int(x_current) >= int(x_surface):
                    return(x_surface)
                else:
                    icelossfactor *= PathLossFactorBryant(temp_current,Frequency,1,c_ice,sal_current)


        Qdot_Relay = Power_to_Melt_Rel*Melt_eff_Relay
        Qdot_Science = Power_to_Melt_Sci*Melt_eff_Sci
        Science_Probe_Profile = [Qdot_Science,R_Science,l_Science,mass_Science]
        Relay_Probe_Profile = [Qdot_Relay,R_Relay,l_Relay,mass_Relay]


###objects to go in setupfile
        slices =  mission_configs['slices']
        timestep_s = mission_configs['timestep_s']
        n_value = mission_configs['n_value']
        filename = mission_configs['filename']
        k_boltz = mission_configs['k_boltz']
        Mission_Time_Limit_d =  mission_configs['Mission_Time_Limit_d']
        Mission_Time_Limit_s = Mission_Time_Limit_d*86400
        ##### Make each slice a 1D interpolation called by SliceList[i]
        TempSliceList = []
        SaltSliceList = []
        RhoSliceList = []
        xSliceList = []
        for i in range(len(index_list)):

            list_index = index_list[i]

            x_slice = x_grid[list_index][~np.isnan(x_grid[list_index])]

            #print('test4:', x_slice)

            temp1 = z_temp[list_index][~np.isnan(z_temp[list_index])]
            temp1 = temp1.reshape(-1)

            saltconc1 = z_salt[list_index][~np.isnan(z_salt[list_index])]
            saltconc1 = saltconc1.reshape(-1)

            rho1 = z_rho[list_index][~np.isnan(z_salt[list_index])]
            rho1 = rho1.reshape(-1)
            #print(temp1_x_rev)
            #Reversing because cubicspline NEEDS increasing sequence
            temp_slice = CubicSpline(x_slice,temp1)
            salt_slice = CubicSpline(x_slice,saltconc1)
            rho_slice = CubicSpline(x_slice,rho1)
            TempSliceList.append(temp_slice)
            SaltSliceList.append(salt_slice)
            RhoSliceList.append(rho_slice)
            xSliceList.append(x_slice)

        little_indexes = range(1,slices)
        little_indexes = little_indexes[n_value-1::n_value]

        temp_profile_list = []
        salt_profile_list = []
        rho_profile_list = []
        x_profile_list = []
        for index in little_indexes:

            temp_profile_list.append(TempSliceList[index])
            salt_profile_list.append(SaltSliceList[index])
            rho_profile_list.append(RhoSliceList[index])
            x_profile_list.append(xSliceList[index])

        flatx = [value for sublist in x_profile_list for value in sublist]

        # Find the minimum value in the flattened lis
        x_abs_min = min(flatx)
        x_abs_max = max(flatx)

        ######Comm Height Calculation
        c_ice=1.6e8
        Max_Comm_Height_list_Sci = []
        Max_Comm_Height_list_Rel = []
        x_range_list = []
        x_surface_list = []
        x_ice_water_list = []
        #for profile_index in range(len(little_indexes)):
        for profile_index in range(len(little_indexes)):
        
            x_range = [x * 1000 for x in x_profile_list[profile_index]]
            x_range_list.append(x_range)

            x_ice_water = min(x_range)
            x_surface = max(x_range)

            x_surface_list.append(x_surface)
            x_ice_water_list.append(x_ice_water)

        #initiate start point
            Sci_Max_Comm_List = []
            Rel_Max_Comm_List = []

        #Go through x positions and calculate max height above which a probe can communicate
        #Returns surface if it can communicate back to surface

            comm_range_list = range(int(x_ice_water), int(x_surface),100)
            #print(len(comm_range_list))
            comm_data_list = []
            for i in comm_range_list:
                Probe_position = i
                #print(Probe_position)
                comm_data_list.append(Probe_position)
                #print('Probe Depth: ', Probe_position)
                x_current = Probe_position
                x_current += 100
                Sci_calc =CommAbove4(temp_profile_list[profile_index],Probe_position,x_surface,Frequency,c_ice,salt_profile_list[profile_index],P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,Target_SNR,temp_profile_list[0],salt_profile_list[0])
                Rel_calc = CommAbove4(temp_profile_list[profile_index],Probe_position,x_surface,Frequency,c_ice,salt_profile_list[profile_index],P_tx_Rel,eff_tx_Rel,G_tx_Rel,eff_rx_Rel,G_rx_Rel,k_boltz,T_sys_Rel,BW,Target_SNR,temp_profile_list[0],salt_profile_list[0])
                Sci_Max_Comm_List.append(Sci_calc)
                Rel_Max_Comm_List.append(Rel_calc)
                #print(g)       
                #Probe_position +=1
            #print(Max_Comm_List)
            #print(Maxx_Comm_List[-1])
            Sci_Max_Comm_spline = CubicSpline(comm_data_list,Sci_Max_Comm_List)
            Rel_Max_Comm_spline = CubicSpline(comm_data_list,Rel_Max_Comm_List)

            Max_Comm_Height_list_Sci.append(Sci_Max_Comm_spline)
            Max_Comm_Height_list_Rel.append(Rel_Max_Comm_spline)

            ####Max_Comm_Height takes the position of a probe x, and returns the position x that the probe is able to communicate to! GOOD job :)
            
            #Max_Comm_Height_list_Sci.append(CubicSpline(comm_range_list,Max_Comm_List))

        ####Calculate the pressure on a point of water and therefore the melt temperature
        #### Go down through the water from the surface, and caluclate the mass of ice above the point
        ####do what needs to be done eith the Haynes calculation reference
        #### return cubic spline of melt temperatures.

        #Melting temp of water  value: 273.152519 p = 101325 Pa

        Mpw = MeltPointWater()
        Mp_w_ScenarioSpline_list = []

        for profile_index in range(len(little_indexes)):
            Pressure_range_list = range(int(min(x_range_list[profile_index])),int(max(x_range_list[profile_index])),500)
            rho_for_pressure_list = []
            Pressure_list = []
            
            MeltPointList =[]
            for i in Pressure_range_list:
                Density_current = rho_profile_list[profile_index](i/1000)
                rho_for_pressure_list.append(Density_current)
                Mean_density_Above = np.mean(rho_for_pressure_list)
                #print(Mean_density_Above)
                ice_above = abs(i-max(x_range_list[profile_index]))
                #print(ice_above)
                Pressure = (Mean_density_Above*ice_above*1.3)
                MeltPointList.append(Mpw(Pressure))


            #### MP_w_Scenario takes the current position in m and returns the melting point of water for that pressure.
            Mp_w_ScenarioSpline = CubicSpline(Pressure_range_list,MeltPointList)
            Mp_w_ScenarioSpline_list.append(Mp_w_ScenarioSpline )

            ###Initisialise Velocity of Probes at the surface and t=0

            Temp_surface = temp_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Salinity_surface = salt_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Rho_surface = rho_profile_list[profile_index](x_surface_list[profile_index]/1000)
            Tm_surface = Mp_w_ScenarioSpline_list[profile_index](x_surface_list[profile_index])

            V_0_Science = VelocityCurrent4(Science_Probe_Profile,Temp_surface,Rho_surface,Tm_surface)
            V_0_Relay = VelocityCurrent4(Relay_Probe_Profile,Temp_surface,Rho_surface,Tm_surface)

            V_0_Science_mh = V_0_Science*3600
            V_0_Science_kmd = V_0_Science*86.4

            V_0_Relay_kmd = V_0_Relay*86.4

        ####For Science&Relay: Go through each depth and caculate the velocity of the probes
        #Ends the run if the probe won't move
        if V_0_Science <= 0:
            return([0,0])
        
        else:
                
            Science_Velocity_profiles = []
            Relay_Velocity_profiles = []
            Temp_profiles = []
            Melt_point_profiles = []
            for profile_index in range(len(little_indexes)):
                Science_Velocity_List = []
                Relay_Velocity_List = []
                TempList =[]
                for x_position in x_range_list[profile_index]:
                    Tm_current= Mp_w_ScenarioSpline_list[profile_index](x_position)
                    Temp_current = temp_profile_list[profile_index](x_position/1000)
                    TempList.append(Temp_current)
                    Rho_current = rho_profile_list[profile_index](x_position/1000)
                    Science_V_Current = VelocityCurrent4(Science_Probe_Profile,Temp_current,Rho_current,Tm_current)
                    Sci_CRF = RefreezeLengthBool(temp_profile_list[profile_index](x_position/1000),Science_V_Current,R_Science,l_Science,Qdot_Science,Mp_w_ScenarioSpline_list[profile_index](x_position),1-Melt_eff_Sci)
                    if Sci_CRF == True:
                        Science_Velocity_List.append(0)
                    else:
                        Science_Velocity_List.append(Science_V_Current*86.4)
                    #print(Science_V_Current)
                    Relay_V_Current = VelocityCurrent4(Relay_Probe_Profile,Temp_current,Rho_current,Tm_current)
                    Relay_Velocity_List.append(Relay_V_Current*86.4)
                
                #print(Science_Velocity_List)
                Science_Vel_splined = CubicSpline(x_range_list[profile_index],Science_Velocity_List)
                Relay_Vel_splined = CubicSpline(x_range_list[profile_index],Relay_Velocity_List)
                Temp_splined = CubicSpline(x_range_list[profile_index],TempList)
                
                
                Science_Velocity_profiles.append(Science_Vel_splined)
                Relay_Velocity_profiles.append(Relay_Vel_splined)
                Temp_profiles.append(Temp_splined)


            ####Calculate the transition of the probes over time ######
            timeline = range(0,Mission_Time_Limit_s,timestep_s)

            Science_Descent_profiles = []
            Relay_Descent_profiles = []
            Transit_times_s = []
            Max_Depth_Reached = []
            Science_Total_Power_list = []
            Science_Energy_per_m_list = []
            Relay_Total_Power_list = []
            Relay_Energy_per_m_list = []

            Break_Reasons = []

            for profile_index in range(len(little_indexes)):
                Science_v = Science_Velocity_profiles[profile_index]
                Relay_v = Relay_Velocity_profiles[profile_index]
                x_0 = max(x_range_list[profile_index])
                
                Science_Transit_list = [0]
                Relay_Transit_list =[0]
                Sci_Max_Comm_check = Max_Comm_Height_list_Sci[profile_index]
                Relay_Max_Comm_check = Max_Comm_Height_list_Rel[profile_index]
                
                Science_x = x_0 
                Relay_x = x_0
                #CHECK SCIENCE_v_current and freeze condition above, almost there! 
                #print(Science_x)
                Science_v_current = Science_v(Science_x)*(1000 / (24 * 3600))
                #print(Science_v_current)

                for t in timeline:
                    
                    Science_v_current = Science_v(Science_x)*(1000 / (24 * 3600))
                    #print(Science_v_current)
                    Science_x -= Science_v_current*timestep_s
                    Science_Transit_list.append(Science_x-x_0)

                    Relay_v_current = Relay_v(Relay_x)*(1000 / (24 * 3600))
                    Relay_x -= Relay_v_current*timestep_s
                    Relay_Transit_list.append(Relay_x-x_0)

                    #print('test: ', Sci_Max_Comm_check(Science_x)-x_0)
                    
                    #Sci_a = RefreezeLengthBool(temp_profile_list[profile_index](Science_x/1000),Science_v_current,R_Science,l_Science,Qdot_Science,Mp_w_ScenarioSpline_list[profile_index](Science_x),1-Melt_eff_Sci)
                    #print(Science_x/1000)
                    #print(Science_x)
                    #print(Sci_a)
                    if int(Science_x) < int(min(x_range_list[profile_index])):
                        Transit_times_s.append(t)
                        Max_Depth_Reached.append(min(x_range_list[profile_index])-x_0)

                        Science_Total_Power_list.append(Qdot_Science*t)
                        Science_Energy_per_m_list.append(-Qdot_Science*t/(min(x_range_list[profile_index])-x_0))
                        Relay_Total_Power_list.append(Qdot_Relay*t)
                        Relay_Energy_per_m_list.append(-Qdot_Relay*t/(min(x_range_list[profile_index])-x_0))

                        Break_Reasons.append('Water Reached')
                        #print('a')
                        break

                    elif t == timeline[-1]:
                        Transit_times_s.append(t)
                        Max_Depth_Reached.append(Science_x-x_0)

                        Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                        Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                        Relay_Total_Power_list.append(Qdot_Relay*t)
                        Relay_Energy_per_m_list.append(-Qdot_Relay*t/(min(x_range_list[profile_index])-x_0))

                        Break_Reasons.append('Mission Limit')
                        #print('b')
                        break

                    elif Science_v_current <= 0:
                        Transit_times_s.append(t)
                        Max_Depth_Reached.append(Science_x-x_0)
                        Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                        Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                        Break_Reasons.append('Sci Frozen')
                        #print('e')
                        break
                    
                    elif int(Relay_Max_Comm_check(Relay_x)) < int(x_0):
                        Transit_times_s.append(t)
                        Max_Depth_Reached.append(Science_x-x_0)

                        Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                        Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                        Relay_Total_Power_list.append(Qdot_Relay*t)
                        Relay_Energy_per_m_list.append(-Qdot_Relay*t/(min(x_range_list[profile_index])-x_0))

                        Break_Reasons.append('Relay Comms')
                        #print('c')
                        break

                    elif int(Sci_Max_Comm_check(Science_x)) < int(Relay_x):
                        Transit_times_s.append(t)
                        Max_Depth_Reached.append(Science_x-x_0)
                        Science_Total_Power_list.append(Qdot_Science*Mission_Time_Limit_s)
                        Science_Energy_per_m_list.append(-Qdot_Science*Mission_Time_Limit_s/(min(x_range_list[profile_index])-x_0))
                        Break_Reasons.append('Science Comms')
                        #print('d')
                        break

                
                #print(Science_Transit_list)
                Science_Descent_profiles.append(Science_Transit_list)

            ####Returns min, as depth is a negative number from surface


            Transit_time_mean = np.mean(Transit_times_s)
            Transit_time_sd = np.std(Transit_times_s)
            Transit_time_min = (min(Transit_times_s))
            Transit_time_max = (max(Transit_times_s))
            percentiles = np.percentile(Transit_times_s, [25,50,75])
            Transit_time_25 = (percentiles[0])
            Transit_time_50 = (percentiles[1])
            Transit_time_75 = (percentiles[2])

            Science_Total_Power_mean = np.mean(Science_Total_Power_list)
            Science_Total_Power_sd = np.std(Science_Total_Power_list)
            Science_Total_Power_min = (min(Science_Total_Power_list))
            Science_Total_Power_max = (max(Science_Total_Power_list))
            percentiles = np.percentile(Science_Total_Power_list, [25,50,75])
            Science_Total_Power_25 = (percentiles[0])
            Science_Total_Power_50 = (percentiles[1])
            Science_Total_Power_75 = (percentiles[2])

            Max_Depth_mean = np.mean(Max_Depth_Reached)
            Max_Depth_sd = np.std(Max_Depth_Reached)
            Max_Depth_min = (min(Max_Depth_Reached))
            Max_Depth_max = (max(Max_Depth_Reached))
            percentiles = np.percentile(Max_Depth_Reached, [25,50,75])
            Max_Depth_25 = (percentiles[0])
            Max_Depth_50 = (percentiles[1])
            Max_Depth_75 = (percentiles[2])

            #Max_data = np.multiply(Frequency,Transit_times_mea)

            #print('Max_depth: ', Max_Depth_mean)
            #print('Transit Time: ', Transit_time_mean)
            return [Max_Depth_mean, np.mean(Transit_time_mean)]

######MAIN CODE RUN######

def Single_Run(ice_pointer,power_pointer,mass_pointer,dev_mass_pointer):
    code_time_0 = time.time()


    ####constant definitions
    k_boltz =  1.38e-23
    #Specific_System_Power = 18.52
    #Specific_System_Power = 18.52
    Specific_System_Power = power_pointer
    slices =  776
    #deliverable_mass = 42.5
    #deliverable_mass = 42.5
    deliverable_mass = dev_mass_pointer
    rho_lead = 11400
    volume_limit = 3.5*3.2
    Melt_Comm_Combined_Power = 0.9
    Melt_Comm_Combined_Mass = 0.8
    diameter_limit = 3.5

    Mission_time_limit_d = 30

    input_location = '/home/ec2-user/AWS Files/AWS Files/InputYamls/'
    Mission_filename = 'MissionVariables.yaml'
    #filename = '40km_Drho5_2D'
    filename = ice_pointer
    ice_filename = filename
    filename_list = ['40km_Drho5_2D','40km_Drho11_2D','40km_Drho23_2D','40km_Drho46_2D']
    with open (input_location+Mission_filename, 'r') as file:
        mission_configs = yaml.safe_load(file)

    variable_symbols = []
    variable_labels = []
    variable_lower_lims = []
    variable_upper_lims = []

   #'MOOSetup.yaml' 
    setup_file_path = '/home/ec2-user/AWS Files/AWS Files/{}'.format(mass_pointer)
    with open(setup_file_path, 'r') as file:
        # Load the contents of the file
        config = yaml.safe_load(file)

    for key in config:
        variable_labels.append(key)
        variable_symbols.append(config[key]['var_symbol'])
        variable_lower_lims.append(config[key]['l_limit'])
        variable_upper_lims.append(config[key]['u_limit'])

    n_var = len(variable_labels)
    #print(variable_labels.index('Duration of mission'))

    
    z_label = ['Temperature', 'Density', 'Salt']
    x_grid_list = []
    z_temp_list = []
    z_salt_list = []
    z_rho_list = []
    index_list_list = []

    for file_i in filename_list:
        xcord,ycord,temp,rho,saltconc = Fileread2D(file_i)
        
        x = np.array(xcord)
        y = np.array(ycord)
        z1 = np.array(temp)
        z2 = np.array(rho)
        z3 = np.array(saltconc)
        
        zlist = [z1,z2,z3]

        linspacing = slices
        xi, yi = np.linspace(x.min(), x.max(), linspacing), np.linspace(y.min(), y.max(), linspacing)
        xi, yi = np.meshgrid(xi, yi, sparse = True)

        # Interpolate
        import scipy.interpolate
        z_temp = scipy.interpolate.griddata((x,y), z1, (xi,yi))
        z_rho = scipy.interpolate.griddata((x,y), z2, (xi,yi))
        z_salt = scipy.interpolate.griddata((x,y), z3, (xi,yi))

        x_grid = scipy.interpolate.griddata((x,y), x, (xi,yi))
        index_list = [x for x in range(1, slices-1)]
        num_points = len(index_list)
        x_grid_list.append(x_grid)
        z_temp_list.append(z_temp)
        z_salt_list.append(z_salt)
        z_rho_list.append(z_rho)
        index_list_list.append(index_list)

    #Initialise constarint list
    constraint_list = []

    # Define the inequality constraints
    #This is all the constraints that can't be described by numbers (e.g. sum of probe masses)
    ###Sum of probe masses 

    #####System Constraints#####

    ###Sum of masses can not be greater than deliveable mass
    def Sys_constraint1(vars):
        probe1mass, probe2mass =  vars[variable_labels.index('science_probe_mass')], vars[variable_labels.index('relay_probe_mass')]
        #return np.sum([probe1mass,probe2mass]) - deliverable_mass  # This should be <= 0
        sol = probe1mass + probe2mass - deliverable_mass
        return sol   # This should be <= 0

    ###total volume of probes DELETED... not needed, just make it implicit in variable limits

    #density of relay probes does not exceed physical limit
    def Sys_constraint3(vars):
        relay_volume = vars[variable_labels.index('relay_probe_length')]*np.pi*(vars[variable_labels.index('relay_probe_diameter')]/2)**2 
        density = vars[variable_labels.index('relay_probe_mass')]/relay_volume
        sol = density - rho_lead
        return  sol # This should be <= 0 (i.e., y >= 0)

    #density of science probes does not exceed physical limit
    def Sys_constraint4(vars):
        sci_volume = vars[variable_labels.index('science_probe_length')]*np.pi*(vars[variable_labels.index('science_probe_diameter')]/2)**2 
        density = vars[variable_labels.index('science_probe_mass')]/sci_volume
        sol = density - rho_lead
        return sol  # This should be <= 0 (i.e., z >= 0)

    #Comm and Power mass must not exceed 0.6 the science probe
    def Sys_constraint5(vars):
        return vars[variable_labels.index('science_probe_RTG_ratio')]+ vars[variable_labels.index('science_probe_Comm_ratio')] - Melt_Comm_Combined_Mass

    #Comm and Power mass must not exceed 0.6 the relay probe
    def Sys_constraint6(vars):
        return vars[variable_labels.index('relay_probe_RTG_ratio')]+ vars[variable_labels.index('relay_probe_Comm_ratio')] - Melt_Comm_Combined_Mass

    #Maintain cylindrical shape of science probe
    #def Sys_constraint9(vars):
    #    return vars[variable_labels.index('science_probe_diameter')]-vars[variable_labels.index('science_probe_length')]

    #Maintain cylindrical shape of relay probe
    #def Sys_constraint10(vars):
    #    return vars[variable_labels.index('relay_probe_diameter')]-vars[variable_labels.index('relay_probe_length')]

    #mass allocated to science should be greater than relay
    def Sys_constraint11(vars):
        return vars[variable_labels.index('relay_probe_mass')] - vars[variable_labels.index('science_probe_mass')]


    #diameter of both probes less than total? 

    def Test_constraint(vars):
        return vars[variable_labels.index('relay_probe_mass')] - 1000



    Sys_constraint_list = [Sys_constraint1,Sys_constraint5,Sys_constraint6,Sys_constraint11]

    #Sys_constraint_list = [Test_constraint]

    #Sys_constraint_list = [Sys_constraint1,Sys_constraint2,Sys_constraint3,Sys_constraint4,Sys_constraint5,
    #                       Sys_constraint6,Sys_constraint9,Sys_constraint10,
    #                       Sys_constraint11]

    #constraitn up to 6 still finds a result, lets's try 7 and 8
    #with 1-8, we get no results
    #with 1-6, 9&10 we get results... something going on with 7&8? Impossible?

    #And no result with constraint 11? Let's remove that and check again

    #Sys_constraint_list = [Sys_constraint1,Sys_constraint2,Sys_constraint3,Sys_constraint4,Sys_constraint5,
    #                       Sys_constraint6,Sys_constraint9,Sys_constraint10]

    #constraint 7&8 seem to result in nothing no matter what I do? Investigate...

    #TODO Ive got it... only have power to head or comms as a variable, and then make the other (max of combination) - power to head. 
    #Then remove 7&8... sorted?
    constraint_list+= Sys_constraint_list



    ####if r is 0.5 and l is 1 then 0.5/1 = 2 

    obj = DepthCalculation()

    n_threads = 4
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    class MyProblem(ElementwiseProblem):

        def __init__(self,*args,**kwargs):
            super().__init__(n_var = len(variable_labels),
                            n_obj = 2,
                            n_ieq_constr = len(constraint_list),
                            constr_ieq= constraint_list,
                            xl=np.array(variable_lower_lims),  # Lower bounds for x, y, and z
                            xu=np.array(variable_upper_lims),
                            elementwise_runner=runner)
        
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def CommAbove4(self,tavg,x_probe,x_surface,Freq,c_ice,salinity_interp,P_tx_Sci,eff_tx_Sci,G_tx_Sci,eff_rx_Sci,G_rx_Sci,k_boltz,T_sys_Sci,BW,target_SNR,temp0,salt0):
            #xcord,tavg,= BryantInterp(filename)

            #x_list = np.linspace(x_probe+0.001,x_surface,1000)
            #seperation = x_list[-1]-x_list[-2]
            #print(temp_profile_list[0](x_probe/1000))
            #print(salt_profile_list[0](x_probe/1000))
            icelossfactor = PathLossFactorBryant(temp0(x_probe/1000),Frequency,1,c_ice,salt0(x_probe/1000))

            SNR_preamble = (P_tx_Sci*eff_tx_Sci*G_tx_Sci*eff_rx_Sci*G_rx_Sci)/(k_boltz*T_sys_Sci*BW)
            #print(SNR_preamble)
            for x_current in range(int(x_probe)+1,int(x_surface)+1):
                temp_current = tavg(x_current/1000)
                sal_current = salinity_interp(x_current/1000)
                x_range = x_current-x_probe
                #print('x_range: ',x_range)
                fspl = (3e8/(math.pi*Freq*(x_range*1000)*4))**2
                loss_without_ice = 10*np.log(SNR_preamble)+10*np.log(fspl)
                ice_noise = 20*np.log(icelossfactor)
                #print('fspl: ', fspl)
                #print('iceloss: ', icelossfactor)
                total_loss_add = loss_without_ice + ice_noise
                #print('add: ', total_loss_add)
                total_loss = fspl*icelossfactor
                
                #print('SNR: ',SNR)
                SNR = loss_without_ice + ice_noise
                
                if SNR <= 0:
                    SNR_db = 100000
                else:
                    SNR_db = 10*math.log(SNR)
                    #print('snr_db: ',SNR_db)

                if SNR_db < target_SNR:
                    return(x_current)
                elif int(x_current) >= int(x_surface):
                    return(x_surface)
                else:
                    icelossfactor *= PathLossFactorBryant(temp_current,Frequency,1,c_ice,sal_current)


        
                
        def _evaluate(self, vars, out, *args, **kwargs):


            Mission_time_limit_d = 30    
            mass_Science_RTG = vars[variable_labels.index('science_probe_mass')]*vars[variable_labels.index('science_probe_RTG_ratio')]
            mass_Relay_RTG = vars[variable_labels.index('relay_probe_mass')]*vars[variable_labels.index('relay_probe_RTG_ratio')]

            mass_Science = vars[variable_labels.index('science_probe_mass')]
            mass_Relay = vars[variable_labels.index('relay_probe_mass')]
            
            sci_power_available = mass_Science_RTG*Specific_System_Power
            rel_power_available = mass_Relay_RTG*Specific_System_Power

            Frequency = vars[variable_labels.index('comm_frequency')]
            BW = vars[variable_labels.index('comm_bandwidth')]
            Target_SNR = vars[variable_labels.index('target_snr')]
            #For some reason I don't fully understand... calling melt_com_comb doesn't work here... switching to single number, 0.9
            P_tx_Sci = (0.9 - vars[variable_labels.index('sci_power_to_head')])*sci_power_available
            P_tx_Rel =  (0.9 - vars[variable_labels.index('rel_power_to_head')])*rel_power_available
            eff_tx_Sci = vars[variable_labels.index('sci_comm_eff')]
            G_tx_Sci = vars[variable_labels.index('sci_antenna_gain')]
            eff_rx_Sci = vars[variable_labels.index('sci_comm_eff')]
            G_rx_Sci = vars[variable_labels.index('sci_antenna_gain')]
            T_sys_Sci = vars[variable_labels.index('sci_sys_temp')]
            P_tx_Rel =  (0.9 - vars[variable_labels.index('rel_power_to_head')])*rel_power_available
            eff_tx_Rel = vars[variable_labels.index('rel_comm_eff')]
            G_tx_Rel = vars[variable_labels.index('rel_antenna_gain')]
            eff_rx_Rel = vars[variable_labels.index('rel_comm_eff')]
            G_rx_Rel = vars[variable_labels.index('rel_antenna_gain')]
            T_sys_Rel = vars[variable_labels.index('rel_sys_temp')]
            Power_mass_Science = vars[variable_labels.index('science_probe_RTG_ratio')]*vars[variable_labels.index('science_probe_mass')]
            #TODO: MAke this a seperate variable
            Melt_eff_Sci= 0.9
            R_Science = vars[variable_labels.index('science_probe_diameter')]/2
            l_Science = vars[variable_labels.index('science_probe_length')]
            
            #Probename_Relay = 
            #TODO: ix this, it might need to be a multiplier in the probedepth file?
            Power_mass_Relay = vars[variable_labels.index('relay_probe_RTG_ratio')]*vars[variable_labels.index('relay_probe_mass')]
            #TODO: MAke this a seperate variable
            Melt_eff_Relay = 0.9
            R_Relay = vars[variable_labels.index('relay_probe_diameter')]/2
            l_Relay = vars[variable_labels.index('relay_probe_length')]
            
            #Mission_Time_Limit_d = 
            
            Power_to_melt_Sci = sci_power_available*vars[variable_labels.index('sci_power_to_head')]
            Power_to_melt_Rel = rel_power_available*vars[variable_labels.index('sci_power_to_head')]

            Max_depth_list = []
            Trans_Time_list = []
            for j in range(len(x_grid_list)):
                result = obj.TwoDepth(
                            mission_configs, z_temp, z_rho, z_salt, index_list, x_grid,
                            Frequency, BW, Target_SNR, P_tx_Sci, eff_tx_Sci, G_tx_Sci, eff_rx_Sci, G_rx_Sci, T_sys_Sci,
                            P_tx_Rel, eff_tx_Rel, G_tx_Rel, eff_rx_Rel, G_rx_Rel, T_sys_Rel,
                            Power_to_melt_Sci, Melt_eff_Sci, R_Science, l_Science, mass_Science,
                            Power_to_melt_Rel, Melt_eff_Relay, R_Relay, l_Relay, mass_Relay)
        
            
                Max_depth_list.append(result[0])
                Trans_Time_list.append(result[1])
            
            f1 = np.mean(Max_depth_list)
            
            f2 = -np.mean(Trans_Time_list)*BW*np.log2(1+Target_SNR)
            #f2 = -result[1]

            g_list = []
            for i in range(len(constraint_list)):
                g_list.append(constraint_list[i](vars))


            out["F"] = [f1, f2]
            out["G"] = g_list

    #################
    import scipy.interpolate


    #n_pop = 25
    #n_gen = 5
    n_pop = 50
    n_gen = 5
    algorithm = NSGA2(pop_size=n_pop)# Choose an optimization algorithm from pymoo (e.g., NSGA-II, NSGA-III, etc.)
    problem = MyProblem(k_boltz,
        Specific_System_Power,
        slices,
        deliverable_mass,
        rho_lead,
        volume_limit,
        Melt_Comm_Combined_Power,
        Melt_Comm_Combined_Mass,
        diameter_limit,
        Mission_time_limit_d)

    result = minimize(problem,
                    algorithm,
                    termination=('n_gen', n_gen),  # Termination criterion (e.g., number of generations)
                    seed=2,
                    verbose = True,
                    return_least_infeasible = True,
                    save_history = True)  # Optional seed for reproducibility
    print('Threads:', result.exec_time)
    all_pop = Population()

    for algorithm in result.history:
        all_pop = Population.merge(all_pop, algorithm.off)
        
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = '/home/ec2-user/AWS Files/AWS Files/Moo_Outputs/{}_MOOStats_{}'.format(current_time,ice_filename)


    with open(filename, 'w') as file:
        file.write(str(ice_filename)+ '\n')
        file.write('Variable Labels:' + '\n')
        file.write(str(variable_labels) + '\n')
        file.write('Variable Lower Lims:'+ '\n')
        file.write(str(variable_lower_lims) + '\n')
        file.write('Variable Upper Lims:'+ '\n')
        file.write(str(variable_upper_lims) + '\n')
        file.write('n_gen,n_pop' + '\n')
        file.write(str(n_gen) + str(n_pop) + '\n')
        file.write('#####RESULTS#####' +'\n')
        file.write('Results X (Design Space Values)' + '\n')
        file.write(str(result.X)+ '\n')
        file.write('Results F (Objective Space Values)' + '\n')
        file.write(str(result.F)+ '\n')
        file.write('Results G (Constraint Values)' + '\n')
        file.write(str(result.G)+ '\n')
        file.write('Results cv (Aggregated Constraint Violation)' + '\n')
        file.write(str(result.CV)+ '\n')
        file.write('Results History' + '\n')
        file.write(str(all_pop.get("F"))+ '\n')
        
    return(filename)


#a = Single_Run('40km_Drho46_2D',18.52,'MOOSetupHeavy.yaml',100)
#print('Run 1 Completed')
print('starting on b')
b = Single_Run('40km_Drho5_2D',18.52,'MOOSetup.yaml',42.5)
print('Run 1 Completed')
c = Single_Run('40km_Drho46_2D',18.52,'MOOSetup.yaml',42.5)
print('Run 2 Completed')

print('All Runs Completed')
