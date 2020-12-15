 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sukrit, modifying sukrit and azra's past codes
"""
import numpy as np #This furnishes array operations.
import scipy as sp
from scipy import interpolate as interp
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.integrate
import pdb
import pandas as pd
#import cookbook.py as cookbook

cm2nm=1.0E+7 #1 cm in nm
hc=1.98645e-9 #value of h*c in erg*nm, useful to convert from ergs/cm2/s/nm to photons/cm2/s/nm

###Plotting switches.
plotBr=False
plotI=False
plotCl=False

plotFeBF42=False
plotferrocyanide=False
plotSNP_FeCl2_FeSO4=False

plot_halide_ferrous_ocean=False

plot_halide_freshwater_lake=True
plot_halide_carbonate_lake=True

plot_ferrous_lake=True


##############
###Define timescale functions
##############

def get_timescale(process, total_abs_abs, total_abs_wav, depth):
    """
    Takes:
        -name of process, options are nucleobase-photolysis, cytidine-deamination, 2AO-photolysis, 2AI-photolysis, 2AT-photolysis
        -total linear decadic absorption coefficient for the solution, in cm**-1, as a function of wavelength
        -Wavelengths corresponding to those absorption coefficients, in nm
        -Depth of the reservoir to be simulated, in cm.
    """
    if process=='nucleobase-photolysis':
        #Direct photodissociation of nucleobases, following the simplifying assumption of Cleaves et al. 1998, i.e. that nucleobase photolysis rate is proportional to the transmittance at 260 nm
        abs_260nm=np.interp(260., total_abs_wav, total_abs_abs) #Get decadic absorption coefficient at 260 nm.
        def transmittance(x, a):
            return 10.0**(-a*x/0.5) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        timescale_enhancement=depth/scipy.integrate.quad(transmittance, 0, depth, args=(abs_260nm), epsabs=0, epsrel=1e-3,limit=1000)[0]
        
    if process=='cytidine-deamination':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='C-to-U') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['C to U']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
    if process=='2AO-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AO']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
    
    if process=='2AI-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AI']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
  
    if process=='2AT-photolysis':
        df=pd.read_excel('./todd_data_for_sukrit.xlsx', sheet_name='Aminoazoles') 
        todd_wav_centers=np.array(df['Wavelength (nm)']) #nm
        todd_rates=np.array(df['AT']) #rel rate (min^-1 cm^-2 -->???)
        
        ###From paper: bin widths are 10 nm, 5 nm either side.
        todd_wav_left=todd_wav_centers-5.0
        todd_wav_right=todd_wav_centers+5.0
        
        ###Densify the rate data as a function of wavelength, to enable
        todd_wav_dense=np.linspace(todd_wav_left[0], todd_wav_right[-1], num=1000, endpoint=True)#new, denser wavelength grid
        todd_rates_dense=np.zeros(np.shape(todd_wav_dense))
        for ind in range(0, len(todd_rates)):
            wav_inds=np.where((todd_wav_dense>=todd_wav_left[ind]) & (todd_wav_dense<=todd_wav_right[ind]))
            todd_rates_dense[wav_inds]=np.ones(np.shape(wav_inds))*todd_rates[ind]
        def integrand(x, wav): #main integral, actual solution
            a=np.interp(wav, total_abs_wav, total_abs_abs)
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return ((10.0**(-a*x/0.5))*phi) #factor of /0.5 is /cos(theta), with theta=60 deg chosen for consistency with 2-stream diffuse stream and Rugheimer et al. 2015 solar zenith angle. 
        def integrand_1d(wav): #surface only
            phi=np.interp(wav, todd_wav_dense, todd_rates_dense)
            return phi
        
        timescale_enhancement=(depth*scipy.integrate.quad(integrand_1d, todd_wav_left[0], todd_wav_right[-1], epsabs=0, epsrel=1e-3, limit=1000)[0])/(scipy.integrate.dblquad(integrand, todd_wav_left[0], todd_wav_right[-1], lambda wav: 0, lambda wav: depth, epsabs=0, epsrel=1e-3)[0])
     
    return timescale_enhancement

##############
###Load TOA irradiance, surface intensity from Ranjan & Sasselov 2017, specifically when we matched Rugheimer et al. 2015 exactly. 
##############
uv_wav, uv_toa_intensity, uv_surface_flux, uv_surface_intensity=np.genfromtxt('./rugheimer_earth_epoch0.dat', skip_header=1, skip_footer=0, usecols=(2,3,4,6), unpack=True) #From Ranjan & Sasselov 2017, the GitHub. 

###Convert from ergs to photons
uv_toa_intensity *= uv_wav/hc
uv_surface_flux *= uv_wav/hc
uv_surface_intensity *= uv_wav/hc

###Restrict data to 190 to 310 nm
inds=np.where((uv_wav>=190.0) & (uv_wav <=310.0))

uv_wav=uv_wav[inds]
uv_toa_intensity=uv_toa_intensity[inds]
uv_surface_flux=uv_surface_flux[inds]
uv_surface_intensity=uv_surface_intensity[inds]

##############
###Load data
##############
data_wav={} #dict to hold wavelength scale for measured molar absorptivities (nm)
data_abs={} #dict to hold molar absorptivities for measured molar absorptivities (M**-1 cm**-1.)


###
#Lit Data
###

###Pure water
h2o_quickenden_wav, h2o_quickenden_abs = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/quickenden.dat', skip_header=2, unpack=True, usecols=(0,1))#nm, cm**-1. Includes both scattering, absorption. 

###Modern oceans: (Smith & Baker) Warning, guessing <300 nm.
pureoceans_smith1981_wav, pureoceans_smith1981_abs=np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/smithbaker_purest.dat', skip_header=2, skip_footer=0, unpack=True, usecols=(0,1)) #purest modern natural water; nm, cm**-1.

###Modern oceans: (Morel+2007)
pureoceans_morel_wav=np.array([300., 305., 310., 315., 320., 325., 330., 335., 340., 345., 350., 355., 360., 365., 370., 375., 380., 385., 390., 395.])
pureoceans_morel_abs=(np.array([0.01150, 0.0110, 0.01050, 0.01013, 0.00975, 0.00926, 0.00877, 0.00836, 0.00794, 0.00753, 0.00712, 0.00684, 0.00656, 0.00629, 0.00602, 0.00561, 0.00520, 0.00499, 0.00478, 0.00469]) + 0.5*np.array([0.0226, 0.0211, 0.0197, 0.0185, 0.0173, 0.0162, 0.0152, 0.0144, 0.0135, 0.0127, 0.0121, 0.0113, 0.0107, 0.0099, 0.0095, 0.0089, 0.0085, 0.0081, 0.0077, 0.0072])) * 0.01 #K_dsw2, converted from m**-1 to cm**-1.

###From Azra's work.
data_wav['Br_azra'], data_abs['Br_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/johnson_br-truncated.dat', skip_header=2, unpack=True, usecols=(0,1))# Br-. Truncated to remove data >220 nm, which looks unreliable to me. 
data_wav['I_azra'], data_abs['I_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/guenther_i.dat', skip_header=2, unpack=True, usecols=(0,1)) #I-
data_wav['NO3_azra'], data_abs['NO3_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/mack_no3.dat', skip_header=2, unpack=True, usecols=(0,1)) #NO3-
data_wav['Fe(BF4)2_azra'], data_abs['Fe(BF4)2_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_fe2bf4.dat', skip_header=2, unpack=True, usecols=(0,1)) #Fe(BF4)2
data_wav['FeCl2_azra'], data_abs['FeCl2_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_fecl2.dat', skip_header=2, unpack=True, usecols=(0,1)) #FeCl2
data_wav['FeSO4_azra'], data_abs['FeSO4_azra']= np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/fontana_feso4.dat', skip_header=2, unpack=True, usecols=(0,1)) #FeSO4
data_wav['gelbstoff_azra'], data_abs['gelbstoff_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/cleaves_gelbstoff.dat', skip_header=2, unpack=True, usecols=(0,1))#organic gunk
data_wav['KCl_azra'], data_abs['KCl_azra'] = np.genfromtxt('./Azra_Project/RSI_UV/Processed-Data/perkampus_kcl.dat', skip_header=2, unpack=True, usecols=(0,1))#organic gunk

###From Birkmann+2018
df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='Br-I-Cl', skiprows=0)
#pdb.set_trace()
data_wav['birk_BrICl']=df['Wavelength (nm)']
data_abs['birk_Br']=df['Br- (M**-1 cm**-1)']
data_abs['birk_I']=df['I- (M**-1 cm**-1)']
data_abs['birk_Cl']=df['Cl- (M**-1 cm**-1)']
#df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='NO3-OH', skiprow=0)
#data_wav['birk_NO3OH']=df['Wavelength (nm)']
#data_abs['birk_NO3']=df['NO3- (M**-1 cm**-1)']
#data_abs['birk_OH']=df['OH- (M**-1 cm**-1)']
#df=pd.read_excel('./birkmann_2018_data.xlsx', sheet_name='SO4-HCO3-CO3', skiprow=0)
#data_wav['birk_SO4HCO3CO3']=df['Wavelength (nm)']
#data_abs['birk_SO4']=df['SO4(2-) (M**-1 cm**-1)']
#data_abs['birk_HCO3']=df['HCO3- (M**-1 cm**-1)']
#data_abs['birk_CO3']=df['CO3(2-) (M**-1 cm**-1)']


###From Strizhakov+2014, for sodium nitroprusside. Need to concatenate shortwave (log scale) and longwave (linear scale) data here. 

strizhakov_snp_shortwav_wav, strizhakov_snp_shortwav_abs=np.genfromtxt('./strizhakov_fig4_snp_shortwave_Dataset.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

strizhakov_snp_longwav_wav, strizhakov_snp_longwav_abs=np.genfromtxt('./strizhakov_fig4_snp_longwave_Dataset.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#nm,  M**-1 cm**-1

data_wav['strizhakov_SNP']=np.concatenate((strizhakov_snp_shortwav_wav, strizhakov_snp_longwav_wav))
data_abs['strizhakov_SNP']=np.concatenate((strizhakov_snp_shortwav_abs, strizhakov_snp_longwav_abs))

###Let's get ferrocyanide and ferricyanide in on the party.
hc_eVnm=1240. #h*c in eV*nm

##Ferrocyanide from Ross+2018
wavthunk, absthunk = np.genfromtxt('./ross_fig3_ferrocyanide.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#eV,  M**-1 cm**-1
data_wav['ross_ferrocyanide']=np.flip(hc_eVnm/wavthunk) #convert eV to nm
data_abs['ross_ferrocyanide']=np.flip(absthunk)


##Ferricyanide from Ross+2018
wavthunk, absthunk = np.genfromtxt('./ross_fig3_ferricyanide.csv', skip_header=1, unpack=True, usecols=(0,1), delimiter=',')#eV,  M**-1 cm**-1
data_wav['ross_ferricyanide']=np.flip(hc_eVnm/wavthunk) #convert eV to nm
data_abs['ross_ferricyanide']=np.flip(absthunk)


###
#Our data
###

###From Gabi/Corinna's work
df=pd.read_excel('./2020-08-23_Extinktion-coefficients-Sukrit-V7_mod.xlsx', sheet_name='Tabelle1') 
data_wav['LK']=np.nan_to_num(df['Wavelength']) #nm LK=Lozano-Kufner
data_abs['NaBr_LK']=np.nan_to_num(df['NaBr'])
data_abs['NaBr_err_LK']=np.nan_to_num(df['NaBr-error'])

data_abs['KBr_LK']=np.nan_to_num(df['KBr'])
data_abs['KBr_err_LK']=np.nan_to_num(df['KBr-error'])

data_abs['NaHCO3_LK']=np.nan_to_num(df['NaHCO3'])
data_abs['NaHCO3_err_LK']=np.nan_to_num(df['NaHCO3-error'])

data_abs['NaCl_LK']=np.nan_to_num(df['NaCl'])
data_abs['NaCl_err_LK']=np.nan_to_num(df['NaCl-error'])

#data_abs['KCl_LK']=np.nan_to_num(df['KCl'])
#data_abs['KCl_err_LK']=np.nan_to_num(df['KCl-error'])

data_abs['Fe(BF4)2_LK']=np.nan_to_num(df['Fe(BF4)2'])
data_abs['Fe(BF4)2_err_LK']=np.nan_to_num(df['Fe(BF4)2-error'])

data_abs['K_ferrocyanide_LK']=np.nan_to_num(df['K4[Fe(CN)6]'])
data_abs['K_ferrocyanide_err_LK']=np.nan_to_num(df['K4[Fe(CN)6]-error'])

data_abs['NaI_LK']=np.nan_to_num(df['NaI'])
data_abs['NaI_err_LK']=np.nan_to_num(df['NaI-error'])

data_abs['KI_LK']=np.nan_to_num(df['KI'])
data_abs['KI_err_LK']=np.nan_to_num(df['KI-error'])





########################
###Synthesize composite data
########################
###Splice the I data
data_abs['I_LK']=np.copy(data_abs['NaI_LK'])
data_abs['I_LK'][np.where(data_wav['LK']>260.0)]=np.copy(data_abs['KI_LK'][np.where(data_wav['LK']>260.0)])

data_abs['I_err_LK']=np.copy(data_abs['NaI_err_LK'])
data_abs['I_err_LK'][np.where(data_wav['LK']>260.0)]=np.copy(data_abs['KI_err_LK'][np.where(data_wav['LK']>260.0)])
########################
###Quality of input data?
########################
if plotBr:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'Br$^-$')    
    ax.plot(data_wav['Br_azra'], data_abs['Br_azra'] , linewidth=1, linestyle='--', marker='d', color='blue', label=r'Br$^-$ (Johnson+2002')
    ax.plot(data_wav['birk_BrICl'], data_abs['birk_Br'] , linewidth=1, linestyle='--', marker='d', color='red', label=r'Br$^-$ (Birkmann+2018)')
    ax.errorbar(data_wav['LK'], data_abs['KBr_LK'] , yerr=data_abs['KBr_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KBr (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaBr_LK'], yerr=data_abs['NaBr_err_LK'],linewidth=2, linestyle='-', marker='o', color='black', label=r'NaBr (This Work)', capsize=5)
    
    # ###Plot the upper limit on absorption
    # nabr_wav_inds=np.squeeze(np.where(data_abs['NaBr_LK']==0.0))
    # kbr_wav_inds=np.squeeze(np.where(data_abs['KBr_LK']==0.0))
    
    # nabr_upperlim=data_abs['NaBr_LK'][nabr_wav_inds[0]-1] + 1.0*data_abs['NaBr_err_LK'][nabr_wav_inds[0]-1]
    # kbr_upperlim=data_abs['KBr_LK'][kbr_wav_inds[0]-1] + 1.0*data_abs['KBr_err_LK'][kbr_wav_inds[0]-1]

    # ax.plot(data_wav['LK'][nabr_wav_inds], np.ones(len(nabr_wav_inds))*nabr_upperlim, linestyle=':', color='black')
    # ax.plot(data_wav['LK'][kbr_wav_inds], np.ones(len(kbr_wav_inds))*kbr_upperlim, linestyle=':', color='green')

    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.set_ylim([1.0E-1, 1.0E4])
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Brm.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Brm.jpg', orientation='portrait', format='jpg')
  
if plotI:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'I$^-$')    
    ax.plot(data_wav['I_azra'], data_abs['I_azra'] , linewidth=1, linestyle='--', marker='d', color='blue', label=r'I$^-$ (Guenther+2001')
    ax.plot(data_wav['birk_BrICl'], data_abs['birk_I'] , linewidth=1, linestyle='--', marker='d', color='red', label=r'I$^-$ (Birkmann+2018)')
    ax.errorbar(data_wav['LK'], data_abs['KI_LK'], yerr=data_abs['KI_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KI (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaI_LK'], yerr=data_abs['NaI_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'NaI (This Work)', capsize=5)
    # ax.plot(data_wav['LK'], data_abs['I_LK'], linewidth=5, linestyle='-', color='purple', label='I$^-$ (this work)')
    ax.errorbar(data_wav['LK'], data_abs['I_LK'], yerr=0, linewidth=1.5, linestyle='-', color='purple', label='I$^-$ (this work)')
    
    # ###Plot the upper limit on absorption
    # i_wav_inds=np.squeeze(np.where(data_abs['I_LK']==0.0))
    
    # i_upperlim=data_abs['I_LK'][i_wav_inds[0]-1] + 1.0*data_abs['I_err_LK'][i_wav_inds[0]-1]

    # ax.plot(data_wav['LK'][i_wav_inds], np.ones(len(i_wav_inds))*i_upperlim, linestyle=':', color='purple')
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Im.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Im.jpg', orientation='portrait', format='jpg')
    
if plotCl:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'Cl$^-$')    
    ax.plot(data_wav['KCl_azra'], data_abs['KCl_azra'] , linewidth=1, linestyle='--', marker='d', color='blue', label=r'KCl (Perkampus+1992)')
    ax.plot(data_wav['birk_BrICl'], data_abs['birk_Cl'] , linewidth=1, linestyle='--', marker='d', color='red', label=r'Cl$^-$ (Birkmann+2018)')
#    ax.errorbar(data_wav['LK'], data_abs['KCl_LK'], yerr=data_abs['KCl_err_LK'], linewidth=2, linestyle='-', marker='o', color='green', label=r'KCl (This Work)', capsize=5)
    ax.errorbar(data_wav['LK'], data_abs['NaCl_LK'], yerr=data_abs['NaCl_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'NaCl (This Work)', capsize=5)
    
    
    # ###Plot the upper limit on absorption
    # nacl_wav_inds=np.squeeze(np.where(data_abs['NaCl_LK']==0.0))
    
    # nacl_upperlim=data_abs['NaCl_LK'][nacl_wav_inds[0]-1] + 1.0*data_abs['NaCl_err_LK'][nacl_wav_inds[0]-1]

    # ax.plot(data_wav['LK'][nacl_wav_inds], np.ones(len(nacl_wav_inds))*nacl_upperlim, linestyle=':', color='black')

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/Clm.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/Clm.jpg', orientation='portrait', format='jpg')

    
if plotFeBF42:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'Fe(BF$_4$)$_2$')    
    ax.errorbar(data_wav['LK'], data_abs['Fe(BF4)2_LK'], data_abs['Fe(BF4)2_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'Fe(BF$_4$)$_2$ (This Work)', capsize=5)

    ax.errorbar(data_wav['Fe(BF4)2_azra'], data_abs['Fe(BF4)2_azra'] , yerr=0, linewidth=1, linestyle='--', marker='d', color='blue', label=r'Fe(BF$_4$)$_2$ (Fontana+2007)')

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/FeBF42.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/FeBF42.jpg', orientation='portrait', format='jpg')            

if plotSNP_FeCl2_FeSO4: 
    #We need to do.
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'Fe(II) Compounds From Literature')    
    ax.plot(data_wav['strizhakov_SNP'], data_abs['strizhakov_SNP'] , linewidth=2, linestyle='-', marker='d', color='blue', label=r'Na$_2$Fe(CN)$_5$NO (Strizhakov+2014)')
    ax.plot(data_wav['FeCl2_azra'], data_abs['FeCl2_azra'] , linewidth=2, linestyle='-', marker='d', color='green', label=r'FeCl$_2$ (Fontana+2007)')
    ax.plot(data_wav['FeSO4_azra'], data_abs['FeSO4_azra'] , linewidth=2, linestyle='-', marker='d', color='red', label=r'FeSO$_4$ (Fontana+2007)')

    ax.errorbar(data_wav['LK'], data_abs['K_ferrocyanide_LK'],yerr=data_abs['K_ferrocyanide_err_LK'], linewidth=1, linestyle='--', color='purple', label=r'K$_4$Fe(CN)$_6$ (This Work)',capsize=2)
    ax.errorbar(data_wav['LK'], data_abs['Fe(BF4)2_LK'], data_abs['Fe(BF4)2_err_LK'], linewidth=1, linestyle='--', color='black', label=r'Fe(BF$_4$)$_2$ (This Work)', capsize=1)

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
#    ax.set_ylim([1E-1, 1.0E0])
    
    plt.savefig('./Plots/molecules/SNP_etc.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/SNP.jpg', orientation='portrait', format='jpg')
    
    
if plotferrocyanide: 
    #We need to do.
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)

    ax.set_title(r'Ferrocyanide')    
    ax.errorbar(data_wav['LK'], data_abs['K_ferrocyanide_LK'],yerr=data_abs['K_ferrocyanide_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label=r'K$_4$Fe(CN)$_6$ (This Work)',capsize=5)

    ax.errorbar(data_wav['ross_ferrocyanide'], data_abs['ross_ferrocyanide'], yerr=0, linewidth=1, linestyle='-', marker='d', color='red', label=r'K$_4$Fe(CN)$_6$ (Ross+2018)')
    ax.plot(data_wav['ross_ferricyanide'], data_abs['ross_ferricyanide'] , linewidth=1, linestyle='--', color='blue', label=r'K$_3$Fe(CN)$_6$ (Ross+2018)')
    
#    ax.plot(data_wav['strizhakov_SNP'], data_abs['strizhakov_SNP'] , linewidth=2, linestyle='-', marker='d', color='green', label=r'Sodium Nitroprusside (Strizhakov+2014)')

    
    ax.set_yscale('log')
    ax.set_ylabel(r'Molar Decadic Absorption Coefficient (M$^{-1}$cm$^{-1}$)')
    ax.legend(ncol=1, loc='best')    
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([200., 300.])
#    ax.set_ylim([1E-1, 1.0E0])
    
    plt.savefig('./Plots/molecules/ferrocyanide.pdf', orientation='portrait', format='pdf')
#    plt.savefig('./Plots/molecules/ferrocyanide.jpg', orientation='portrait', format='jpg')


if plot_halide_ferrous_ocean:
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    
    conc_NaCl_mod=0.6 #M
    conc_NaBr_mod=0.9*1.0E-3 #M
    conc_NaI_mod=0.5*1.0E-6 #M
    
    conc_fe_konhauser_zheng=80.0E-6 #M. Via Konhauser+2017, Zheng+2018 (not published)
    conc_fe_halevy=1.0E-9 #M. Halevy+2017, green rust paper. Big difference due to photooxidation -- but views may have evolved? Ask Xie etc?
    
    halide_low=0.5# lower estimate of salinity
    halide_high=10.0#upper estimate of salinity
    
    ocean_low=halide_low*(conc_NaCl_mod*data_abs['NaCl_LK']+conc_NaBr_mod*data_abs['NaBr_LK']+conc_NaI_mod*data_abs['I_LK']) + conc_fe_halevy*data_abs['Fe(BF4)2_LK']
    ocean_low_err=np.sqrt((conc_fe_halevy*data_abs['Fe(BF4)2_err_LK'])**2.0 + (halide_low**2.0)*((conc_NaCl_mod*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_mod*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_mod*data_abs['I_err_LK']))**2.0)
    
    ocean_high=halide_high*(conc_NaCl_mod*data_abs['NaCl_LK']+conc_NaBr_mod*data_abs['NaBr_LK']+conc_NaI_mod*data_abs['I_LK']) + conc_fe_konhauser_zheng*data_abs['Fe(BF4)2_LK']
    ocean_high_err=np.sqrt((conc_fe_konhauser_zheng*data_abs['Fe(BF4)2_err_LK'])**2.0 + (halide_high**2.0)*((conc_NaCl_mod*data_abs['NaCl_err_LK'])**2.0 + (conc_NaBr_mod*data_abs['NaBr_err_LK'])**2.0 + (conc_NaI_mod*data_abs['I_err_LK']))**2.0)


    ax.errorbar(data_wav['LK'], ocean_low, yerr=ocean_low_err, linewidth=2, linestyle=':', marker='o', color='darkgrey', label=r'Ocean, Lower Limit',capsize=3)
    ax.plot(data_wav['LK'], halide_low*conc_NaCl_mod*data_abs['NaCl_LK'], linewidth=2, linestyle=':', color='purple', label=r'Cl$^-$, Low')
    ax.plot(data_wav['LK'], halide_low*conc_NaBr_mod*data_abs['NaBr_LK'], linewidth=2, linestyle=':', color='blue', label=r'Br$^-$, Low')
    ax.plot(data_wav['LK'], halide_low*conc_NaI_mod*data_abs['I_LK'], linewidth=2, linestyle=':', color='green', label=r'I$^-$, Low')
    ax.plot(data_wav['LK'], conc_fe_halevy*data_abs['Fe(BF4)2_LK'], linewidth=2, linestyle=':', color='red', label=r'Fe$^{2+}$, Low')

 
    ax.errorbar(data_wav['LK'], ocean_high, yerr=ocean_high_err, linewidth=2, linestyle='-', marker='o', color='black', label=r'Ocean, Higher Limit',capsize=3)
    ax.plot(data_wav['LK'], halide_high*conc_NaCl_mod*data_abs['NaCl_LK'], linewidth=2, linestyle='-', color='purple', label=r'Cl$^-$, High')
    ax.plot(data_wav['LK'], halide_high*conc_NaBr_mod*data_abs['NaBr_LK'], linewidth=2, linestyle='-', color='blue', label=r'Br$^-$, High')
    ax.plot(data_wav['LK'], halide_high*conc_NaI_mod*data_abs['I_LK'], linewidth=2, linestyle='-', color='green', label=r'I$^-$, High')
    ax.plot(data_wav['LK'], conc_fe_konhauser_zheng*data_abs['Fe(BF4)2_LK'], linewidth=2, linestyle='-', color='red', label=r'Fe$^{2+}$, High')

    
    ax.axhline(1.0, color='black', linestyle=':')
    ax.axhline(1.0e-1, color='black', linestyle=':')
    ax.axhline(1.0e-2, color='black', linestyle=':')
    
    ax.legend(ncol=1, loc='best')
    ax.set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')
            
    ax.set_yscale('log')
    ax.set_ylim([1E-3, 1.0E1]) 
    ax.set_xlim([200., 300.])
    
    plt.savefig('./Plots/molecules/halide_ferrous_ocean.pdf', orientation='portrait', format='pdf')

#Plot prebiotic lake (freshwater)
if plot_halide_freshwater_lake:
    #These are all mean values
    conc_NaCl=0.2*1.0E-3 #M
    conc_NaBr=0.15*1.0E-6 #M
    conc_NaI=40.*1.0E-9 #M
    conc_NaI_high=0.6*1E-6 #M
    
    freshwater_lake=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI*data_abs['I_LK']
    freshwater_lake_err=conc_NaCl*data_abs['NaCl_err_LK'] + conc_NaBr*data_abs['NaBr_err_LK'] + conc_NaI*data_abs['I_err_LK']
    
    freshwater_lake_highI=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI_high*data_abs['I_LK']
    freshwater_lake_highI_err=conc_NaCl*data_abs['NaCl_err_LK'] + conc_NaBr*data_abs['NaBr_err_LK'] + conc_NaI_high*data_abs['I_err_LK']
    
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Freshwater Lake')
    ax.errorbar(data_wav['LK'], freshwater_lake, yerr=freshwater_lake_err, linewidth=2, linestyle='-', marker='o', color='black', label='Cl$^-$ + Br$^-$ + I$^-$ (Average Lake)')
    ax.errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color='blue', label='Cl$^-$')
    ax.errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color='green', label='Br$^-$')
    ax.errorbar(data_wav['LK'], conc_NaI*data_abs['I_LK'], yerr=conc_NaI*data_abs['I_err_LK'], linewidth=1, linestyle='--', color='red', label='I$^-$ (average)')
    
    ax.errorbar(data_wav['LK'], freshwater_lake_highI, yerr=freshwater_lake_highI_err, linewidth=2, linestyle='-', marker='o', color='purple', label='Cl$^-$ + Br$^-$ + I$^-$ (High-I Lake)')
    ax.errorbar(data_wav['LK'], conc_NaI_high*data_abs['I_LK'], yerr=conc_NaI_high*data_abs['I_err_LK'], linewidth=1, linestyle=':', color='red', label='I$^-$ (high-I)')

    
    ax.legend(ncol=1, loc='best')
    ax.set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylim([1E-4, 1.0E-1]) 
    ax.set_xlim([200., 300.])
    
    ax.axhline(1.0E1, color='black', linestyle=':')
    ax.axhline(1.0E0, color='black', linestyle=':')
    ax.axhline(1.0E-1, color='black', linestyle=':')
    ax.axhline(1.0E-2, color='black', linestyle=':')
    ax.axhline(1.0E-3, color='black', linestyle=':')
    
    plt.savefig('./Plots/molecules/halide-freshwater-lake.pdf', orientation='portrait', format='pdf')
    print('Freshwater lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    print('Freshwater lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', freshwater_lake, data_wav['LK'], 100.0)))
    print('Freshwater lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    print('Freshwater lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    print('Freshwater lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', freshwater_lake, data_wav['LK'], 100.0)))
    print()
    print('High-I Freshwater lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', freshwater_lake_highI, data_wav['LK'], 100.0)))
    print('High-I Freshwater lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', freshwater_lake_highI, data_wav['LK'], 100.0)))
    print('High-I Freshwater lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', freshwater_lake_highI, data_wav['LK'], 100.0)))
    print('High-I Freshwater lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', freshwater_lake_highI, data_wav['LK'], 100.0)))
    print('High-I Freshwater lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', freshwater_lake_highI, data_wav['LK'], 100.0)))
    print()

    
#Plot prebiotic lake (closed-basin carbonate/phosphate rich)
if plot_halide_carbonate_lake:
    #These are all mean values
    conc_NaCl=1.0E-1 #M
    conc_NaBr=1.0E-3 #M
    conc_NaI=40.*1.0E-9#0.6*1.0E-6#40.*1.0E-9 #M
    
    conc_NaCl_high=6.0#M
    conc_NaBr_high=1.0E-2 #M
    conc_NaI_high=0.6*1E-6 #M
    
    carbonate_lake=conc_NaCl*data_abs['NaCl_LK'] + conc_NaBr*data_abs['NaBr_LK'] + conc_NaI*data_abs['I_LK']
    carbonate_lake_err=conc_NaCl*data_abs['NaCl_err_LK'] + conc_NaBr*data_abs['NaBr_err_LK'] + conc_NaI*data_abs['I_err_LK']
    
    carbonate_lake_high=conc_NaCl_high*data_abs['NaCl_LK'] + conc_NaBr_high*data_abs['NaBr_LK'] + conc_NaI_high*data_abs['I_LK']
    carbonate_lake_high_err=conc_NaCl_high*data_abs['NaCl_err_LK'] + conc_NaBr_high*data_abs['NaBr_err_LK'] + conc_NaI_high*data_abs['I_err_LK']
    
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Carbonate Lake')
    ax.errorbar(data_wav['LK'], carbonate_lake, yerr=carbonate_lake_err, linewidth=2, linestyle='-', marker='o', color='black', label='Cl$^-$ + Br$^-$ + I$^-$ (Low)')
    ax.errorbar(data_wav['LK'], conc_NaCl*data_abs['NaCl_LK'], yerr=conc_NaCl*data_abs['NaCl_err_LK'], linewidth=1, linestyle='--', color='blue', label='Cl$^-$ (Low)')
    ax.errorbar(data_wav['LK'], conc_NaBr*data_abs['NaBr_LK'], yerr=conc_NaBr*data_abs['NaBr_err_LK'], linewidth=1, linestyle='--', color='green', label='Br$^-$ (Low)')
    ax.errorbar(data_wav['LK'], conc_NaI*data_abs['I_LK'], yerr=conc_NaI*data_abs['I_err_LK'], linewidth=1, linestyle='--', color='red', label='I$^-$ (Low)')
    
    ax.errorbar(data_wav['LK'], carbonate_lake_high, yerr=carbonate_lake_high_err, linewidth=2, linestyle='--', marker='o', color='black', label='Cl$^-$ + Br$^-$ + I$^-$ (High)')
    ax.errorbar(data_wav['LK'], conc_NaCl_high*data_abs['NaCl_LK'], yerr=conc_NaCl_high*data_abs['NaCl_err_LK'], linewidth=1, linestyle=':', color='blue', label='Cl$^-$ (High)')
    ax.errorbar(data_wav['LK'], conc_NaBr_high*data_abs['NaBr_LK'], yerr=conc_NaBr_high*data_abs['NaBr_err_LK'], linewidth=1, linestyle=':', color='green', label='Br$^-$ (High)')
    ax.errorbar(data_wav['LK'], conc_NaI_high*data_abs['I_LK'], yerr=conc_NaI_high*data_abs['I_err_LK'], linewidth=1, linestyle=':', color='red', label='I$^-$ (High)')
    
    ax.legend(ncol=1, loc='best')
    ax.set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')
    
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylim([1E-4, 1.0E2]) 
    ax.set_xlim([200., 300.])
    
    ax.axhline(1.0E1, color='black', linestyle=':')
    ax.axhline(1.0E0, color='black', linestyle=':')
    ax.axhline(1.0E-1, color='black', linestyle=':')
    ax.axhline(1.0E-2, color='black', linestyle=':')
    ax.axhline(1.0E-3, color='black', linestyle=':')

    
    plt.savefig('./Plots/molecules/halide-carbonate-lake.pdf', orientation='portrait', format='pdf')

    ######
    #Plot UV in simulated reservoir
    ######
    ##Project pure water absorption to relevant wavelength scale, superpose
    carbonate_lake_withwater=np.interp(uv_wav, data_wav['LK'], carbonate_lake, left=0, right=0) + np.interp(uv_wav, h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0) #set absorptions to 0 where therem is no data; completely unphysical, but we shouldn't (and don't need to) trust those data anyway. Better to know clearly when things are full of shit. 
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Upper Bounds on Carbonate Lake UV')
    ax.plot(uv_wav, uv_toa_intensity, linewidth=2, linestyle='-', marker='o', color='black', label='Stellar Irradiation')

    ax.plot(uv_wav, uv_surface_intensity, linewidth=2, linestyle='-', marker='o', color='purple', label='Surface')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_withwater*1.0/0.5), linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_withwater*10.0/0.5), linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-carbonate_lake_withwater*100.0/0.5), linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')
    
    ax.legend(ncol=1, loc='best')
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylabel(r'Intensity (photons cm$^{2}$ s$^{-1}$ nm$^{-1}$')
    ax.set_ylim([1E8, 1.0E14]) 
    ax.set_xlim([200., 300.])
    plt.savefig('./Plots/molecules/uv_halide-carbonate-lake.pdf', orientation='portrait', format='pdf')
    
    ####
    ##
    ####
    print('Carbonate lake, nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    print('Carbonate lake, cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', carbonate_lake, data_wav['LK'], 100.0)))
    print('Carbonate lake, 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    print('Carbonate lake, 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    print('Carbonate lake, 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', carbonate_lake, data_wav['LK'], 100.0)))
    print()
    print('Carbonate lake (High), nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', carbonate_lake_high, data_wav['LK'], 100.0)))
    print('Carbonate lake (High), cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', carbonate_lake_high, data_wav['LK'], 100.0)))
    print('Carbonate lake (High), 2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', carbonate_lake_high, data_wav['LK'], 100.0)))
    print('Carbonate lake (High), 2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', carbonate_lake_high, data_wav['LK'], 100.0)))
    print('Carbonate lake (High), 2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', carbonate_lake_high, data_wav['LK'], 100.0)))
    print()
    

    
#########
#########
#########
    
if plot_ferrous_lake:
    conc_fe_halevy=100.0*1.0E-9 #M; riverine "estimate" from Halevy+2017 green rust
    conc_fe_hao_toner=100.0*1.0E-6 #M. Hao+2017 and Toner+2019 take this, probably following Hao+2017. Toner+2019 state that it would be Fe2+ for pH < 9 and Ferrocyanide for pH > 9, so this corresponds to non-basic conditions
    # 29% yield on the Toner ferrocyanide, and assuming 29% yield. 
    
    

    fig, ax=plt.subplots(2, figsize=(8., 10.), sharex=True, sharey=False)
    ax[0].set_title('[Fe(II)]$_0$=0.1 mM')
    ax[0].errorbar(data_wav['LK'], conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_hao_toner*data_abs['Fe(BF4)2_err_LK'], linewidth=2, linestyle='-', marker='o', color='black', label='100 $\mu$M Fe$^{2+}$ (Hao+2017)')
    ax[0].plot(data_wav['FeCl2_azra'], conc_fe_hao_toner*data_abs['FeCl2_azra'], linewidth=2, linestyle='-', marker='o', color='green', label='100 $\mu$M FeCl$_2$')
    ax[0].plot(data_wav['FeSO4_azra'], conc_fe_hao_toner*data_abs['FeSO4_azra'], linewidth=2, linestyle='-', marker='o', color='red', label='100 $\mu$M FeSO$_4$')  
    
    ax[0].errorbar(data_wav['LK'], conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], yerr=conc_fe_hao_toner*data_abs['K_ferrocyanide_err_LK'], linewidth=2, linestyle='-', marker='o', color='purple', label='100 $\mu$M K$_4$Fe(CN)$_6$')
    ax[0].plot(data_wav['strizhakov_SNP'], 0.29*conc_fe_hao_toner*data_abs['strizhakov_SNP'], linewidth=2, linestyle='-', marker='o', color='blue', label='29 $\mu$M Na$_2$Fe(CN)$_5$NO')

  
    ax[0].legend(ncol=1, loc='upper right')
    ax[0].set_ylabel(r'Linear Decadic Absorption (cm$^{-1}$)')    
    ax[0].set_yscale('log')
    ax[0].set_ylim([1E-3, 1.0E1]) 


    ax[1].set_title(r'[Fe(II)]$_0$=0.1 $\mu$M')
    ax[1].errorbar(data_wav['LK'], conc_fe_halevy*data_abs['Fe(BF4)2_LK'], yerr=conc_fe_halevy*data_abs['Fe(BF4)2_err_LK'], linewidth=2, linestyle='--', marker='o', color='black', label='100 nM Fe$^{2+}$ (Hao+2017)')
    ax[1].plot(data_wav['FeCl2_azra'], conc_fe_halevy*data_abs['FeCl2_azra'], linewidth=2, linestyle='--', marker='o', color='green', label='100 nM FeCl$_2$')
    ax[1].plot(data_wav['FeSO4_azra'], conc_fe_halevy*data_abs['FeSO4_azra'], linewidth=2, linestyle='--', marker='o', color='red', label='100 nM FeSO$_4$')  
    
    ax[1].errorbar(data_wav['LK'], conc_fe_halevy*data_abs['K_ferrocyanide_LK'], yerr=conc_fe_halevy*data_abs['K_ferrocyanide_err_LK'], linewidth=2, linestyle='--', marker='o', color='purple', label='100 nM K$_4$Fe(CN)$_6$')
    ax[1].plot(data_wav['strizhakov_SNP'], 0.29*conc_fe_halevy*data_abs['strizhakov_SNP'], linewidth=2, linestyle='--', marker='o', color='blue', label='29 nM Na$_2$Fe(CN)$_5$NO')

    ax[1].legend(ncol=1, loc='upper right')
    ax[1].set_ylabel(r'Linear Decadic Absorption Coefficient (cm$^{-1}$)')    
    ax[1].set_yscale('log')
    ax[1].set_ylim([1E-6, 1.0E-2]) 
    ax[1].set_xscale('linear')
    ax[1].set_xlim([200., 300.])
    ax[1].set_xlabel('Wavelength (nm)')        
#    ax.axhline(1.0E1, color='black', linestyle=':')
#    ax.axhline(1.0E0, color='black', linestyle=':')
#    ax.axhline(1.0E-1, color='black', linestyle=':')
#    ax.axhline(1.0E-2, color='black', linestyle=':')
#    ax.axhline(1.0E-3, color='black', linestyle=':')
#    ax.axhline(1.0E-4, color='black', linestyle=':')
#    ax.axhline(1.0E-5, color='black', linestyle=':')

    
    plt.savefig('./Plots/molecules/ferrous-lake.pdf', orientation='portrait', format='pdf')
    
    ######
    #Plot UV in simulated reservoir
    ######
    ##Project pure water absorption to relevant wavelength scale, superpose
    ferrocyanide_lake_withwater=np.interp(uv_wav, data_wav['LK'], conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], left=0, right=0) + np.interp(uv_wav, h2o_quickenden_wav, h2o_quickenden_abs, left=0, right=0) #set absorptions to 0 where therem is no data; completely unphysical, but we shouldn't (and don't need to) trust those data anyway. Better to know clearly when things are full of shit. 
    fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True, sharey=True)
    ax.set_title('Ferrocyanide Lake UV')
    ax.plot(uv_wav, uv_toa_intensity, linewidth=2, linestyle='-', marker='o', color='black', label='Stellar Irradiation')

    ax.plot(uv_wav, uv_surface_intensity, linewidth=2, linestyle='-', marker='o', color='purple', label='Surface')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_withwater*1.0/0.5), linewidth=2, linestyle='-', marker='o', color='blue', label='1 cm')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_withwater*10.0/0.5), linewidth=2, linestyle='-', marker='o', color='cyan', label='10 cm')
    ax.plot(uv_wav, uv_surface_intensity*10.0**(-ferrocyanide_lake_withwater*100.0/0.5), linewidth=2, linestyle='-', marker='o', color='green', label='100 cm')
    
    ax.legend(ncol=1, loc='best')
    ax.set_xscale('linear')
    ax.set_xlabel('Wavelength (nm)')    
    ax.set_yscale('log')
    ax.set_ylabel(r'Intensity (photons cm$^{2}$ s$^{-1}$ nm$^{-1}$')
    ax.set_ylim([1E8, 1.0E14]) 
    ax.set_xlim([200., 300.])
    plt.savefig('./Plots/molecules/uv_ferrocyanide-lake.pdf', orientation='portrait', format='pdf')
    
    print('Fe2+, 100 uM')
    print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], data_wav['LK'], 100.0)))
    print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], data_wav['LK'], 100.0)))
    print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], data_wav['LK'], 100.0)))
    print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], data_wav['LK'], 100.0)))
    print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', conc_fe_hao_toner*data_abs['Fe(BF4)2_LK'], data_wav['LK'], 100.0)))
    print()
    print('FeCl2, 100 uM')
    print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', conc_fe_hao_toner*data_abs['FeCl2_azra'], data_wav['FeCl2_azra'], 100.0)))
    print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', conc_fe_hao_toner*data_abs['FeCl2_azra'], data_wav['FeCl2_azra'], 100.0)))
    print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', conc_fe_hao_toner*data_abs['FeCl2_azra'], data_wav['FeCl2_azra'], 100.0)))
    print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', conc_fe_hao_toner*data_abs['FeCl2_azra'], data_wav['FeCl2_azra'], 100.0)))
    print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', conc_fe_hao_toner*data_abs['FeCl2_azra'], data_wav['FeCl2_azra'], 100.0)))
    print()
    print('FeSO4, 100 uM')
    print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', conc_fe_hao_toner*data_abs['FeSO4_azra'], data_wav['FeSO4_azra'], 100.0)))
    print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', conc_fe_hao_toner*data_abs['FeSO4_azra'], data_wav['FeSO4_azra'], 100.0)))
    print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', conc_fe_hao_toner*data_abs['FeSO4_azra'], data_wav['FeSO4_azra'], 100.0)))
    print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', conc_fe_hao_toner*data_abs['FeSO4_azra'], data_wav['FeSO4_azra'], 100.0)))
    print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', conc_fe_hao_toner*data_abs['FeSO4_azra'], data_wav['FeSO4_azra'], 100.0)))
    print()
    print('Ferrocyanide, 100 uM')
    print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], data_wav['LK'], 100.0)))
    print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], data_wav['LK'], 100.0)))
    print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], data_wav['LK'], 100.0)))
    print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], data_wav['LK'], 100.0)))
    print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', conc_fe_hao_toner*data_abs['K_ferrocyanide_LK'], data_wav['LK'], 100.0)))
    print()
    print('Nitroprusside, 100 uM')
    print('Nucleobase photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('nucleobase-photolysis', conc_fe_hao_toner*data_abs['strizhakov_SNP'], data_wav['strizhakov_SNP'], 100.0)))
    print('Cytidine deamination timescale enhancement: {0:1.2e}'.format(get_timescale('cytidine-deamination', conc_fe_hao_toner*data_abs['strizhakov_SNP'], data_wav['strizhakov_SNP'], 100.0)))
    print('2AO photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AO-photolysis', conc_fe_hao_toner*data_abs['strizhakov_SNP'], data_wav['strizhakov_SNP'], 100.0)))
    print('2AI photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AI-photolysis', conc_fe_hao_toner*data_abs['strizhakov_SNP'], data_wav['strizhakov_SNP'], 100.0)))
    print('2AT photolysis timescale enhancement: {0:1.2e}'.format(get_timescale('2AT-photolysis', conc_fe_hao_toner*data_abs['strizhakov_SNP'], data_wav['strizhakov_SNP'], 100.0)))
    print()