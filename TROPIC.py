"""
Copyright (C) 2023 by Kevin Lee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

----------------------------------------------------------------------

TROPIC.py - TRanstitiOn ProbabIlity Caculator

Inputs data from a .csv file, uses the data to calculate the reduced transition probabilities. 
Output displayed in terminal, csv format, and LaTex table

Language: Python3 with Numpy and Scipy
University of Notre Dame

Created Jan 27, 2017 by Clark Casarella
Updated beginning Jun 6, 2017 by Anne Stratman
Updated beginning Jan 21, 2023 by Kevin Lee
"""

import math as m
import scipy.constants as sc
import numpy as np
import decimal as d
import os

# Disable default division by zero warnings for custom warnings
np.seterr(divide='ignore')

# Get path to directory of TROPIC.py
pathtodir = os.getcwd()+'/'

# Get input filename of .csv file containing data
filename = str(input('\nEnter csv filename (LEAVE OUT file extension): '))
csvpath = filename + '.csv'
print('Using the input parameters from:',pathtodir+csvpath)
# Create output filename using input filename
outpath = filename + '_output'
print('Output files placed at:',pathtodir)

# Creates output files
latex_output = open(outpath+'.txt','w')
csv_output = open(outpath+'.csv','w')

# Get desired precision 
decimal_places = int(input('\nEnter number of decimal places to report in results (Enter an integer): '))

# Get error threshold.
errorthreshold = input('\nEnter threshold for propagating errors by using Min/Max method (e.g. enter 0.1 for 10%, leave blank for standard error propagation): ')
if not errorthreshold:
    print('Using Standard Error Propagation')
else:
    print('Using Min/Max method when uncertainty is above ' + str(float(errorthreshold)*100) + '%')

# Get desired units
see_weisskopf_units=input('\nDo you want to see the Weisskopf unit conversion? [Y/N]: ')

# Specify data type of each field name in the output array
dtype_full=[('A','int'),('E_lev','f8'),('tau','f8'),('tau_up','f8'),('tau_down','f8'),('E_g','f8'),('E_g_error','f8'),('I_g','f8'),('I_g_error','f8'),('ICC','f8'),('ICC_error','f8'),('multipolarity','S6'),('delta_mixing','f8'),('delta_upper','f8'),('delta_lower','f8')]

ndtype=str

# Define field names for the output array
npnames=['A','E_lev','tau','tau_up','tau_down','E_g','E_g_error','I_g','I_g_error','ICC','ICC_error','multipolarity','delta_mixing','delta_upper','delta_lower']

# Separate list of parameters that could submitted as blank inputs
possibleblanks=['tau_up','tau_down','E_g_error','I_g','I_g_error','ICC','ICC_error','delta_mixing','delta_upper','delta_lower']

# Imports data from input file, with the field names and data types given above
csvfile = np.genfromtxt(csvpath,delimiter=",",skip_header=1,names=npnames,dtype=dtype_full)

# Units from scipy
m_p=sc.value('proton mass energy equivalent in MeV')
hc=sc.value('reduced Planck constant times c in MeV fm')
hbar=sc.value('reduced Planck constant in eV s')/10**6

def get_L(pi_l):
    """Extract the angular momentum from the multipolarity of a transition
    
    Arguments:
    pi_l (str): Multipolarity
    
    Returns:
    Angular Momentum (int)
    """
    return int(pi_l[-1])

def BwE(A,pi_l):
    """Weisskopf estimate for an EL transition, in units of e^2*b^L 
    
    Arguments:
    A (int): Mass number
    pi_l (str): Multipolarity
    
    Returns:
    Weisskopf estimate (float) if W.u. was indicated in prompt or 100^-l (float) if using e^2*b^L 
    """
    l=get_L(pi_l)
    if see_weisskopf_units.upper() == 'Y':
        return np.double(1.2**(2*l)/(4*m.pi)*(3/(l+3))**2*A**(2*l/3))
    elif see_weisskopf_units.upper() == 'N':
        return np.double(100**l)
    else:
        print('Not a valid option for setting the Weisskopf units, please try again.')
        exit()

def BwM(A,pi_l):
    """Weisskopf estimate for an ML transition, in units of mu_N^2*b^(L-1) 
    
    Arguments:
    A (int): Mass number
    pi_l (str): Multipolarity
    
    Returns:
    Weisskopf estimate (float) if W.u. was indicated in prompt or 1 (int) if using mu_N^2*b^(l-1)     
    """
    l=get_L(pi_l)
    if see_weisskopf_units.upper() == 'Y':
        return np.double(1.2**(2*(l-1))*10/m.pi*(3/(l+3))**2*A**(2*(l-1)/3))
    elif see_weisskopf_units.upper() == 'N':
        return np.double(100**(l-1))
    else:
        print('Not a valid option for setting the Weisskopf units, please try again.')
        exit()

def doublefactorial(n):
    """Double factorial (every other n factorialed)
    
    Arguments:
    n (int): The number to calculate the double factorial of
    
    Returns:
    Double factorial of n!! (int)
    """
    if n <=0:
        return 1
    else:
        return n*doublefactorial(n-2)
   
def mult_coefficient(pi_l):
    """Coefficient related to angular momentum parameter from B(pl) formula
    
    Arguments: 
    pi_l (str): Multipolarity
    
    Returns:
    Coefficient related to angular momentum parameter (float)  
    """
    l=get_L(pi_l)
    return np.double(l*(doublefactorial(2*l+1))**2/(l+1))

def mixing_fraction(delta,pi_l):
    """Calculates the relative EL strength to ML strength for any mixed-multipolarity transitions
    
    Arguments:
    delta (float): mixing ratio
    pi_l (str): Multipolarity
    
    Returns:
    Fraction of each component of the mixed transition (float) or 1 (int) if pure transition
    """
    if delta != 0:
        # Calculates the appropriate mixing fraction for the magnetic and electric components of a mixed transition
        if pi_l[-1]==label[1]:
            return np.double(1/(1+delta**2))
        else:
            return np.double(delta**2/(1+delta**2))   
    else:
        return 1

def BR(Ig,Itot):
    """Calculates the branching ratio (ratio of measured intensity to total intensity leaving the state
    
    Arguments:
    Ig (float): gamma ray intensity
    Itot (float): total intensity
    
    Returns:
    Branching ratio (float)
    """
    return np.double(Ig/Itot)

def B_coefficients(pi_l):
    """Consolidate coefficients for final B(pl) calculation. Make exception for E1 transition if W.u. is indicated (reported in mW.u.)
    
    Arguments:
    pi_l (str): Multipolarity   

    Returns:
    consolidated coefficient for final B(pl) calculation (float)    
    """
    l=get_L(pi_l)
    if pi_l=='E1' and see_weisskopf_units.upper() == 'Y':
        return np.double(hbar/(8*m.pi)*mult_coefficient(pi_l)*hc**(1+2*l)*1000)
    else:
        return np.double(hbar/(8*m.pi)*mult_coefficient(pi_l)*hc**(1+2*l))

def units(A,pi_l):
    """Final unit conversion (if necessary) of B(pl) calculation
    
    Arguments:
    A (int): Mass number
    pi_l (str): Multipolarity    
    
    Returns:
    Unit conversion for final B(pl) calculation (float)
    """
    l=get_L(pi_l)
    # Differentiate between El and Ml transitions
    if pi_l[0]=='E':
        return np.double(BwE(A,pi_l)*sc.alpha*hc)
    else:
        return np.double(BwM(A,pi_l)*sc.alpha*hc*(hc/(2*m_p))**2)
    
def latex_friendly_units(pi_l, delta):
    """Write units in LaTeX-friendly syntax
    
    Arguments:
    pi_l (str): Multipolarity 
    delta (float): mixing ratio
    
    Returns:
    Final units in latex syntax (str)    
    """
    # Assume Weisskopf units, switch to cgs if indicated by input
    unit = 'W.u.'
    if pi_l == 'E1':
        unit = 'mW.u.'
    if see_weisskopf_units.upper() == 'N':
        l=get_L(pi_l)
        if pi_l[0]=='E':
            if pi_l == 'E1':
                unit = 'e$^2$b'
            else:
                unit = 'e$^2$b$^'+str(l)+'$'
        elif pi_l == 'M1':
            unit = '$\\mu_N^2$'
        elif pi_l == 'M2':
            unit = '$\\mu_N^2$b'
        else:
            unit = '$\\mu_N^2$b$^'+str(l-1)+'$'
    if mixedflag == 'yes':
        if delta == 0:
            unit = unit + ' (assumed pure ' + pi_l + ')'
        else:
            unit = unit + ' (' + pi_l + ' component)'
    return unit         

def B(paramdict):
    """Calculation of transition probability B(pl) from all inputs necessary
    
    Arguments:
    paramdict (dict): dictionary containing all input parameters needed for calculation
    
    Returns:
    B(pl) value (float)
    """
    l=get_L(paramdict['pi_l'])
    return np.double(mixing_fraction(paramdict['delta'],paramdict['pi_l'])*BR(paramdict['I_g'],paramdict['I_tot'])/(paramdict['tau']*10**-15*(paramdict['E_g']/1000)**(2*l+1))*B_coefficients(paramdict['pi_l'])/units(paramdict['A'],paramdict['pi_l']))

# Functions for standard error propagation: take derivatives w.r.t. each variable that can have an error
def dbdtau(Bvalue, tau, tauerrors):
    """Standard error propagation of lifetime uncertainty
    
    Arguments:
    Bvalue (float): B(pl) value
    tau (float): lifetime
    tauerrors (list): list containing upper and lower uncertainties of lifetime 
    
    Returns:
    List containing calculated derivative values of B(pi) w.r.t. lifetime for the upper and lower uncertainties
    """
    tau_error = [0,0]
    for i in range(len(tauerrors)):
        tau_error[i] = np.double((-Bvalue*tauerrors[1-i]/tau)**2)
    return tau_error

def dBdE(Bvalue, Eg, Eg_error, pi_l):
    """Standard error propagation of gamma ray energy uncertainty
    
    Arguments:
    Bvalue (float): B(pl) value
    Eg (float): gamma ray energy
    Eg_error (float): gamma ray energy uncertainty
    pi_l (str): multipolarity
    
    Returns:
    Calculated derivative value of B(pi) w.r.t. gamma ray energy (float)
    """
    l=get_L(pi_l)
    return np.double((-Bvalue*(2*l+1)*Eg_error/Eg)**2)

def dBdI(Bvalue, Ig, Ig_error):
    """Standard error propagation of intensity uncertainty
    
    Arguments:
    Bvalue (float): B(pl) value
    Ig (float): gamma ray intensity
    Ig_error (float): gamma ray intensity uncertainty
    
    Returns:
    Calculated derivative value of B(pi) w.r.t. gamma ray intensity (float)
    """
    return np.double((Bvalue*Ig_error/Ig)**2)
    
def dBdI_tot(Bvalue, Itot, Itot_error):
    """Standard error propagation of total intensity uncertainty
    
    Arguments:
    Bvalue (float): B(pl) value
    Itot (float): gamma ray intensity
    Itot_error (float): gamma ray intensity uncertainty
    
    Returns:
    Calculated derivative value of B(pi) w.r.t. total intensity (float)
    """
    return np.double((-Bvalue*Itot_error[0]/Itot)**2)    

def dbd_delta_str(Bvalue, delta, deltaerrors, pi_l):
    """Standard error propagation of delta uncertainty for the stronger component in mixed transitions
    
    Arguments:
    Bvalue (float): B(pl) value
    delta (float): mixing ratio
    deltaerrors (list): list containing upper and lower uncertainties of delta
    pi_l (str): multipolarity
    
    Returns:
    List containing calculated derivative values of B(pi) w.r.t. delta for the upper and lower uncertainties for the stronger component in mixed transitions (float)
    """
    delta_str_error = [0,0]
    if delta == 0:
        return delta_str_error
    if mixedflag == 'yes' and pi_l[-1]==label[4]:    
        derivative = (2/delta) - (2*delta)/(1+delta**2)
        for i in range(len(deltaerrors)):
            delta_str_error[i] = np.double((Bvalue*deltaerrors[i]*derivative)**2) 
    return delta_str_error
    
def dbd_delta_wk(Bvalue, delta, deltaerrors, pi_l):
    """Standard error propagation of delta uncertainty for the weaker component in mixed transitions
    
    Arguments:
    Bvalue (float): B(pl) value
    delta (float): mixing ratio
    deltaerrors (list): list containing upper and lower uncertaintiess of delta
    pi_l (str): multipolarity
    
    Returns:
    List containing calculated derivative values of B(pi) w.r.t. delta for the upper and lower uncertainties for the weaker component in mixed transitions (float)
    """
    delta_wk_error = [0,0]
    if delta == 0:
        return delta_wk_error
    if mixedflag == 'yes' and pi_l[-1]==label[1]:
        for i in range(len(deltaerrors)):
            delta_wk_error[i] = np.double((-Bvalue*deltaerrors[1-i]*2*delta/(1+delta**2))**2)
    return delta_wk_error

def dBdalpha(Bvalue, icc, icc_error):
    """Standard error propagation of ICC uncertainty
    
    Arguments:
    Bvalue (float): B(pl) value
    icc (float): conversion coefficient
    icc_error (float): conversion coefficient uncertainty
    
    Returns:
    Calculated derivative value of B(pi) w.r.t. conversion coefficient (float)
    """
    return np.double((-Bvalue*icc_error/(1+icc))**2)

def uncertainty(paramdict):
    """Aggregates uncertainties from every parameter using standard error propagation
    
    Arguments:
    paramdict (dict): dictionary containing all input parameters needed for calculation
    
    Returns:
    List containing final upper and lower uncertainties obtained through standard error propagation (float)
    """
    bvalue = B(paramdict)
    bounds = [0,0]
    for i in range(len(bounds)):
        bounds[i] = np.double((dbdtau(bvalue,paramdict['tau'],paramdict['tau_errors'])[i]+dBdE(bvalue,paramdict['E_g'],paramdict['E_g_error'],paramdict['pi_l'])+dBdI(bvalue,paramdict['I_g'],paramdict['I_g_error'])+dBdI_tot(bvalue,paramdict['I_tot'],paramdict['I_tot_error'])+dbd_delta_wk(bvalue,paramdict['delta'],paramdict['delta_errors'],paramdict['pi_l'])[i]+dbd_delta_str(bvalue,paramdict['delta'],paramdict['delta_errors'],paramdict['pi_l'])[i])**0.5)
    return bounds    

# Functions for formatting

def formatnotation(val, precision):
    """Write value in scientific notaion using the number of decimals as precision if it's too small (<0.1) or too big (>10000) 
    
    Arguments:
    val (float): value to consider writing in scientific notation
    precision (int): number of decimal numbers specified in prompt

    Returns:
    Value in scientic notation or as is (str)
    """
    txt = "{:."+str(precision)+"e}" 
    if val < 0.1 or val > 10000:
        return txt.format(val)
    else:
        return str(round(val,precision))   

def shorthand_error(val, err):
    """Write nominal value and its uncertainty in shorthand notation
    
    Arguments:
    val (float): nominal value
    err (float): uncertainty
    
    Returns:
    nominal value and uncertainity in shorthand notation (str)
    """
    # Isolate few cases such as 0 for nominal or error and both values being integers
    if err == 0:
        if val == int(val):
            val = int(val)
        return str(val)  
    if val == 0:
        return str(0)
    if val == int(val) and  err == int(err):
        return str(int(val))+'('+str(int(err))+')'
    # Otherwise create shorthand notation
    valdecimalnum = abs(d.Decimal(str(val)).as_tuple().exponent)
    errdecimalnum = abs(d.Decimal(str(err)).as_tuple().exponent)
    error = str(int(err*(10**errdecimalnum)))
    value = str(val)
    if valdecimalnum < errdecimalnum:
        value = value+'0'*(errdecimalnum-valdecimalnum)
    elif valdecimalnum > errdecimalnum: 
        error = error+'0'*(valdecimalnum-errdecimalnum)
    return value+'('+error+')'

def fillblank(index):
    """Fill in blanks inputs in csv file with zeros
    
    Arguments:
    index (int): the row number in the csvfile
    
    Returns:
    None
    """
    # Check for blank inputs in intensity, ICC, and delta and fill with 0
    for key in possibleblanks:
        if m.isnan(csvfile[index][key]):
            csvfile[index][key] = 0
    return
            
# Read and Calculate the probability

def write_line(paramdict):
    """Write the results in latex syntax, in the csv file, and also print out in terminal
    
    Arguments:
    paramdict (dict): dictionary containing all input parameters needed for calculation    
    
    Returns:
    None
    """
    # Header for each level in terminal output
    if paramdict['A_out'] != '':
        print('-'*69)
        print('A:',paramdict['A_out'],'| E_lev (keV):',paramdict['E_lev_out'],'| tau (fs): ',str(paramdict['tau_out']),'\n')
        print('Transitions for this level: \n')
        print('E_gamma (keV) | E_f (kev) | Intensity | ICC | Multipolarity | B(pi*l) \n')
    # If there's a potential division by 0, print warning
    if paramdict['E_g'] == 0 or paramdict['tau'] == 0 or paramdict['I_tot'] == 0 or (paramdict['I_g'] == 0 and paramdict['minmaxflag'] == 'false'):
        print('WARNING: DIVISION BY ZERO DETECTED IN THIS CALCULATION. PLEASE CHECK INPUT PARAMETERS.')  
    # Calculate B(pi*l) value
    bvalue = B(paramdict)
    # Write A, E(level), tau, E(gamma), Intensity, alpha, multipolarity in latex syntax 
    lineMassNumber=str(paramdict['A_out']).ljust(4,' ')+' & '
    lineTau=(str(paramdict['tau'])+'$^{+'+str(paramdict['tau_errors'][0])+'}_{-'+str(paramdict['tau_errors'][1])+'}$ ').ljust(20,' ') + ' & '
    if paramdict['tau_errors'][0] == paramdict['tau_errors'][1]:
        lineTau = shorthand_error(paramdict['tau'],paramdict['tau_errors'][0]).ljust(15,' ') + ' & '        
    if paramdict['tau_out'] == '':
        lineTau = ' & '  
    lineEnergy=str(paramdict['E_lev_out']).ljust(15,' ')+' & '+ lineTau +shorthand_error(paramdict['E_g'],paramdict['E_g_error']).ljust(15,' ')+' & '+str(round(paramdict['E_lev']-paramdict['E_g'],2)).ljust(15,' ')+' & '   
    lineIntandICC=shorthand_error(paramdict['I_g'],paramdict['I_g_error']).ljust(15,' ')+' & '+ shorthand_error(paramdict['ICC'],paramdict['ICC_error']).ljust(15,' ') + ' & ' 
    lineMultipolarity=str(label).ljust(10,' ')+' & '
    uncertainties = [0,0]
    # If the error is above a certain threshold for any of the parameters, get uncertainties by using upper/lower bounds of each parameter to obtain upper/lower bound of the transition
    errorpercent = [paramdict['E_g_error']/paramdict['E_g'],paramdict['I_g_error']/paramdict['I_g'],paramdict['tau_errors'][0]/paramdict['tau'],paramdict['tau_errors'][1]/paramdict['tau']]
    if paramdict['delta'] != 0:
        errorpercent.extend([paramdict['delta_errors'][0]/paramdict['delta'],paramdict['delta_errors'][1]/paramdict['delta']])
    # Check if the errors of the other parameters are above the error threshold
    if paramdict['minmaxflag'] == 'false':
        for unc in errorpercent:
            # If blank input was submitted, use standard propagation regardless
            if not errorthreshold:
                break
            # If errorthreshold is met, use min/max method
            if unc >= float(errorthreshold):
                paramdict['minmaxflag'] = 'true'
                break    
    # Min/max method for error propagation 
    if paramdict['minmaxflag'] == 'true':
        E_bounds = [paramdict['E_g']+paramdict['E_g_error'],paramdict['E_g']-paramdict['E_g_error']]
        I_bounds = [paramdict['I_g']+paramdict['I_g_error'],paramdict['I_g']-paramdict['I_g_error']]
        I_tot_bounds = [paramdict['I_tot']+paramdict['I_tot_error'][0],paramdict['I_tot']-paramdict['I_tot_error'][1]]
        delta_bounds = [abs(paramdict['delta'])+paramdict['delta_errors'][0],abs(paramdict['delta'])-paramdict['delta_errors'][1]]
        tau_bounds = [paramdict['tau']+paramdict['tau_errors'][0],paramdict['tau']-paramdict['tau_errors'][1]]
        for i in range(len(uncertainties)):
            tempparam = paramdict.copy()
            tempparam['E_g'] = E_bounds[1-i]
            tempparam['I_g'] = I_bounds[i]
            tempparam['I_tot'] = I_tot_bounds[1-i]
            if tempparam['pi_l'][-1]==label[1]:
                tempparam['delta'] = delta_bounds[1-i]
            else:
                tempparam['delta'] = delta_bounds[i]
            tempparam['tau'] = tau_bounds[1-i]
            uncertainties[i] = abs(B(tempparam)-bvalue)
    # Standard error propagation
    else:
        uncertainties[0] = uncertainty(paramdict)[0]
        uncertainties[1] = uncertainty(paramdict)[1]
    # Format notation before displaying/writing results   
    bvalue = formatnotation(bvalue,decimal_places)    
    for j in range(len(uncertainties)):
        uncertainties[j] = formatnotation(uncertainties[j],decimal_places)
    # Write results in latex syntax, csv file, and also print to terminal    
    # Latex
    lineMult=(bvalue+'$^{+'+uncertainties[0]+'}_{-'+uncertainties[1]+'}$ '+latex_friendly_units(paramdict['pi_l'],paramdict['delta'])).ljust(30,' ')+' \\\\ \n'
    latex_output.write(lineMassNumber+lineEnergy+lineIntandICC+lineMultipolarity+lineMult)
    # CSV file
    csv_output.write(','.join([str(paramdict['A_out']),str(paramdict['E_lev_out']),str(paramdict['tau_out']),str(paramdict['E_g']),str(round(paramdict['E_lev']-paramdict['E_g'],2)),str(paramdict['I_g']),str(paramdict['ICC']),label,bvalue,uncertainties[0],uncertainties[1],latex_friendly_units(paramdict['pi_l'],paramdict['delta']).translate(str.maketrans({'$': '', '\\': ''})),'\n']))
    # Terminal 
    print(shorthand_error(paramdict['E_g'],paramdict['E_g_error']),'|',round(paramdict['E_lev']-paramdict['E_g'],2),'|',shorthand_error(paramdict['I_g'],paramdict['I_g_error']),'|',shorthand_error(paramdict['ICC'],paramdict['ICC_error']),'|', label, '|', bvalue,'+/-','('+uncertainties[0]+', '+uncertainties[1]+')',latex_friendly_units(paramdict['pi_l'],paramdict['delta']).translate(str.maketrans({'$': '', '\\': ''})),'\n')
    return
  
# LaTeX Table header
latex_output.write('\\begin{table}[ht]\n')
latex_output.write('\\caption{REMEMBER TO CHANGE TABLE CAPTION AND REFERENCE TAG HERE! \\label{tab:results}}\n')
latex_output.write('\\begin{tabular}{l l l l l l l l l}\n')
latex_output.write('A & E$_{lev}$ (keV) & $\\tau$ (fs) & E$_{\\gamma}$ (keV) & E$_{f}$ (keV) & Intensity & $\\alpha$ & $\\pi\\ell$ & B($\\pi\\ell$) \\\\ \n')
latex_output.write('\\hline \n')

# CSV output header
csv_output.write('A,E_lev (keV),tau (fs),E_gamma (keV),E_f (keV),Intensity,ICC,Multipolarity,B(pi*l),B(pi*l)_err_up,B(pi*l)_err_down,Unit \n')

# Calculate intensity sums for all levels
catalog = []
Lev_info = {}
i = 0
j = 1
k = 0
while i < len(csvfile):
    catalog.append('index'+str(i))
    minmaxflag = 'false'
    Lev_I_gs = []
    Lev_ICCs = []
    # Get all intensities and ICC's from a level
    # First transition for a level
    # Fill in blanks for intensity, ICC, and delta
    fillblank(i)
    Lev_I_gs.append([csvfile[i]['I_g'],csvfile[i]['I_g_error'],csvfile[i]['E_g'],csvfile[i]['E_lev']])
    Lev_ICCs.append([csvfile[i]['ICC'],csvfile[i]['ICC_error'],csvfile[i]['E_g'],csvfile[i]['E_lev']])
    # Subsequent transitions from the same level
    if i+j < len(csvfile):
        while m.isnan(csvfile[i+j]['E_lev']) == True:
            if i+j == len(csvfile):
                break
            # Fill in blanks for intensity, ICC, and delta
            fillblank(i+j)
            # Fill in missing parameters
            csvfile[i+j]['A'] = csvfile[i]['A']
            csvfile[i+j]['E_lev'] = csvfile[i]['E_lev']
            csvfile[i+j]['tau'] = csvfile[i]['tau']
            csvfile[i+j]['tau_up'] = csvfile[i]['tau_up']
            csvfile[i+j]['tau_down'] = csvfile[i]['tau_down']
            catalog.append('index'+str(i))         
            Lev_I_gs.append([csvfile[i+j]['I_g'],csvfile[i+j]['I_g_error'],csvfile[i+j]['E_g'],csvfile[i+j]['E_lev']])
            Lev_ICCs.append([csvfile[i+j]['ICC'],csvfile[i+j]['ICC_error'],csvfile[i+j]['E_g'],csvfile[i+j]['E_lev']])
            j = j + 1
            if i+j == len(csvfile):
                break
    # Check if any of the uncertainties are above the errorthreshold. If blankinput was submitted for errorthreshold, default to standard error propagation
    # Check intensities
    for unc in Lev_I_gs:
        if not errorthreshold:
            break
        if unc[0] == 0:
            continue
        if unc[1]/unc[0] >= float(errorthreshold):
            minmaxflag = 'true'
            break    
    # Check ICC's if none of the intensities passed the errorthreshold
    if minmaxflag == 'false':
        for unc in Lev_ICCs:
            if not errorthreshold:
                break
            if unc[0] == 0:
                continue
            if unc[1]/unc[0] >= float(errorthreshold):
                minmaxflag = 'true'
                break    
    I_tot = 0
    I_tot_err = 0
    I_tot_max = 0
    I_tot_min = 0
    while k < len(Lev_I_gs):
        I_ce = Lev_I_gs[k][0]*Lev_ICCs[k][0]
        I_tot = I_tot + Lev_I_gs[k][0]+I_ce
        # Obtain min/max total intensity for min/max error propagation
        if minmaxflag == 'true':
            I_g_max = Lev_I_gs[k][0] + Lev_I_gs[k][1] 
            I_g_min = Lev_I_gs[k][0] - Lev_I_gs[k][1] 
            I_ce_max = I_g_max * (Lev_ICCs[k][0] + Lev_ICCs[k][1])
            I_ce_min = I_g_min * (Lev_ICCs[k][0] - Lev_ICCs[k][1])
            I_tot_max = I_tot_max + I_g_max + I_ce_max
            I_tot_min = I_tot_min + I_g_min + I_ce_min
        # Propagate errors by standard error propagation
        else:
            ICC_err_percent = 0
            if Lev_ICCs[k][0] != 0:
                ICC_err_percent = Lev_ICCs[k][1]/Lev_ICCs[k][0]
            I_g_err_precent = 0
            if Lev_I_gs[k][0] != 0:
                I_g_err_precent = Lev_I_gs[k][1]/Lev_I_gs[k][0]
            I_ce_err = I_ce*m.sqrt((I_g_err_precent)**2+(ICC_err_percent)**2)
            I_tot_err = I_tot_err + Lev_I_gs[k][1]**2+I_ce_err**2    
        k = k + 1
    # Record information about each level
    if minmaxflag == 'true':
        Lev_info['index'+str(i)] = [I_tot,[I_tot_max - I_tot,I_tot - I_tot_min],minmaxflag]
    else:
        Lev_info['index'+str(i)] = [I_tot,[I_tot_err,I_tot_err],minmaxflag]
    i = i+j
    j = 1
    k = 0
    
# Do calculations for each row of interest in CSVfile
print('\nCalculation Results:')

for row in list(range(len(csvfile))):
    # Skip over E0 transitions
    if m.isnan(csvfile[row]['E_g'])==True:
        continue
    # Extract parameters from csv in dictionary
    parameters = {'A':csvfile[row]['A'],'E_lev':csvfile[row]['E_lev'],'tau':csvfile[row]['tau'],'tau_errors':[csvfile[row]['tau_up'],csvfile[row]['tau_down']],'E_g':csvfile[row]['E_g'], 'E_g_error':csvfile[row]['E_g_error'],'I_g':csvfile[row]['I_g'],'I_g_error':csvfile[row]['I_g_error'],'ICC':csvfile[row]['ICC'],'ICC_error':csvfile[row]['ICC_error'],'pi_l':csvfile[row]['multipolarity'].decode("utf-8"),'delta':csvfile[row]['delta_mixing'],'delta_errors':[csvfile[row]['delta_upper'],csvfile[row]['delta_lower']]}
    # Add intensity and lifetimes from earlier
    parameters['I_tot'] = Lev_info[catalog[row]][0]
    parameters['I_tot_error'] = Lev_info[catalog[row]][1]
    parameters['minmaxflag'] = Lev_info[catalog[row]][2]
    # Skip adding A and E_lev for subsequent transitions from a level
    parameters['A_out'] = csvfile[row]['A']
    parameters['E_lev_out'] = csvfile[row]['E_lev']
    parameters['tau_out'] = csvfile[row]['tau']
    if row > 0 and catalog[row] == catalog[row-1]:
        parameters['A_out'] = ''
        parameters['E_lev_out'] = ''   
        parameters['tau_out'] = ''
    # Mixedflag used to identify mixed transitions and adjust calculations accordingly  
    mixedflag='no'
    # Write out result in three formats (latex, csv, and terminal)
    label = parameters['pi_l'] 
    if '/' in label:
        mixedflag='yes'
        mixedtrans = label.split('/')
        for component in range(len(mixedtrans)):
            parameters['pi_l'] = mixedtrans[component]            
            parameters['minmaxflag'] = Lev_info[catalog[row]][2]
            if component > 0:
                parameters['A_out'] = ''
                parameters['E_lev_out'] = ''   
                parameters['tau_out'] = ''  
            write_line(parameters)
    else:
        write_line(parameters)
          
latex_output.write('\\end{tabular}\n')
latex_output.write('\\end{table}')
latex_output.close()
csv_output.close()