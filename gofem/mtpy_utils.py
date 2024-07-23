'''
    Various auxiliary routines to work with the MTpy for GoFEM
    
    Alexander Grayver, 2020 - 2024
'''
import math
import sys
import numpy as np
import pandas as pd

import mtpy.core.transfer_function.pt as pt
import mtpy.core.transfer_function.z as mtz

from mtpy import MT, MTCollection, MTData

from gofem.data_utils import *

def write_mt_collection_to_gofem(outfile, mt_collection = None, mt_objects = None,\
                                  error_floor = 0.05, data_type = 'Z',\
                                  period_range = [-math.inf, math.inf],\
                                  error_floor_type = 'rowwise'):
    '''
        Write down MT impdeance tensors from the mt collection
        to the GoFEM data file with the given error floor.
        
        Error floor is applied to the impdeance (or derived quantities) 
        row-wise or by taking the geometric mean of Zxy and Zyx.
        
        @p data_type can take several values:
        'Z' -- write full impedance
        'Z_offdiag' -- write only off diagonal elements
        'RP' -- write apparent resistivity and phase
        'RP_offdiag' -- write rho and phase for off-diag elements
        'PT' -- phase tensor
        'Tipper' -- write vertical magnetic TFs
    '''

    mt_data = []
    ptol = 0.05
    
    str_codes = []
    if data_type == 'Z_offdiag':
        str_codes = ['RealZxy', 'ImagZxy', 'RealZyx', 'ImagZyx']
    elif data_type == 'Z':
        str_codes = ['RealZxy', 'ImagZxy', 'RealZyx', 'ImagZyx',
                     'RealZxx', 'ImagZxx', 'RealZyy', 'ImagZyy']
    elif data_type == 'PT':
        str_codes = ['PTxx', 'PTxy', 'PTyx', 'PTyy']
    elif data_type == 'RP':
        str_codes = ['RhoZxy', 'RhoZyx', 'PhsZxy', 'PhsZyx',
                     'RhoZxx', 'RhoZyy', 'PhsZxx', 'PhsZyy']
    elif data_type == 'RP_offdiag':
        str_codes = ['RhoZxy', 'RhoZyx', 'PhsZxy', 'PhsZyx']
    elif data_type == 'Tipper':
        str_codes = ['RealTzx', 'ImagTzx', 'RealTzy', 'ImagTzy']
    else:
        raise RuntimeError('Unsupported data type')
        
    rho_err_func = lambda z, ze, freq: 2. * abs(z) / (2. * math.pi * freq * mu) * ze
    phi_err_func = lambda z, ze: 180. / math.pi * ze / abs(z)
    
    # Convert from [mV/km]/[nT] to Ohm
    factor = (4 * math.pi) / 10000.0
    
    if mt_objects is not None:
        all_frequencies = []
        
        for mt_obj in mt_objects:
            for freq in mt_obj.Z.frequency:
                freq_max = freq * (1 + ptol)
                freq_min = freq * (1 - ptol)
                f_index_list = np.where((all_frequencies < freq_max) & (all_frequencies > freq_min))[0]    
                
                if f_index_list.size == 0:
                    all_frequencies.append(freq)
    elif mt_collection is not None:
        all_frequencies = np.reciprocal(mt_collection.to_mt_data().get_periods())

        mt_objects = []
        all_stations = list(mt_collection.dataframe['station'])

        for station in all_stations:
            mt_objects.append(mt_collection.get_tf(station))
    else:
        raise RuntimeError('Provide EDI collection or list of MT objects')
    
    for freq in all_frequencies:
        
        period = 1./freq
        if period < period_range[0] or period > period_range[1]:
            continue
        
        for mt_obj in mt_objects:
            freq_max = freq * (1 + ptol)
            freq_min = freq * (1 - ptol)
            f_index_list = np.where((mt_obj.Z.frequency < freq_max) & (mt_obj.Z.frequency > freq_min))[0]
            
            #print(f_index_list, freq_max, freq_min, len(f_index_list), type(f_index_list))
            
            if f_index_list.size > 1:
                print("more than one freq found %s", f_index_list)

            if f_index_list.size < 1:
                continue
                
            p_index = f_index_list[0]
            
            zobj = mt_obj.Z
            tobj = mt_obj.Tipper
            ptobj = mt_obj.pt
            
            if(error_floor_type == 'rowwise'):
                dZxy = max([zobj.z_error[p_index, 0, 1], error_floor * abs(zobj.z[p_index, 0, 1])]) * factor
                dZyx = max([zobj.z_error[p_index, 1, 0], error_floor * abs(zobj.z[p_index, 1, 0])]) * factor
            elif(error_floor_type == 'offdiag'):
                Z_mean = np.sqrt(np.abs(zobj.z[p_index, 0, 1]*zobj.z[p_index, 1, 0]))
                dZxy = max([zobj.z_error[p_index, 0, 1], error_floor * Z_mean]) * factor
                dZyx = max([zobj.z_error[p_index, 1, 0], error_floor * Z_mean]) * factor
                
            dZ = np.zeros(shape=(2,2))
            dZ[0,:] = dZxy
            dZ[1,:] = dZyx
                        
            if data_type == 'Z_offdiag' or data_type == 'Z':
                mt_data.append([str_codes[0], freq, mt_obj.station, zobj.z[p_index, 0, 1].real * factor, dZxy])
                mt_data.append([str_codes[1], freq, mt_obj.station, zobj.z[p_index, 0, 1].imag * factor, dZxy])
                mt_data.append([str_codes[2], freq, mt_obj.station, zobj.z[p_index, 1, 0].real * factor, dZyx])
                mt_data.append([str_codes[3], freq, mt_obj.station, zobj.z[p_index, 1, 0].imag * factor, dZyx])
                               
                if data_type == 'Z':
                    mt_data.append([str_codes[4], freq, mt_obj.station, zobj.z[p_index, 0, 0].real * factor, dZxy])
                    mt_data.append([str_codes[5], freq, mt_obj.station, zobj.z[p_index, 0, 0].imag * factor, dZxy])
                    mt_data.append([str_codes[6], freq, mt_obj.station, zobj.z[p_index, 1, 1].real * factor, dZyx])
                    mt_data.append([str_codes[7], freq, mt_obj.station, zobj.z[p_index, 1, 1].imag * factor, dZyx])
                                   
            elif data_type == 'RP_offdiag' or data_type == 'RP':
                mt_data.append([str_codes[0], freq, mt_obj.station, zobj.resistivity[p_index, 0, 1], rho_err_func(zobj.z[p_index, 0, 1], dZxy, freq)])
                mt_data.append([str_codes[1], freq, mt_obj.station, zobj.resistivity[p_index, 1, 0], rho_err_func(zobj.z[p_index, 1, 0], dZyx, freq)])
                mt_data.append([str_codes[2], freq, mt_obj.station, zobj.phase[p_index, 0, 1], phi_err_func(zobj.z[p_index, 0, 1], dZxy)])
                mt_data.append([str_codes[3], freq, mt_obj.station, zobj.phase[p_index, 1, 0], phi_err_func(zobj.z[p_index, 1, 0], dZyx)])
                
                if data_type == 'RP':
                    mt_data.append([str_codes[4], freq, mt_obj.station, zobj.resistivity[p_index, 0, 0], rho_err_func(zobj.z[p_index, 0, 0], dZxy, freq)])
                    mt_data.append([str_codes[5], freq, mt_obj.station, zobj.resistivity[p_index, 1, 1], rho_err_func(zobj.z[p_index, 1, 1], dZyx, freq)])
                    mt_data.append([str_codes[6], freq, mt_obj.station, zobj.phase[p_index, 0, 0], phi_err_func(zobj.z[p_index, 0, 0], dZxy)])
                    mt_data.append([str_codes[7], freq, mt_obj.station, zobj.phase[p_index, 1, 1], phi_err_func(zobj.z[p_index, 1, 1], dZyx)])
                    
            elif data_type == 'Tipper':
                mt_data.append([str_codes[0], freq, mt_obj.station, tobj.tipper[p_index, 0, 0].real, max(tobj.tipper_err[p_index, 0, 0], error_floor)])
                mt_data.append([str_codes[1], freq, mt_obj.station, tobj.tipper[p_index, 0, 0].imag, max(tobj.tipper_err[p_index, 0, 0], error_floor)])
                mt_data.append([str_codes[2], freq, mt_obj.station, tobj.tipper[p_index, 0, 1].real, max(tobj.tipper_err[p_index, 0, 1], error_floor)])
                mt_data.append([str_codes[3], freq, mt_obj.station, tobj.tipper[p_index, 0, 1].imag, max(tobj.tipper_err[p_index, 0, 1], error_floor)])

            elif data_type == 'PT':
                PT, dPT = calculate_phase_tensor(zobj.z[p_index] * factor, dZ)
                
                mt_data.append([str_codes[0], freq, mt_obj.station, ptobj.pt[p_index, 0, 0].real, dPT[0, 0]])
                mt_data.append([str_codes[1], freq, mt_obj.station, ptobj.pt[p_index, 0, 1].real, dPT[0, 1]])
                mt_data.append([str_codes[2], freq, mt_obj.station, ptobj.pt[p_index, 1, 0].real, dPT[1, 0]])
                mt_data.append([str_codes[3], freq, mt_obj.station, ptobj.pt[p_index, 1, 1].real, dPT[1, 1]])
    
    with open(outfile, 'w') as f:
        f.write('# DataType Frequency Source Receiver Value Error\n')
        idx = 0
        for idx in range(len(mt_data)):
            data_row = mt_data[idx]
            
            if math.isfinite(data_row[3]) and abs(data_row[3]) > 1e-25:
                f.write("%s %0.6e Plane_wave %s %0.6e %0.6e\n" % (data_row[0], data_row[1], data_row[2], data_row[3], data_row[4]))


def read_gofem_modelling_output(fileformat, frequency_list, station_list, station_coords):
    '''
        Read in GoFEM modelling output and return a list of MTpy objects
    '''
        
    dfs = read_modeling_responses(fileformat, frequency_list)
    
    assert len(dfs) == len(frequency_list)
    
    z_dummy = np.zeros((len(frequency_list), 2, 2), dtype='complex')
    t_dummy = np.zeros((len(frequency_list), 1, 2), dtype='complex')
    
    # Conversion factor from Ohm to [mV/km]/[nT]
    factor = 10000.0 / (4 * math.pi)
    
    Z_data_dict = {}
    T_data_dict = {}
    for station, xyz in zip(station_list, station_coords):
        Z_data_dict[station] = z_dummy.copy()
        T_data_dict[station] = t_dummy.copy()

    for fidx in range(len(frequency_list)):
        for index, row in dfs[fidx].iterrows():
            if row['Receiver'] not in Z_data_dict.keys():
                raise Exception('Station', row['Receiver'], ' is not in the station list.')
                
            station = row['Receiver']
            
            Z_data_dict[station][fidx, 0, 0] = row['Zxx'] * factor
            Z_data_dict[station][fidx, 0, 1] = row['Zxy'] * factor
            Z_data_dict[station][fidx, 1, 0] = row['Zyx'] * factor
            Z_data_dict[station][fidx, 1, 1] = row['Zyy'] * factor
            
            T_data_dict[station][fidx, 0, 0] = row['Tzx']
            T_data_dict[station][fidx, 0, 1] = row['Tzy']
            
    mt_obj_list = []
    for station in Z_data_dict.keys():
        mt_obj = MT(station = station, survey = 'Modelling', frequency = np.array(frequency_list),
                    impedance=Z_data_dict[station], impedance_error=z_dummy.copy().real,
                    tipper=T_data_dict[station], tipper_error=t_dummy.copy().real)

        mt_obj_list.append(mt_obj)
    
    return mt_obj_list

def read_gofem_inversion_output(data_file):
    '''
        Read in GoFEM inversion data file and return a list of MTpy objects
    '''
    
    colnames = ['type', 'frequency', 'source', 'station', 'value', 'std_error']
    df = pd.read_csv(data_file, sep="[ \t]+", header=None, names=colnames, comment='#')

    all_frequencies = df['frequency'].unique()
    stations = df['station'].unique()

    ptol = 0.03

    # Conversion factor from Ohm to [mV/km]/[nT]
    factor = 10000.0 / (4 * math.pi)

    mt_obj_list = []
    for station in stations:
        dfs = df[df['station'] == station]
    
        z = np.zeros((len(all_frequencies), 2, 2), dtype='complex')
        z_err = np.zeros((len(all_frequencies), 2, 2))

        tipper = np.zeros((len(all_frequencies), 1, 2), dtype='complex')
        tipper_err = np.zeros((len(all_frequencies), 1, 2))
            
        for n, frequency in enumerate(all_frequencies):
            freq_max = frequency * (1 + ptol)
            freq_min = frequency * (1 - ptol)
        
            dfs_freq = dfs[(dfs['frequency'] < freq_max) & (dfs['frequency'] > freq_min)]
        
            if(dfs_freq.size == 0):
                continue;
            
            dfs_freq.set_index('type',inplace=True)
        
            if 'RealZxx' in dfs_freq.index and 'ImagZxx' in dfs_freq.index:
                Zxx = complex(dfs_freq.loc['RealZxx'].value, 
                              dfs_freq.loc['ImagZxx'].value)
                z[n, 0, 0] = Zxx
                z_err[n, 0, 0] = dfs_freq.loc['RealZxx'].std_error
            
            if 'RealZxy' in dfs_freq.index and 'ImagZxy' in dfs_freq.index:
                Zxy = complex(dfs_freq.loc['RealZxy'].value, 
                              dfs_freq.loc['ImagZxy'].value)
                z[n, 0, 1] = Zxy
                z_err[n, 0, 1] = dfs_freq.loc['RealZxy'].std_error
            
            if 'RealZyx' in dfs_freq.index and 'ImagZyx' in dfs_freq.index:
                Zyx = complex(dfs_freq.loc['RealZyx'].value, 
                              dfs_freq.loc['ImagZyx'].value)
                z[n, 1, 0] = Zyx
                z_err[n, 1, 0] = dfs_freq.loc['RealZyx'].std_error
            
            if 'RealZyy' in dfs_freq.index and 'ImagZyy' in dfs_freq.index:
                Zyy = complex(dfs_freq.loc['RealZyy'].value, 
                              dfs_freq.loc['ImagZyy'].value)
                z[n, 1, 1] = Zyy
                z_err[n, 1, 1] = dfs_freq.loc['RealZyy'].std_error
            
            if 'RealTzy' in dfs_freq.index and 'ImagTzy' in dfs_freq.index:
                Tzy = complex(dfs_freq.loc['RealTzy'].value, 
                              dfs_freq.loc['ImagTzy'].value)
                tipper[n, 0, 1] = Tzy
                tipper_err[n, 0, 1] = dfs_freq.loc['RealTzy'].std_error
            
            if 'RealTzx' in dfs_freq.index and 'ImagTzx' in dfs_freq.index:
                Tzx = complex(dfs_freq.loc['RealTzx'].value, 
                              dfs_freq.loc['ImagTzx'].value)
                tipper[n, 0, 0] = Tzx
                tipper_err[n, 0, 0] = dfs_freq.loc['RealTzx'].std_error
            
        z *= factor
        z_err *= factor
        
        mt_obj = MT(station = station, survey = 'Modelling', frequency = all_frequencies,
                    impedance=z, impedance_error=z_err, tipper=tipper, tipper_error=tipper_err)

        mt_obj_list.append(mt_obj)
        
    return mt_obj_list, all_frequencies

def calculate_rms_Z(mt_obs_list, mt_mod_list, ftol = 0.03):
    
    frequencies_mod = np.array([])
    for mt_obj_modelled in mt_mod_list:
        for frequency in mt_obj_modelled.Z.frequency:
            freq_max = frequency * (1 + ftol)
            freq_min = frequency * (1 - ftol)
            
            fidx = np.where((frequencies_mod < freq_max) & (frequencies_mod > freq_min))[0]
        
            if np.size(fidx)==0:
                frequencies_mod = np.append(frequencies_mod, frequency)
            
    stn_obs_codes = np.array([mt_obs.station for mt_obs in mt_obs_list])

    mse_per_period = np.zeros(shape=(len(frequencies_mod),))
    mse_per_station = np.zeros(shape=(len(stn_obs_codes),))
    mse_total = 0

    n_data_per_period = np.zeros(shape=(len(frequencies_mod),))
    n_data_per_station = np.zeros(shape=(len(stn_obs_codes),))
    
    
    # Compute the misfit for each station
    for mt_obj_modelled in mt_mod_list:
        sidx = np.where( stn_obs_codes == mt_obj_modelled.station )[0]
                
        if np.size(sidx)==0:
            raise Exception('Your observed data file contains stations which are not in the modelled response file. Sure inversion was ran with this data file?')
        else:
            sidx = sidx[0]
            
        mt_obj_observed = mt_obs_list[sidx]
    
        mse_total_station = 0
        for frequency in frequencies_mod:
            freq_max = frequency * (1 + ftol)
            freq_min = frequency * (1 - ftol)
        
            fidx_obs = np.where((mt_obj_observed.Z.frequency < freq_max) & (mt_obj_observed.Z.frequency > freq_min))[0]
        
            if np.size(fidx_obs)==0:
                raise Exception('Your observed data file contains frequencies which are not in the modelled response file or vice versa. Sure inversion was ran with this data file?')
            else:
                fidx_obs = fidx_obs[0]
                
            fidx_mod = np.where((mt_obj_modelled.Z.frequency < freq_max) & (mt_obj_modelled.Z.frequency > freq_min))[0]
            
            if np.size(fidx_mod)==0:
                raise Exception('This should not happen...')
            else:
                fidx_mod = fidx_mod[0]
        
            Z_obs = mt_obj_observed.Z.z[fidx_obs]
            Z_mod = mt_obj_modelled.Z.z[fidx_mod]
            Z_err = mt_obj_observed.Z.z_error[fidx_obs]
        
            mse = 0
            # If Observation are mutted, skip
            if np.all( Z_obs == 0 ):
                continue
                
            # Full Impedance
            if((np.abs(Z_obs[0,0]) > 0.) & (np.abs(Z_obs[1,1]) > 0.)):
                mse = np.divide((Z_obs.real - Z_mod.real)**2, Z_err**2) +\
                      np.divide((Z_obs.imag - Z_mod.imag)**2, Z_err**2)
                mse = np.sum(mse)
                n_data_per_period[fidx_obs] += 8
                n_data_per_station[sidx] += 8
            # Only off-diagonal components
            else:
                mse = np.divide((Z_obs.real - Z_mod.real)**2, Z_err**2) +\
                      np.divide((Z_obs.imag - Z_mod.imag)**2, Z_err**2)
                mse = mse[0,1] + mse[1,0]
                n_data_per_period[fidx_obs] += 4
                n_data_per_station[sidx] += 4
            
            mse_per_period[fidx_obs] += mse
            mse_per_station[sidx] += mse
            mse_total += mse
            mse_total_station += mse

        # remove misfit for station with very large rmse
        #if np.sqrt(mse_total_station/n_data_per_station[sidx]) > 5:
        #    mse_total -= mse_total_station

    rmse_per_period = np.sqrt(np.divide(mse_per_period, n_data_per_period))
    rmse_per_station = np.sqrt(np.divide(mse_per_station, n_data_per_station))
    rmse_total = np.sqrt(mse_total / np.sum(n_data_per_period))
    
    return rmse_total, rmse_per_station, rmse_per_period, 1./ frequencies_mod, stn_obs_codes


def calculate_rms_T(mt_obs_list, mt_mod_list, ftol = 0.03):
    
    frequencies_mod = np.array([])
    for mt_obj_modelled in mt_mod_list:
        for frequency in mt_obj_modelled.Z.freq:
            freq_max = frequency * (1 + ftol)
            freq_min = frequency * (1 - ftol)
            
            fidx = np.where((frequencies_mod < freq_max) & (frequencies_mod > freq_min))[0]
        
            if np.size(fidx)==0:
                frequencies_mod = np.append(frequencies_mod, frequency)
            
    stn_obs_codes = np.array([mt_obs.station for mt_obs in mt_obs_list])

    mse_per_period = np.zeros(shape=(len(frequencies_mod),))
    mse_per_station = np.zeros(shape=(len(stn_obs_codes),))
    mse_total = 0

    n_data_per_period = np.zeros(shape=(len(frequencies_mod),))
    n_data_per_station = np.zeros(shape=(len(stn_obs_codes),))
    
    stations = []
    
    # Compute the misfit for each station
    for mt_obj_modelled in mt_mod_list:
        sidx = np.where( stn_obs_codes == mt_obj_modelled.station )[0]
                
        if np.size(sidx)==0:
            raise Exception('Your modelled data file contains station ' + mt_obj_modelled.station + ' which is not in the observed response file. Sure both files came out of the same inversion run?')
        else:
            sidx = sidx[0]
            
        stations.append(mt_obj_modelled.station)

        mt_obj_observed = mt_obs_list[sidx]
    
        for frequency in frequencies_mod:
            freq_max = frequency * (1 + ftol)
            freq_min = frequency * (1 - ftol)
        
            fidx_obs = np.where((mt_obj_observed.Tipper.freq < freq_max) & (mt_obj_observed.Tipper.freq > freq_min))[0]
        
            if np.size(fidx_obs)==0:
                raise Exception('Your observed data file contains frequencies which are not in the modelled response file or vice versa. Sure inversion was ran with this data file?')
            else:
                fidx_obs = fidx_obs[0]
                
            fidx_mod = np.where((mt_obj_modelled.Tipper.freq < freq_max) & (mt_obj_modelled.Tipper.freq > freq_min))[0]
            
            if np.size(fidx_mod)==0:
                raise Exception('This should not happen...')
            else:
                fidx_mod = fidx_mod[0]
        
            T_obs = mt_obj_observed.Tipper.tipper[fidx_obs]
            T_err = mt_obj_observed.Tipper.tipper_err[fidx_obs]
            
            T_mod = mt_obj_modelled.Tipper.tipper[fidx_mod]
        
            mse = 0
            # Tzx
            if(np.abs(T_obs[0,0]) > 0.):
                mse = (T_obs[0,0].real - T_mod[0,0].real)**2 / T_err[0,0]**2 +\
                      (T_obs[0,0].imag - T_mod[0,0].imag)**2 / T_err[0,0]**2
                n_data_per_period[fidx_obs] += 2
                n_data_per_station[sidx] += 2
                
                mse_per_period[fidx_obs] += mse
                mse_per_station[sidx] += mse
                mse_total += mse
            
            # Tzy
            if(np.abs(T_obs[0,1]) > 0.):
                mse = (T_obs[0,1].real - T_mod[0,1].real)**2 / T_err[0,1]**2 +\
                      (T_obs[0,1].imag - T_mod[0,1].imag)**2 / T_err[0,1]**2
                n_data_per_period[fidx_obs] += 2
                n_data_per_station[sidx] += 2
            
                mse_per_period[fidx_obs] += mse
                mse_per_station[sidx] += mse
                mse_total += mse
        
    rmse_per_period = np.sqrt(np.divide(mse_per_period, n_data_per_period))
    rmse_per_station = np.sqrt(np.divide(mse_per_station, n_data_per_station))
    rmse_total = np.sqrt(mse_total / np.sum(n_data_per_period))
    
    return rmse_total, rmse_per_station, rmse_per_period, 1./ frequencies_mod, stn_obs_codes
