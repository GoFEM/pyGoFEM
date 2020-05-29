'''
    Various auxiliary routines to work with the MTpy for GoFEM
    
    Alexander Grayver, 2020
'''
import math
import numpy as np
import pandas as pd

import mtpy.analysis.pt as pt
from mtpy.core import mt as mt
from mtpy.core import z as mtz

from mtpy.core.edi_collection import EdiCollection

from gofem.data_utils import *

def write_edi_collection_to_gofem(edi_collection, outfile, error_floor, data_type = 'Z'):
    '''
        Write down MT impdeance tensors from the edi collection 
        to the GoFEM data file with the given error floor.
        
        Error floor is applied to the impdeance (or derived) row-wise.
        
        @p data_type can take several values:
        'Z' -- write full impedance
        'Z_offdiag' -- write only off diagonal elements
        'RP' -- write apparent resistivity and phase
        'RP_offdiag' -- write rho and phase for off-diag elements
        'PT' -- phase tensor
        'Tipper' -- write vertical magnetic TFs
    '''

    mt_data = []
    ptol = 0.03
    
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
        
    rho_err_func = lambda z, ze, freq: 2. * abs(z) / (2. * math.pi * freq * mu) * ze;
    phi_err_func = lambda z, ze: 180. / math.pi * ze / abs(z);
    
    # Convert from [mV/km]/[nT] to Ohm
    factor = (4 * math.pi) / 10000.0
    
    for freq in edi_collection.all_frequencies:
        for mt_obj in edi_collection.mt_obj_list:
            freq_max = freq * (1 + ptol)
            freq_min = freq * (1 - ptol)
            f_index_list = np.where((mt_obj.Z.freq < freq_max) & (mt_obj.Z.freq > freq_min))[0]
            
            #print(f_index_list, freq_max, freq_min, len(f_index_list), type(f_index_list))
            
            if f_index_list.size > 1:
                print("more than one freq found %s", f_index_list)

            if f_index_list.size < 1:
                continue
                
            p_index = f_index_list[0]
            
            zobj = mt_obj.Z
            tobj = mt_obj.Tipper
            ptobj = mt_obj.pt
            
            dZxy = max([zobj.z_err[p_index, 0, 1], error_floor * abs(zobj.z[p_index, 0, 1])]) * factor
            dZyx = max([zobj.z_err[p_index, 1, 0], error_floor * abs(zobj.z[p_index, 1, 0])]) * factor
                        
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
                mt_data.append([str_codes[0], freq, mt_obj.station, zobj.resistivity[p_index, 0, 1], zobj.resistivity_err[p_index, 0, 1]])
                mt_data.append([str_codes[1], freq, mt_obj.station, zobj.resistivity[p_index, 1, 0], zobj.resistivity_err[p_index, 1, 0]])
                mt_data.append([str_codes[2], freq, mt_obj.station, zobj.phase[p_index, 0, 1], zobj.phase_err[p_index, 0, 1]])
                mt_data.append([str_codes[3], freq, mt_obj.station, zobj.phase[p_index, 1, 0], zobj.phase_err[p_index, 1, 0]])
                
                if data_type == 'RP':
                    mt_data.append([str_codes[4], freq, mt_obj.station, zobj.resistivity[p_index, 0, 0], zobj.resistivity_err[p_index, 0, 0]])
                    mt_data.append([str_codes[5], freq, mt_obj.station, zobj.resistivity[p_index, 1, 1], zobj.resistivity_err[p_index, 1, 1]])
                    mt_data.append([str_codes[6], freq, mt_obj.station, zobj.phase[p_index, 0, 0], zobj.phase_err[p_index, 0, 0]])
                    mt_data.append([str_codes[7], freq, mt_obj.station, zobj.phase[p_index, 1, 1], zobj.phase_err[p_index, 1, 1]])
                    
            elif data_type == 'Tipper':
                mt_data.append([str_codes[0], freq, mt_obj.station, tobj.tipper[p_index, 0, 0].real, tobj.tipper_err[p_index, 0, 0]])
                mt_data.append([str_codes[1], freq, mt_obj.station, tobj.tipper[p_index, 0, 0].imag, tobj.tipper_err[p_index, 0, 0]])
                mt_data.append([str_codes[2], freq, mt_obj.station, tobj.tipper[p_index, 0, 1].real, tobj.tipper_err[p_index, 0, 1]])
                mt_data.append([str_codes[3], freq, mt_obj.station, tobj.tipper[p_index, 0, 1].imag, tobj.tipper_err[p_index, 0, 1]])

            elif data_type == 'PT':
                mt_data.append([str_codes[0], freq, mt_obj.station, ptobj.pt[p_index, 0, 0].real, ptobj.pt_err[p_index, 0, 0]])
                mt_data.append([str_codes[1], freq, mt_obj.station, ptobj.pt[p_index, 0, 1].imag, ptobj.pt_err[p_index, 0, 1]])
                mt_data.append([str_codes[2], freq, mt_obj.station, ptobj.pt[p_index, 1, 0].real, ptobj.pt_err[p_index, 1, 0]])
                mt_data.append([str_codes[3], freq, mt_obj.station, ptobj.pt[p_index, 1, 1].imag, ptobj.pt_err[p_index, 1, 1]])
    
    with open(outfile, 'w') as f:
        f.write('# DataType Frequency Source Receiver Value Error\n')
        idx = 0
        for idx in range(len(mt_data)):
            data_row = mt_data[idx]
            f.write("%s %0.6e Plane_wave %s %0.6e %0.6e\n" % (data_row[0], data_row[1], data_row[2], data_row[3], data_row[4]))


def read_gofem_modelling_output(fileformat, frequency_list, station_list, station_coords):
    '''
        Read in GoFEM and return a list of MT object list
    '''
        
    dfs = read_modeling_output(fileformat, frequency_list)
    
    assert len(dfs) == len(frequency_list)
    
    z_dummy = np.zeros((len(frequency_list), 2, 2), dtype='complex')
    t_dummy = np.zeros((len(frequency_list), 1, 2), dtype='complex')
    
    # Conversion factor from Ohm to [mV/km]/[nT]
    factor = 10000.0 / (4 * math.pi)
    
    data_dict = {}
    for station, xyz in zip(station_list, station_coords):
        data_dict[station] = mt.MT()
        data_dict[station].Z = mtz.Z(z_array=z_dummy.copy(),
                                     z_err_array=z_dummy.copy().real,
                                     freq=np.array(frequency_list))
        data_dict[station].Tipper = mtz.Tipper(tipper_array=t_dummy.copy(),
                                               tipper_err_array=t_dummy.copy().real,
                                               freq=np.array(frequency_list))
        
        data_dict[station].station = station
        #data_dict[station].east = xyz[1]
        #data_dict[station].north = xyz[0]
        #data_dict[station].elev = -xyz[2]
        
    for fidx in range(len(frequency_list)):
        for index, row in dfs[fidx].iterrows():
            if row['Receiver'] not in data_dict.keys():
                raise Exception('Station', row['Receiver'], ' is not in the station list.')
                
            station = row['Receiver']
            
            data_dict[station].Z.z[fidx, 0, 0] = row['Zxx']
            data_dict[station].Z.z[fidx, 0, 1] = row['Zxy']
            data_dict[station].Z.z[fidx, 1, 0] = row['Zyx']
            data_dict[station].Z.z[fidx, 1, 1] = row['Zyy']
            
            data_dict[station].Tipper.tipper[fidx, 0, 0] = row['Tzx']
            data_dict[station].Tipper.tipper[fidx, 0, 1] = row['Tzy']
            
    mt_obj_list = []
    mu0 = 4. * np.pi * 1e-7
    for station, mt_obj in data_dict.items():
        
        mt_obj.Z.z *= factor
        
        mt_obj.Z.compute_resistivity_phase()
        mt_obj.pt.set_z_object(mt_obj.Z)
        mt_obj.Tipper.compute_amp_phase()
        mt_obj.Tipper.compute_mag_direction()
        
        mt_obj_list.append(mt_obj)
    
    return mt_obj_list