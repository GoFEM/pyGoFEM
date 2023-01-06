'''
    Various auxiliary routines to work with the GoFEM data
    
    Alexander Grayver, 2020-2021
'''
import numpy as np
import pandas as pd

def read_modeling_responses_bin(file_format, frequencies):
    
    n_columns = 28
    n_frequencies = len(frequencies)
    for fidx in range(n_frequencies):
        filename = file_format % frequencies[fidx]
        with open(filename, "rb") as f:
            data = np.fromfile(f, dtype=np.float64)
            data = np.reshape(data, [round(data.size/n_columns), n_columns])

            if fidx == 0:
                coords = data[:,0:3]
                weights = data[:,27]
                
                E = np.zeros(shape=(n_frequencies,data.shape[0], 3), dtype=np.complex128)
                H = np.zeros(shape=(n_frequencies,data.shape[0], 3), dtype=np.complex128)
                Ep = np.zeros(shape=(n_frequencies,data.shape[0], 3), dtype=np.complex128)
                Hp = np.zeros(shape=(n_frequencies,data.shape[0], 3), dtype=np.complex128)
                
            E[fidx].real = data[:,[3,5,7]]
            E[fidx].imag = data[:,[4,6,8]]
            H[fidx].real = data[:,[9,11,13]]
            H[fidx].imag = data[:,[10,12,14]]

            Ep[fidx].real = data[:,[15,17,19]]
            Ep[fidx].imag = data[:,[16,18,20]]
            Hp[fidx].real = data[:,[21,23,25]]
            Hp[fidx].imag = data[:,[22,24,26]]
            
    return coords, E, H, Ep, Hp, weights

def calculate_MT_impedance(E1, H1, E2, H2, omega):
    
    mu0 = 4*np.pi*1e-7
    Z = np.zeros(shape=(E1.shape[0], 4), dtype=np.complex128)

    H_i = np.zeros(shape=(2,2), dtype = np.complex128)
    E_i = np.zeros(shape=(2,2), dtype = np.complex128)

    for i in range(E1.shape[0]):
        H_i[0,0] = -H1[i,1]
        H_i[0,1] = -H2[i,1]
        H_i[1,0] = H1[i,0]
        H_i[1,1] = H2[i,0]
        
        E_i[0,0] = -E1[i,1]
        E_i[0,1] = -E2[i,1]
        E_i[1,0] = E1[i,0]
        E_i[1,1] = E2[i,0]

        Z_i = np.matmul(E_i, np.linalg.inv(H_i))

        Z[i,0] = Z_i[0,0]
        Z[i,1] = Z_i[0,1]
        Z[i,2] = Z_i[1,0]
        Z[i,3] = Z_i[1,1]
    
    rho_app = np.abs(Z)**2 / (mu0 * omega)
    phase = np.arctan2(Z.imag,Z.real) * 180/np.pi
    
    return Z, rho_app, phase

def read_modeling_responses(file_format, frequencies):
    """
        Read GoFEM forward modelling output. The GoFEM
        file name convention is the following:
        
        '%prefix%_c=%d_f=%.8e.txt', where %prefix% is an arbitrary
        string specified by the user in the *.prm file,
        'c=%d' specifies cycle number (in case of adaptive mesh 
        refinement, more than one cycle is typically done to eventually
        get accurate responses) and 'f=%.8e' is the frequency.
        
        :Example: ::
        
        >>> read_modeling_responses('./results/output_c=0_f=%.8e.txt', 
                                    frequencies = [0.1, 0.01])
        
        This call will look for the following files:
        
        ./results/output_c=0_f=1.00000000e-01.txt
        ./results/output_c=0_f=1.00000000e-02.txt
    """
    
    data_frames = []
    
    for fidx in range(len(frequencies)):
        df = pd.read_csv(file_format % frequencies[fidx], sep="\t")
        
        df_num = pd.DataFrame()
        df_num.insert(0, df.columns[0], df[df.columns[0]])
        df_num.insert(1, df.columns[1], df[df.columns[1]])
        
        for col_idx in range(2, len(df.columns)):
            col_str = df[df.columns[col_idx]]
            col_num = np.zeros(shape=(len(col_str),), dtype = np.complex128)
            
            for row_idx in range(len(col_str)):
                col_num[row_idx] = complex(*eval(col_str[row_idx]))
                
            df_num.insert(col_idx, df.columns[col_idx], col_num)
            
        data_frames.append(df_num)
            

    return data_frames


def read_inversion_responses(datafile):
    """
        Read GoFEM modelled data from inversion. The GoFEM
        file name convention is the following.
    """
    
    df = pd.DataFrame()
    df = pd.read_csv(file_format % frequencies[fidx], sep="\t")
    
    
def calculate_phase_tensor(Z, dZ = None):
    
    PT = np.zeros(shape=(2,2))
    dPT = np.zeros(shape=(2,2))
    
    #PT = np.linalg.inv(Z.real) * Z.imag
    detX = Z[0,0].real*Z[1,1].real - Z[1,0].real*Z[0,1].real

    PT[0, 0] = Z.real[1, 1] * Z.imag[0, 0] - Z.real[0, 1] * Z.imag[1, 0]
    PT[0, 1] = Z.real[1, 1] * Z.imag[0, 1] - Z.real[0, 1] * Z.imag[1, 1]
    PT[1, 0] = Z.real[0, 0] * Z.imag[1, 0] - Z.real[1, 0] * Z.imag[0, 0]
    PT[1, 1] = Z.real[0, 0] * Z.imag[1, 1] - Z.real[1, 0] * Z.imag[0, 1]

    PT /= detX
        
    if dZ is None:
        return PT
    else:
        dPT = np.zeros(shape=(2,2))
        
        dPTdX = np.zeros(shape=(2, 2, 2, 2))
        dPTdY = np.zeros(shape=(2, 2, 2, 2))

        # dPTxx
        dPTdX[0,0,0,0] =(-PT[0,0] * Z[1,1].real) / detX;
        dPTdX[0,0,0,1] =( PT[0,0] * Z[1,0].real - Z[1,0].imag) / detX;
        dPTdX[0,0,1,0] =( PT[0,0] * Z[0,1].real) / detX;
        dPTdX[0,0,1,1] =(-PT[0,0] * Z[0,0].real + Z[0,0].imag) / detX;

        dPTdY[0,0,0,0] = Z[1,1].real / detX;
        dPTdY[0,0,0,1] = 0;
        dPTdY[0,0,1,0] =-Z[0,1].real / detX;
        dPTdY[0,0,1,1] = 0;

        # dPTxy
        dPTdX[0,1,0,0] = (-PT[0,1] * Z[1,1].real) / detX;
        dPTdX[0,1,0,1] = ( PT[0,1] * Z[1,0].real - Z[1,1].imag) / detX;
        dPTdX[0,1,1,0] = ( PT[0,1] * Z[0,1].real) / detX;
        dPTdX[0,1,1,1] = (-PT[0,1] * Z[0,0].real + Z[0,1].imag) / detX;

        dPTdY[0,1,0,0] = 0;
        dPTdY[0,1,0,1] = Z[1,1].real / detX;
        dPTdY[0,1,1,0] = 0;
        dPTdY[0,1,1,1] =-Z[0,1].real / detX;

        # dPTyx
        dPTdX[1][0,0][0] = (-PT[1,0] * Z[1,1].real + Z[1,0].imag) / detX;
        dPTdX[1][0,0][1] = ( PT[1,0] * Z[1,0].real) / detX;
        dPTdX[1,0][1,0] = ( PT[1,0] * Z[0,1].real - Z[0,0].imag) / detX;
        dPTdX[1,0][1,1] = (-PT[1,0] * Z[0,0].real) / detX;

        dPTdY[1][0,0][0] =-Z[1,0].real / detX;
        dPTdY[1][0,0][1] = 0;
        dPTdY[1,0][1,0] = Z[0,0].real / detX;
        dPTdY[1,0][1,1] = 0;

        # dPTyy
        dPTdX[1,1][0,0] = (-PT[1,1] * Z[1,1].real + Z[1,1].imag) /  detX;
        dPTdX[1,1][0,1] = ( PT[1,1] * Z[1,0].real) / detX;
        dPTdX[1,1][1,0] = ( PT[1,1] * Z[0,1].real - Z[0,1].imag) / detX;
        dPTdX[1,1][1,1] = (-PT[1,1] * Z[0,0].real) / detX;

        dPTdY[1,1][0,0] = 0;
        dPTdY[1,1][0,1] =-Z[1,0].real / detX;
        dPTdY[1,1][1,0] = 0;
        dPTdY[1,1][1,1] = Z[0,0].real / detX;

        for k in range(2):
            for l in range(2):
                propagated_error = 0
                for i in range(2):
                    for j in range(2):
                        propagated_error += (dPTdY[k,l,i,j] ** 2.)*(dZ[i,j] ** 2.) + (dPTdX[k,l,i,j] ** 2.)*(dZ[i,j] ** 2.)

                dPT[k, l] = np.sqrt(propagated_error)
        
        return PT, dPT