'''
    Various auxiliary routines to work with the GoFEM data
    
    Alexander Grayver, 2020
'''
import numpy as np
import pandas as pd

def read_modeling_output(file_format, frequencies):
    """
        Read GoFEM forward modelling output. The GoFEM
        file name convention is the following:
        
        '%prefix%_c=%d_f=%.8e.txt', where %prefix% is an arbitrary
        string specified by the user in the *.prm file,
        'c=%d' specifies cycle number (in case of adaptive mesh 
        refinement, more than one cycle is typically done to eventually
        get accurate responses) and 'f=%.8e' is the frequency.
        
        :Example: ::
        
        >>> read_modeling_output('./results/output_c=0_f=%.8e.txt', 
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
        df_num.insert(2, df.columns[2], df[df.columns[2]])
        
        for col_idx in range(3, len(df.columns)):
            col_str = df[df.columns[col_idx]]
            col_num = np.zeros(shape=(len(col_str),), dtype = np.complex128)
            
            for row_idx in range(len(col_str)):
                col_num[row_idx] = complex(*eval(col_str[row_idx]))
                
            df_num.insert(col_idx, df.columns[col_idx], col_num)
            
        data_frames.append(df_num)
            

    return data_frames