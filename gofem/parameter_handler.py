'''
    Handles parameters for the *.prm files suitable for the 
    GoFEM forward modeling and inversion
    
    Alexander Grayver, 2019
'''


class ParameterHandler:
    '''
        This class manages GoFEM parameters. It can read and write
        the relevant prm files that can be passed to the GoFEM.
        
        The parameters are grouped into several categories:
        - Modeling: general modeling parameters
        - 2.5D: parameters that are specific to the 2.5D modeling
        - Solver: solver parameters
        - Model: parameters relevant for the model (mesh, physical properties, etc)
        - Output: parameters that tell GoFEM if and how to write results
        - Inversion: inversion specific parameters (ignored for forward modeling)
    '''
    
    def __init__(self, inversion = False, dim = 3):
        self.parameters = {'Modeling': dict(),\
                           'Solver': dict(),\
                           'Model': dict(),\
                           'Survey': dict(),\
                           'Output': dict()}
        
        self.parameters['Modeling'] = {'Adaptive strategy': 'global',\
                                      'Theta': 0.12,\
                                      'Refinement strategy':'Number',\
                                      'Number of refinements':0,\
                                      'Error reduction':1000,\
                                      'DoFs budget':10000000,\
                                      'Number of initial refinements':0,\
                                      'Order':1,\
                                      'BC':'Dirichlet',\
                                      'Number of parallel frequencies':1,\
                                      'Refine cells around receivers':0,\
                                      'Field formulation':'E',\
                                      'Field approach':'Total'}
        
        if dim == 2:
            self.parameters['2.5D'] = {'Minimum and maximun wavenumbers': [1e-5, 1e-1],\
                                       'Number of wavenumbers': 30,\
                                       'Strike filter': '',\
                                       'Time filter': ''}
            
        self.parameters['Solver'] = {'Preconditioner': 'Direct',\
                                    'Iterations': 100,\
                                    'Residual': 1e-9,\
                                    'Adjoint residual': 1e-6,\
                                    'Preconditioner iterations': 30,\
                                    'Preconditioner residual': 1e-2}
            
        self.parameters['Model'] = {'Model definition file': '',\
                                    'Inversion model definition file': '',\
                                    'Materials definition file': '',\
                                    'Inversion materials definition file': '',\
                                    'Background model definition file': '',\
                                    'Background materials definition file': '',\
                                    'Active domain box': [0,0,0,0,0,0],\
                                    'Active domain mask': ''}
            
        self.parameters['Survey'] = {'Frequencies file': '',\
                                    'Times file': '',\
                                    'Stations file': '',\
                                    'Sources file': '',\
                                    'Sources-receiver map': ''}
            
        self.parameters['Output'] = {'Type': 'point',\
                                    'Data file': '',\
                                    'Mesh order': 1,\
                                    'Parallel output': True,\
                                    'Sources-receiver map': ''}
            
        if inversion:
            self.parameters['Inversion'] = {'Inversion input data': '',\
                                            'Number of iterations': 20,\
                                            'Target RMS': 1.,\
                                            'Scaling factor': [1.],\
                                            'Regularization operator': 'Roughness',\
                                            'Regularization update': 1,\
                                            'Use starting model as reference': False,\
                                            'Face weighting': [1.,1.,1.],\
                                            'Non-conforming interface weighting': 1.,\
                                            'Cell weighting': [0.,0.,0.,1.],\
                                            'Model transformation': 'BOUNDED',
                                            'Minimum conductivity': 1e-5,\
                                            'Maximum conductivity': 1e2,\
                                            'Number of inner iterations': 20,\
                                            'Inner solver type': 'Krylov',\
                                            'Steplength iterations': 1,\
                                            'Step lengths file': '',\
                                            'Output files prefix': 'inv'}
        
        
    def getitem(self, section, parameter):
        if section in self.parameters:
            if parameter in self.parameters[section]:
                return self.parameters[section][parameter]
            else:
                raise Exception('Parameter %s is not a valid name for section %s.' % (section, parameter))
        else:
            raise Exception('Section {} is not a valid section name.'.format(section))
                
                
    def setitem(self, section, parameter, value):
        if section in self.parameters:
            if parameter in self.parameters[section]:
                self.parameters[section][parameter] = value
            else:
                raise Exception('Parameter %s is not a valid name for section %s.' % (section, parameter))
        else:
            raise Exception('Section {} is not a valid section name.'.format(section))
            
        
    def write(self,filename):
        with open(filename, 'w') as f:
            for section,parameters in self.parameters.items():
                f.write('subsection {} parameters\n'.format(section))
                for pname,pvalue in parameters.items():
                    if not pvalue:
                        continue
                    
                    if type(pvalue) is list:
                        f.write('\tset {} = {}\n\n'.format(pname, ', '.join(map(str, pvalue))))
                    elif type(pvalue) is bool:
                        f.write('\tset {} = {}\n\n'.format(pname, 'true' if pvalue else 'false'))
                    else:
                        f.write('\tset {} = {}\n\n'.format(pname, pvalue))
                f.write('end\n\n\n')
                    
            
            
            
            
        
        