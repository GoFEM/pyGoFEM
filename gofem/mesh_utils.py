'''
    Various auxiliary routines to work with the GoFEM models
    
    Alexander Grayver, 2019
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot_2d_triangulation(triangulation, color_scheme = None):
    
    fig, ax = plt.subplots()
    patches = []
    colors = []

    for cell in triangulation.active_cells():
        quad_vertices = np.zeros((4,2))
        # The shift variable is used to reorder the vertices because
        # deal.II and matplotlib require different ordering
        shift = [0,1,3,2]
        for i in range(4):
            vertex = cell.get_vertex(i)
            quad_vertices[shift[i]][0] = vertex.x
            quad_vertices[shift[i]][1] = vertex.y
        quad = Polygon(quad_vertices, closed=True)
        patches.append(quad)
        
        if color_scheme:
            colors.append(color_scheme(cell))

    p = PatchCollection(patches, edgecolors='k', facecolors=None)
    p.set_array(np.array(colors))
    
    ax.add_collection(p, autolim=True)
    ax.autoscale_view()  
    ax.invert_yaxis()
    
    ax.set_xlabel('y (m)')
    ax.set_ylabel('z (m)')
        
    return fig, ax
    

def refine_at_interface(triangulation, material_ids, repeat = 1, center = None, radius = None, pnorm = 2):
    '''
        Refine cells with provided material ids in triangulation that neighbor cells with
        other ids. Repeat several times if requested. Only cells that are within
        a given radius are refined.
    '''
    
    dim = triangulation.dim()
    
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if not cell.material_id in material_ids:
                continue
                
            dist = 0
            if center and radius:
                p_center = cell.center().to_list()
                for d in range(dim-1):
                    dist += abs(center[d] - p_center[d])**pnorm / radius[d]**pnorm
                    
            if not dist <= 1.:
                continue
                
            faces = cell.faces()
            for n in range(len(faces)):
                if not faces[n].at_boundary():
                    neighbor = cell.neighbor(n)
                    if not neighbor.material_id in material_ids:
                        cell.refine_flag = 'isotropic'
                            
        triangulation.execute_coarsening_and_refinement()
        
        
def refine_around_points(triangulation, points, center, radius, repeat = 1, exclude_materials = [], pnorm = 2):
    '''
        Refine cells around points Repeat several times if requested. 
        Only cells that are within a given radius are refined.
    '''
    
    dim = triangulation.dim()
        
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if cell.material_id in exclude_materials:
                continue
                
            center = cell.center().to_list()
            for point in points:
                dist = 0
                for d in range(dim):
                    dist += abs(center[d] - point[d])**pnorm / radius[d]**pnorm
                
                if dist <= 1.:
                    cell.refine_flag = 'isotropic'
                    
        triangulation.execute_coarsening_and_refinement()