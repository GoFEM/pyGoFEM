'''
    Various auxiliary routines to work with the GoFEM models
    
    Alexander Grayver, 2019
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from functools import partial

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
        
def project_points_on_interface(triangulation, points, material_id, mapping):
    '''
        Project points to the given material interface. During this
        procedure, only the vertical coordinate of the points is
        adjusted. Normally, material id given to this function will
        correspond to the air or sea, such that points are projected
        to the ground or seafloor. 
    '''
    
    dim = triangulation.dim()
    
    projected_points = []
    
    for cell in triangulation.active_cells():
        face_no = 0
        for face in cell.faces():            
            if not face.at_boundary():
                my_material = cell.material_id
                neighbor_material = cell.neighbor(face_no).material_id
                
                if (my_material != neighbor_material) and\
                   (my_material != material_id) and\
                   (neighbor_material == material_id):
                    
                    face_v = []
                    for i in range(2**(dim-1)):
                        face_v.append(face.get_vertex(i))
                        
                    for point in points:
                        px = point[0];
                        py = point[1];

                        within = (px >= face_v[0].x) and (px < face_v[1].x)
                        if dim == 3:
                            within &= (py >= face_v[0].y) and (py < face_v[2].y)

                        if not within:
                            continue
                            
                        p_cell = cell.center(True)
                        p_cell.x = px
                        if dim == 3:
                            p_cell.y = py

                        point_on_face = mapping.project_real_point_to_unit_point_on_face(cell, face_no, p_cell);
                        projected_points.append(mapping.transform_unit_to_real_cell (cell, point_on_face).to_list());

            
            face_no += 1
            
    return projected_points
        
class Topography:
    def __init__(self, topography, dim, center, radius, pnorm = 2):
        self.topography = topography
        self.center = center
        self.radius = radius
        self.pnorm = pnorm
        self.dim = dim
        
        self.z_top_sea = 0
        self.z_0_land = 0
        
    def fit_to(self, triangulation, z_top, z_bottom, z_mean_bathymetry = 0, inverse = False):
        
        self.z_top_land = z_top
        self.z_bottom = z_bottom
        self.z_0_sea = z_mean_bathymetry
        
        if inverse:
            transformation = partial(self.__pull_back)
        else:
            transformation = partial(self.__push_forward)
            
        triangulation.transform(transformation)
        
        
    def __pull_back(self, p):
        
        z = p[self.dim - 1]
        p_hat = p
        
        dist = 0
        for d in range(self.dim):
            dist += abs(self.center[d] - p[d])**self.pnorm / self.radius[d]**self.pnorm
                
        zt = self.topography(p[:-1])
                
        if dist <= 1.:
            if zt < 0: # land
                if (z - zt) < 0: 
                    z_hat = self.z_top_land * (z - zt) / (self.z_top_land + zt) # z is above ground
                else:
                    z_hat = self.z_bottom * (z - zt) / (self.z_bottom - zt) # z is below ground
            elif zt > 0: # sea
                if (z >= self.z_top_sea) and (z <= self.z_0_sea):
                    z_hat = (self.z_top_sea * (self.z_0_sea - zt)) / (self.z_top_sea - zt)
                elif(z > self.z_0_sea) and (z <= self.z_bottom):
                    z_hat = (self.z_bottom * (self.z_0_sea - zt)) / (self.z_bottom - zt)
            else:
                z_hat = z

            p_hat[self.dim - 1] = z_hat
            
        return p_hat
        
        
    def __push_forward(self, p_hat):
        
        z_hat = p_hat[self.dim - 1]
        p = p_hat
        
        dist = 0
        for d in range(self.dim):
            dist += abs(self.center[d] - p[d])**self.pnorm / self.radius[d]**self.pnorm
        
        zt = self.topography(p_hat[:-1])
        
        if dist <= 1.:
            if zt < 0: # land
                if z_hat < 0: 
                    z = z_hat + (z_hat + self.z_top_land) / (self.z_top_land - self.z_0_land) * zt
                else:
                    z = z_hat - (z_hat - self.z_bottom) / (self.z_bottom - self.z_0_land) * zt
            elif zt > 0: # sea
                if (z >= self.z_top_sea) and (z <= self.z_0_sea):
                    z = (self.z_0_sea - zt) / (self.z_0_sea - self.z_top_sea) * self.z_top_sea +\
                    (zt - self.z_top_sea) / (self.z_0_sea - self.z_top_sea) * z_hat
                elif(z > self.z_0_sea) and (z <= self.z_bottom):
                    z = (self.z_bottom - zt) / (self.z_bottom - self.z_0_sea) * z_hat +\
                        (zt - self.z_0_sea) / (self.z_bottom - self.z_0_sea) * self.z_bottom
            else:
                z = z_hat

            p[self.dim - 1] = z
            
        return p