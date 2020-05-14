'''
    Various auxiliary routines to work with the GoFEM
    
    Alexander Grayver, 2019 - 2020
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
    
    ax.add_collection(p)
    ax.autoscale_view()  
    ax.invert_yaxis()
    
    ax.set_xlabel('y (m)')
    ax.set_ylabel('z (m)')
        
    return fig, ax

def is_within_ellipsoid(point, center, radii, pnorm = 2):
    '''
        Returns true if a point is within an ellipsoid. Ellipsoid is
        centred at center and with radii along principal axes. 
        For pnorm = 1 the ellipsoid degenerated into a parallelepiped.
    '''   
    
    assert len(point) == len(center) == len(radii)
    
    assert pnorm in [1, 2]
    
    dim = len(point)
    
    dist = 0
    within = True

    if(pnorm > 1.):
        for d in range(dim):
            dist += (abs(center[d] - point[d]) / radii[d])**pnorm
        
        if not dist <= 1.:
            within = False
            
    elif pnorm == 1:        
        for d in range(dim):
            within &= abs(center[d] - point[d]) < radii[d]

    return within


def refine_at_interface(triangulation, material_ids, repeat = 1, center = None, radii = None, pnorm = 2):
    '''
        Refine cells with provided material ids in triangulation that neighbor cells with
        other ids. Repeat several times if requested. Only cells that are within
        a given radii are refined.
    '''
    
    dim = triangulation.dim()
    
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if not cell.material_id in material_ids:
                continue
                
            if center and radii:
                p_center = cell.center().to_list()
                if not is_within_ellipsoid(p_center[:dim-1], center[:dim-1], radii[:dim-1], pnorm):
                    continue

            for n, face in enumerate(cell.faces()):
                if not face.at_boundary():
                    neighbor = cell.neighbor(n)
                    if not neighbor.material_id in material_ids:
                        cell.refine_flag = 'isotropic'
                            
        triangulation.execute_coarsening_and_refinement()
        
        
def refine_around_points(triangulation, points, center, radii, repeat = 1, exclude_materials = [], pnorm = 2):
    '''
        Refine cells around points Repeat several times if requested. 
        Only cells that are within a given radii are refined.
    '''
    
    dim = triangulation.dim()
        
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if cell.material_id in exclude_materials:
                continue
                
            center = cell.center().to_list()
            for point in points:
                if is_within_ellipsoid(point, center, radii, pnorm):
                    cell.refine_flag = 'isotropic'
                    
        triangulation.execute_coarsening_and_refinement()
        
def refine_at_polygon_boundary(triangulation, polygon, material_id, center, radii, repeat = 1, pnorm = 2):
    '''
        
    '''
    import shapely.geometry
    
    dim = triangulation.dim()
    
    assert dim == 3
    
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if cell.material_id == material_id:
                continue
            
            cell_center = cell.center().to_list()
            
            if not is_within_ellipsoid(cell_center[:dim-1], center[:dim-1], radii[:dim-1], pnorm):
                continue
                
            point = shapely.geometry.Point(cell_center[0], cell_center[1])
            is_cell_inside = polygon.contains(point)
            
            faces = cell.faces()
            top_face = 4
            if not faces[top_face].at_boundary():
                top_material_id = cell.neighbor(top_face).material_id
                if top_material_id != material_id:
                    continue
            
            for n, face in enumerate(faces):
                if not face.at_boundary():
                    neighbor = cell.neighbor(n)
                    
                    neighbor_center = neighbor.center().to_list()
                    
                    point = shapely.geometry.Point(neighbor_center[0], neighbor_center[1])
                    is_neighbor_inside = polygon.contains(point)
                    
                    if is_cell_inside != is_neighbor_inside:
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
    def __init__(self, topography, dim, center, radii, pnorm = 2):
        self.topography = topography
        self.center = center
        self.radii = radii
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
            dist += abs(self.center[d] - p[d])**self.pnorm / self.radii[d]**self.pnorm
                
        zt = self.topography(p[:-1])
                
        if dist <= 1.:
            z_hat = z
            if zt < 0: # land
                if (z - zt) < 0: 
                    z_hat = self.z_top_land * (z - zt) / (self.z_top_land + zt) # z is above ground
                else:
                    z_hat = self.z_bottom * (z - zt) / (self.z_bottom - zt) # z is below ground
            elif zt > 0: # sea
                if (z >= self.z_top_sea) and (z <= zt):
                    z_hat = (self.z_0_sea * self.z_top_sea - self.z_0_sea * z + self.z_top_sea * z - self.z_top_sea * zt) / (self.z_top_sea - zt)
                elif(z > zt) and (z <= self.z_bottom):
                    z_hat = (self.z_0_sea * self.z_bottom - self.z_0_sea * z + self.z_bottom * z - self.z_bottom * zt) / (self.z_bottom - zt)
            else:
                z_hat = z

            p_hat[self.dim - 1] = z_hat
            
        return p_hat
        
        
    def __push_forward(self, p_hat):
        
        z_hat = p_hat[self.dim - 1]
        p = p_hat
        
        dist = 0
        for d in range(self.dim):
            dist += abs(self.center[d] - p[d])**self.pnorm / self.radii[d]**self.pnorm
        
        zt = self.topography(p_hat[:-1])
        
        if dist <= 1.:
            z = z_hat
            if zt < 0: # land
                if z_hat < 0: 
                    z = z_hat + (z_hat + self.z_top_land) / (self.z_top_land - self.z_0_land) * zt # z is above ground
                else:
                    z = z_hat - (z_hat - self.z_bottom) / (self.z_bottom - self.z_0_land) * zt # z is below ground
            elif zt > 0: # sea
                if (z_hat >= self.z_top_sea) and (z_hat <= self.z_0_sea):
                    z = (self.z_0_sea - zt) / (self.z_0_sea - self.z_top_sea) * self.z_top_sea +\
                        (zt - self.z_top_sea) / (self.z_0_sea - self.z_top_sea) * z_hat
                elif (z_hat > self.z_0_sea) and (z_hat <= self.z_bottom):
                    z = (self.z_bottom - zt) / (self.z_bottom - self.z_0_sea) * z_hat +\
                        (zt - self.z_0_sea) / (self.z_bottom - self.z_0_sea) * self.z_bottom

            p[self.dim - 1] = z
            
        return p