'''
    Various auxiliary routines to work with the GoFEM meshes
    
    Alexander Grayver, 2019 - 2020
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from functools import partial
from shapely.geometry import Point

def plot_2d_triangulation(triangulation, color_scheme = None, edge_color = 'k'):
    
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

    p = PatchCollection(patches, edgecolors=edge_color, facecolors=None)
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

def polar_transform(p):
    '''
        Transform [phi (0..2pi), theta (0..pi), r] to [X,Y,Z] in ECEF frame. 
    '''
    return  [p[2] * math.sin(p[1]) * math.cos(p[0]),
             p[2] * math.sin(p[1]) * math.sin(p[0]),
             p[2] * math.cos(p[1])]

def create_part_shell(phi_limits, theta_limits, r_limits, n_cells_phi, n_cells_theta, dr):
    
    '''
        Create part shell mesh given lateral limits and radial discretization
        Input angles are in radians, radial discretization is in metres
    '''
    
    import PyDealII.Release as dealii
    
    lat_limits = [math.pi/2. - theta_limits[1], math.pi/2. - theta_limits[0]]
    
    p_begin = dealii.Point([phi_limits[0], lat_limits[0], r_limits[0]])
    p_end = dealii.Point([phi_limits[1], lat_limits[1], r_limits[1]])

    dtheta = np.ones(n_cells_theta) * (theta_limits[1] - theta_limits[0]) / n_cells_theta
    dphi = np.ones(n_cells_phi) * (phi_limits[1] - phi_limits[0]) / n_cells_phi

    triangulation = dealii.Triangulation('3D')
    triangulation.generate_subdivided_steps_hyper_rectangle([dphi.tolist(),\
                                                             dtheta.tolist(),\
                                                             np.flip(dr).tolist()], p_begin, p_end, False)
    
    lat2colat = lambda p: [p[0], math.pi / 2. - p[1], p[2]]
    triangulation.transform(lat2colat)
    
    return triangulation

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
                  
    (c) https://gist.github.com/dwyerk/10561690
    """
    
    from shapely.ops import cascaded_union, polygonize
    from scipy.spatial import Delaunay
    import shapely.geometry as geometry

    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

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
        
        
def refine_via_callback(triangulation, refine_callback, repeat = 1):
    '''
        Refine cells for which refine_callback gives true
    '''
    
    dim = triangulation.dim()
        
    for i in range(repeat):
        for cell in triangulation.active_cells():
            if refine_callback(cell):
                cell.refine_flag = 'isotropic'
                    
        triangulation.execute_coarsening_and_refinement()
            
        
def points_in_polygon(points_xy, polygon, quadrat_width):
    
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon, MultiPolygon, Point
    
    gdf_nodes = gpd.GeoDataFrame(data={'x':points_xy[0], 'y':points_xy[1]})
    gdf_nodes.name = 'nodes'
    gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point((row['x'], row['y'])), axis=1)
    
    geometry_cut = ox.utils_geo._quadrat_cut_geometry(polygon, quadrat_width=quadrat_width)

    # build the r-tree index
    sindex = gdf_nodes.sindex

    # find the points that intersect with each subpolygon and add them to points_within_geometry
    points_within = pd.DataFrame()
    for poly in geometry_cut:
        # buffer by the <1 micron dist to account for any space lost in the quadrat cutting
        # otherwise may miss point(s) that lay directly on quadrat line
        poly = poly.buffer(1e-14).buffer(0)

        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = gdf_nodes.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        points_within = points_within.append(precise_matches)

    points_outside = gdf_nodes[~gdf_nodes.isin(points_within)]
    
    return points_within, points_outside


def refine_within_polygon(triangulation, polygon, repeat = 1, z_range = [float('-inf'), float('inf')], n_quadrats = 10):
    '''
        Refine cells for which refine_callback gives true
    '''
    
    import shapely.geometry
    
    dim = triangulation.dim()
    
    assert dim == 3

    qwidth = (polygon.bounds[2] - polygon.bounds[0]) / n_quadrats
    
    for i in range(repeat):
        
        xc = []
        yc = []
        idx = 0
        cell_indices = dict()
        for cell in triangulation.active_cells():    
            center = cell.center().to_list()
            
            if center[2] > z_range[0] and center[2] < z_range[1]:
                xc.append(center[0])
                yc.append(center[1])
                cell_indices[(cell.level(), cell.index())] = idx
                idx += 1
        
        points_in, points_out = points_in_polygon([xc, yc], polygon, quadrat_width = qwidth)
                
        for cell in triangulation.active_cells():
            if (cell.level(), cell.index()) in cell_indices.keys() and\
               cell_indices[(cell.level(), cell.index())] in points_in.index:
                cell.refine_flag = 'isotropic'
                
        triangulation.execute_coarsening_and_refinement()
    
        
def refine_at_polygon_boundary(triangulation, polygon, material_id, center, radii, n_quadrats = 10, repeat = 1, pnorm = 2, top_face = 4):
    '''
        
    '''
    import shapely.geometry
    
    dim = triangulation.dim()
    
    assert dim == 3

    qwidth = (polygon.bounds[2] - polygon.bounds[0]) / n_quadrats
    
    for i in range(repeat):
        
        xc = []
        yc = []
        idx = 0
        cell_indices = dict()
        for cell in triangulation.active_cells():    
            cell_center = cell.center().to_list()
            xc.append(cell_center[0])
            yc.append(cell_center[1])
            
            cell_indices[(cell.level(), cell.index())] = idx
            idx += 1

            for n, face in enumerate(cell.faces()):
                if not face.at_boundary():
                    neighbor = cell.neighbor(n)
                    
                    if (neighbor.level(), neighbor.index()) not in cell_indices:
                        neighbor_center = neighbor.center().to_list()
                        xc.append(neighbor_center[0])
                        yc.append(neighbor_center[1])
                        cell_indices[(neighbor.level(), neighbor.index())] = idx
                        idx += 1
        
        points_in, points_out = points_in_polygon([xc, yc], polygon, quadrat_width = qwidth)
        
        for cell in triangulation.active_cells():
            if cell.material_id == material_id:
                continue
            
            cell_center = cell.center().to_list()
            
            if not is_within_ellipsoid(cell_center[:dim-1], center[:dim-1], radii[:dim-1], pnorm):
                continue
                
            is_cell_inside = cell_indices[(cell.level(), cell.index())] in points_in.index
            
            faces = cell.faces()
            if not faces[top_face].at_boundary():
                top_material_id = cell.neighbor(top_face).material_id
                if top_material_id != material_id:
                    continue
            
            for n, face in enumerate(faces):
                if not face.at_boundary():
                    neighbor = cell.neighbor(n)
                    
                    is_neighbor_inside = cell_indices[(neighbor.level(), neighbor.index())] in points_in.index
                    
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

    def __init__(self, topography, dim, center, radii, pnorm = 2, shoreline = None):
        self.topography = topography
        self.center = center
        self.radii = radii
        self.pnorm = pnorm
        self.dim = dim
        
        self.z_top_sea = 0
        self.z_0_land = 0

        self.shoreline = shoreline
        
    def fit_to(self, triangulation, z_top, z_bottom, z_mean_bathymetry = 0, ignore_bathymetry_distance = 0., inverse = False):
        
        self.z_top_land = z_top
        self.z_bottom = z_bottom
        self.z_0_sea = z_mean_bathymetry
        self.ignore_distance = ignore_bathymetry_distance
        
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
                
        dist_to_shoreline = 1e10
        if(self.shoreline != None and self.ignore_distance > 0):
            if(self.dim == 3):
                dist_to_shoreline = self.shoreline.exterior.distance(Point(p_hat[0], p_hat[1]))
            elif self.dim == 2:
                dist_to_shoreline = abs(p_hat[0] - self.shoreline)

        zt = self.topography(p[:-1])
                
        if dist <= 1.:
            z_hat = z
            if zt < 0: # land
                if (z >= self.z_top_land) and (z <= zt):
                    z_hat = (self.z_0_land * self.z_top_land - self.z_0_land * z + self.z_top_land * z - self.z_top_land * zt) / (self.z_top_land - zt)
                elif(z > zt) and (z <= self.z_bottom):
                    z_hat = (self.z_0_land * self.z_bottom - self.z_0_land * z + self.z_bottom * z - self.z_bottom * zt) / (self.z_bottom - zt)
            elif zt > 0 and dist_to_shoreline > self.ignore_distance: # sea
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

        dist_to_shoreline = 1e10
        if(self.shoreline != None and self.ignore_distance > 0):
            if(self.dim == 3):
                dist_to_shoreline = self.shoreline.exterior.distance(Point(p_hat[0], p_hat[1]))
            elif self.dim == 2:
                dist_to_shoreline = abs(p_hat[0] - self.shoreline)
        
        zt = self.topography(p_hat[:-1])
        
        if dist <= 1.:
            z = z_hat
            if zt < 0: # land
                if (z_hat >= self.z_top_land) and (z_hat <= self.z_0_land):
                    z = (self.z_0_land - zt) / (self.z_0_land - self.z_top_land) * self.z_top_land +\
                        (zt - self.z_top_land) / (self.z_0_land - self.z_top_land) * z_hat
                elif (z_hat > self.z_0_land) and (z_hat <= self.z_bottom):
                    z = (self.z_bottom - zt) / (self.z_bottom - self.z_0_land) * z_hat +\
                        (zt - self.z_0_land) / (self.z_bottom - self.z_0_land) * self.z_bottom
            elif zt > 0 and dist_to_shoreline > self.ignore_distance: # sea
                if (z_hat >= self.z_top_sea) and (z_hat <= self.z_0_sea):
                    z = (self.z_0_sea - zt) / (self.z_0_sea - self.z_top_sea) * self.z_top_sea +\
                        (zt - self.z_top_sea) / (self.z_0_sea - self.z_top_sea) * z_hat
                elif (z_hat > self.z_0_sea) and (z_hat <= self.z_bottom):
                    z = (self.z_bottom - zt) / (self.z_bottom - self.z_0_sea) * z_hat +\
                        (zt - self.z_0_sea) / (self.z_bottom - self.z_0_sea) * self.z_bottom

            p[self.dim - 1] = z
            
        return p
