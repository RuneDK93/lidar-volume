import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

def compute_volume(data,true_volume,print_flag=False):
    """
    Compute the volume of a point cloud using a convex hull.

    Parameters:
    - data: Point cloud object.
    - true_volume: Float of true volume. If this is not known use "None".

    Returns:
    - volume: The estimated volume of the object.
    """

    verticies = data.points

    # Convert Open3D point cloud to numpy array
    point_cloud_np = np.asarray(verticies)

    hull = ConvexHull(point_cloud_np)
    volume = hull.volume * 1000 # In units Liters
    
    if print_flag==True:
        print(f"Estimated volume: {volume:.5f} Liters")
    
    if true_volume != None:
        error = abs( (true_volume-volume) / true_volume  )
        if print_flag==True:
            print(f"Percentage error: {round(error*100,2)} %")
    
    return volume


def close_bottom(mesh, ground_level,ground_threshold = 1e-2):
    """
    Close the bottom of the mesh by adding a flat plane at the ground level.

    Parameters:
    - mesh: The input mesh (Open3D TriangleMesh object).
    - ground_level: The Y-coordinate representing the flat ground.
    - ground_threshold: Float to estimate the bottom level of the mesh. 

    Returns:
    - closed_mesh: The mesh with the bottom closed.
    """
    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Find vertices close to the ground level
    bottom_vertices = vertices[vertices[:, 1] <= ground_level + ground_threshold]  # Assuming Y is the up/down axis
    
    if len(bottom_vertices) > 0:
        # Use only the X and Z coordinates of the bottom vertices to form the flat plane
        bottom_2d = bottom_vertices[:, [0, 2]]  # Project the bottom vertices to the XZ plane
        
        # Compute the convex hull of the 2D projection of the bottom points
        hull = ConvexHull(bottom_2d)
        
        # Get the hull vertex indices from the original bottom_vertices array
        hull_indices = hull.vertices
        
        # Create a flat 3D plane using the hull's vertices (indices, not coordinates)
        new_triangles = [[hull_indices[0], hull_indices[i], hull_indices[i+1]] 
                         for i in range(1, len(hull_indices)-1)]
        
        # Convert hull vertices and triangles to Open3D format
        hull_mesh = o3d.geometry.TriangleMesh()
        hull_mesh.vertices = o3d.utility.Vector3dVector(bottom_vertices)
        hull_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        hull_mesh.compute_vertex_normals()
        
        # Merge the flat plane with the original mesh
        closed_mesh = mesh + hull_mesh
        closed_mesh.compute_vertex_normals()
        
        return closed_mesh
    else:
        print("No bottom vertices found at the specified ground level.")
        return mesh
    


def close_bottom_adv(mesh, ground_level, ground_threshold=1e-2, edge_threshold=1):
    """
    More advanced function for closing close the bottom of the mesh by 
    adding a flat plane at the ground level using a custom concave hull method.

    Parameters:
    - mesh: The input mesh (Open3D TriangleMesh object).
    - ground_level: The Y-coordinate representing the flat ground.
    - ground_threshold: Float to estimate the bottom level of the mesh.
    - edge_threshold: Threshold for filtering long edges (to form a concave hull).

    Returns:
    - closed_mesh: The mesh with the bottom closed.
    """
    # Get the vertices of the mesh
    vertices = np.asarray(mesh.vertices)
    
    # Find vertices close to the ground level
    bottom_vertices = vertices[vertices[:, 1] <= ground_level + ground_threshold]  # Assuming Y is the up/down axis
    

    if len(bottom_vertices) > 0:

        # Project all the bottom vertices to the ground level
        bottom_vertices[:, 1] = ground_level  # Set the Y-coordinate to ground_level

        # Use only the X and Z coordinates of the bottom vertices to form the flat plane
        bottom_2d = bottom_vertices[:, [0, 2]]  # Project the bottom vertices to the XZ plane
        
        # Compute the Delaunay triangulation of the 2D projection
        delaunay = Delaunay(bottom_2d)
        
        # List of edges to form the concave hull
        edges = set()
        
        # Add edges from Delaunay triangulation
        for simplex in delaunay.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges.add(edge)
        
        # Filter edges based on their length (keep only shorter ones)
        filtered_edges = []
        for edge in edges:
            pt1, pt2 = bottom_2d[edge[0]], bottom_2d[edge[1]]
            dist = np.linalg.norm(pt1 - pt2)
            if dist <= edge_threshold:
                filtered_edges.append((edge[0], edge[1]))

        # Create a set of vertices that are part of the concave boundary
        boundary_vertices = set()
        for edge in filtered_edges:
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])

        # Create the triangles for the concave hull
        boundary_points = bottom_vertices[list(boundary_vertices)]
        boundary_indices = list(boundary_vertices)
        
        # Create triangles for the concave hull (essentially connecting the boundary points)
        new_triangles = []
        for i in range(1, len(boundary_indices) - 1):
            new_triangles.append([boundary_indices[0], boundary_indices[i], boundary_indices[i + 1]])
        


        # Convert hull vertices and triangles to Open3D format
        hull_mesh = o3d.geometry.TriangleMesh()
        hull_mesh.vertices = o3d.utility.Vector3dVector(bottom_vertices)
        hull_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        hull_mesh.compute_vertex_normals()

        # Merge the flat plane with the original mesh
        closed_mesh = mesh + hull_mesh
        closed_mesh.compute_vertex_normals()

        return closed_mesh
    else:
        print("No bottom vertices found at the specified ground level.")
        return mesh
    
        


    

def compute_signed_volume(mesh):
    """
    Compute the signed volume of a triangular mesh using the divergence theorem.

    Parameters:
    - mesh: Open3D TriangleMesh object.

    Returns:
    - volume: The signed volume of the mesh.
    """
    volume = 0.0
    for triangle in mesh.triangles:
        # Get the vertices of the triangle
        v0 = np.asarray(mesh.vertices[triangle[0]])
        v1 = np.asarray(mesh.vertices[triangle[1]])
        v2 = np.asarray(mesh.vertices[triangle[2]])
        
        # Calculate the signed volume of the tetrahedron formed by the triangle and the origin
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0

    return volume

def compute_poisson_volume(point_cloud, true_volume=None,depth=12,low_density_threshold =0.2, cut_mesh= False, close_mesh = False,close_mesh_adv=False, ground_threshold = 1e-2,ground_level=None,overestimation=1.0,edge_threshold=1,print_flag=False):
    """
    Compute the volume of a point cloud using Poisson surface reconstruction.

    Parameters:
    - point_cloud: Open3D point cloud object.
    - true_volume: Float of true volume. If this is not known, use "None".
    - depth: Int of the poisson depth. Higher is more accurate but also more computationally expensive
    - low_density_threshold: Fraction threshold of low density regions to remove from the mesh. 
    - close_mesh: Bool of wether to close the bottom of the mesh.
    - ground_threshold: float of the threshold used to estimate the location of the ground level for closing the mesh.

    Returns:
    - volume: The estimated volume of the object.
    - mesh: The reconstructed Poisson surface. 
    """

    if point_cloud is None:
       # print("No point cloud found. Returning Artificial error of 1000 and None type mesh")
        return 1000, None
    
    # Estimate normals if not already present
    #point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #point_cloud.orient_normals_consistent_tangent_plane(10)
    point_cloud.estimate_normals()
    
    # to obtain a consistent normal orientation
    point_cloud.orient_normals_towards_camera_location(point_cloud.get_center())

    # Flip the normals to make them point outward (to get correct sign on volume)
    point_cloud.normals = o3d.utility.Vector3dVector( - np.asarray(point_cloud.normals))
    
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    
    # Crop out low-density vertices (optional but often useful for noisy data)
    vertices_to_remove = densities < np.quantile(densities, low_density_threshold)  # Remove lowest % of vertices
    mesh.remove_vertices_by_mask(vertices_to_remove)
        
    if cut_mesh == True:
        # Define the ground level (use the lowest Y-coordinate or specify it)
        if ground_level=='auto':
            min_y_value = np.min(np.asarray(point_cloud.points)[:, 1])  

        else:
            min_y_value = ground_level

        # Remove vertices from the mesh that are below this Y-value
        vertices = np.asarray(mesh.vertices)
        vertices_to_remove = vertices[:, 1] < min_y_value  # Vertices with Y lower than min_y_value
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    

    # Define the ground level (use the lowest Y-coordinate or specify it)
    if ground_level=='auto':
        ground_level = np.min(np.asarray(mesh.vertices)[:, 1])    
    #ground_level = 0.0
    
    
    if close_mesh == True:
        # Close the bottom of the mesh with a simple approach
        mesh = close_bottom(mesh, ground_level,ground_threshold)

    if close_mesh_adv == True:
        # Close the bottom of the mesh with a more sophisticated approach. 
        mesh = close_bottom_adv(mesh, ground_level,ground_threshold,edge_threshold)

    # Compute the volume of the mesh using the signed volume function
    volume = compute_signed_volume(mesh) * 1000  # Convert to liters (assuming units are in meters)
    volume = volume * overestimation # scale the volume with the expected overestimation factor
    if print_flag == True:
        print(f'True Volume       {true_volume}')
        print(f"Estimated volume: {volume:.5f} Liters")
    
        if true_volume is not None:
            error = abs((true_volume - volume) / true_volume)
            if print_flag==True:
                print(f"Absolute error:   {round(abs((true_volume - volume)), 2)} Liters")                
                print(f"Percentage error: {round(error * 100, 2)} %")
        
        
        
    return volume, mesh
