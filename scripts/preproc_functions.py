import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull



def preproc(data,segment=True,threshold=0.03,crop=False,crop_type="sphere",crop_length=2,center=True,n_points=30000,print_flag=False):
    
    '''
    Function to preprocess the raw lidar or photogrammetry data. 
    The preprocessing involves converting the data to pointcloud format, if not already in this format. 
    
    The preprocessing optionally involes cropping the scene to a cube around the center of the scene. 
    The preprocessing optionally involves segmenting the scene into ground plane and objects of interest.
    
    
    Input Parameters
    ----------
    data : '.obj' or '.ply' 
        3D scene in either mesh or pointcloud format. Function supports .obj and.ply file formats. 
        
    segment: 'bool'
        Flag to determine if the function should segment the 3D scene into ground plane and objects of interest
        
    threshold: 'float'
        Value to determine the ground segmentation sensitivity.
        
    crop: 'bool'
        Flag to determine if function should croup the 3D scene around the center of the scene
        
    crop_type: 'string'
        The type of cropping. Either "cube" or "sphere".
                
        
    crop_length 'float'
       Value to set the cropping dimension in meters. 
       This is either the cube side length or the sphere radius depending on chosen option in crop_type.



    Returns
    ---------
    object_points : 'pointcloud'
        The preprocessed object points
        
    '''
    
    
        
    
    
    # Check data type
    if isinstance(data, o3d.geometry.TriangleMesh):
        data_type = 'mesh'
    elif isinstance(data, o3d.geometry.PointCloud):
        data_type = 'pointcloud'
    
    if print_flag==True:
        print('Data is of type',data_type)
    
    if data_type == 'mesh':    
        data = data.sample_points_uniformly(number_of_points=n_points)  # You can adjust the number of points as needed

        
    if center == True:
        # Center the coordinate frame in the scene
        # Compute the mean of all points
        points = np.asarray(data.points)
        mean_location = np.mean(points, axis=0)

        # Translate the point cloud to center it at the origin [0, 0, 0]
        data.translate(-mean_location)        
        
                                        

    if crop == True: 
        crop_tightness = crop_length/2       
        if crop_type == "cube":
            # Define a region of interest (ROI) by creating a bounding box
            min_bound = [-crop_tightness, -crop_tightness, -crop_tightness]  
            max_bound = [crop_tightness, crop_tightness, crop_tightness]     
    
            # Crop the point cloud to the region of interest
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            data = data.crop(bbox)   
        
        if crop_type == "sphere":         
            # Define the center of the circle (e.g., the origin or the center of the point cloud)
            sphere_center = np.array([0, 0, 0])  # You can adjust this to your desired center point
    
            # Define the radius for the circular crop
            radius = crop_tightness  # Set the crop radius
    
            # Get the point cloud data as a numpy array
            points = np.asarray(data.points)
    
            # Compute the distance of each point from the center
            distances = np.linalg.norm(points - sphere_center, axis=1)
    
            # Create a mask for points within the specified radius
            mask = distances <= radius
    
            # Apply the mask to retain only the points within the circular region
            cropped_points = points[mask]
    
            # Update the point cloud with the cropped points
            data.points = o3d.utility.Vector3dVector(cropped_points)
                   
        
        
    
    if segment==True:
        # Perform plane segmentation to find the ground plane
        plane_model, inliers = data.segment_plane(distance_threshold=threshold, ransac_n=5, num_iterations=1000)
        ground_plane = data.select_by_index(inliers)
        data = data.select_by_index(inliers, invert=True)
        
    data.paint_uniform_color([1, 0, 0])  # RGB values between 0 and 1
    #data.paint_uniform_color([0.5, 0.5, 0.5])  # Medium grey
                
    object_points = data
    
    
    
    return object_points
   
    
    
def cluster_points(object_points, choice=3, eps=0.1, min_points=10,choice4_points=500,print_flag=False):
    '''
    Function to identify the object of interest.
    The function works by clustering the input points and identifying the cluster
    that contains the object of interest.

    Input Parameters
    ----------
    object_points : 'pointcloud'
        The preprocessed object points

    choice : 'integer' in [1,2,3,4]
        The chosen method for identifying the object of interest.
        1: Picking cluster based on largest mean y-value (height dimension)
        2: Picking cluster based on largest single y-value point.
        3: Picking cluster based on most number of points.
        4: Picking cluster closest to origin [0,0,0] with at least choice4_points points.

    eps : 'float'
        Parameter for the dbscan clustering

    min_points : 'int'
        Parameter for the dbscan clustering        

    Returns
    ---------
    object_cluster : 'pointcloud'
        The identified cluster of main object points
    '''    

    # Perform DBSCAN clustering on the remaining points
    labels = np.array(object_points.cluster_dbscan(eps=eps, min_points=min_points))

    # If only one cluster then set labels to zero
    if len(np.unique(labels)) == 1:
        labels = (labels * 0)

    # Cluster identification based on user choice
    if choice == 1:
        # Method 1: Cluster with the highest mean Y value
        max_label = labels.max()
        mean_y_values = []
    
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = np.asarray(object_points.points)[cluster_indices]
            mean_y = cluster_points[:, 1].mean()  # Calculate mean Y value
            mean_y_values.append((i, mean_y))
    
        # Find the cluster with the highest mean Y value
        target_cluster_label = max(mean_y_values, key=lambda x: x[1])[0]
    
    elif choice == 2:
        # Method 2: Cluster with the highest Y value point
        max_label = labels.max()
        highest_y_values = []
    
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = np.asarray(object_points.points)[cluster_indices]
            highest_y = cluster_points[:, 1].max()  # Find the highest Y value in each cluster
            highest_y_values.append((i, highest_y))
    
        # Find the cluster with the highest Y value point
        target_cluster_label = max(highest_y_values, key=lambda x: x[1])[0]
    
    elif choice == 3:
        # Method 3: Cluster with the most points
        max_label = labels.max()
        cluster_sizes = [(i, np.sum(labels == i)) for i in range(max_label + 1)]
    
        # Find the cluster with the most points
        target_cluster_label = max(cluster_sizes, key=lambda x: x[1])[0]
    
    elif choice == 4:
        # Method 4: Cluster closest to origin with at least 100 points
        max_label = labels.max()
        origin_distances = []
    
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = np.asarray(object_points.points)[cluster_indices]
            
            # Check if cluster has at least choice4_points points (to make sure it is the main object)
            if len(cluster_points) >= choice4_points:
                # Compute distance to origin
                distance_to_origin = np.linalg.norm(cluster_points.mean(axis=0))
                origin_distances.append((i, distance_to_origin))
    
        # Find the cluster closest to origin among those with >= 100 points
        if origin_distances:
            target_cluster_label = min(origin_distances, key=lambda x: x[1])[0]
        else:
            print("No cluster with at least 100 points found. Returning None")
            return None

    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        exit()
    

    # Extract the target cluster
    object_cluster = object_points.select_by_index(np.where(labels == target_cluster_label)[0])
    
    if print_flag == True:
        print('N clusters:', len(np.unique(labels)))
        print('Points in identified object cluster', sum(labels == target_cluster_label))       

    return object_cluster

    
    
