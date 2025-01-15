import open3d as o3d
from scripts.preproc_functions import preproc, cluster_points
from scripts.volume_functions import compute_poisson_volume
import itertools
from itertools import product
import numpy as np


# Define the different samples to optimize parameters on
samples     = ["sample1", 
               "sample2",
               "sample3",
               "sample4"]

materials   = ["mur"]

# Hyperparameters for grid search
param_grid = {
    "segment_threshold": [0.02], 

    "choice1":           [4],
    "eps1":              [0.1],
    "min_points1":       [10],
    "choice2":           [3],
    "eps2":              [0.03],
    "min_points2":       [10],

    "depth":             [10],
    "ldt":               [0.1], 
    "cut_mesh":          [True],
    "close_mesh":        [False],
    "close_mesh_adv":    [True],
    "gt":                [1],
    "edge_threshold":    [1],
    "ground_level":      ['auto'],    
    "overestimation":    [1,0.9,0.88,65,0.45]
}


# Generate all combinations of hyperparameters
param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]

# Store results
overall_results = []

# Total number of combinations
total_combinations = len(samples) * len(materials)

# Iterate over samples and material
for idx, (sample, material) in enumerate(product(samples, materials), 1):    
    print(f"Processing {idx}/{total_combinations}: sample={sample}, material={material}")
    # Set up the filename and load the mesh
    file = f"{sample}"    
    data = o3d.io.read_triangle_mesh(f"Data/{material}/{file}/textured_output.obj")

    # Define true volume based on material
    true_volume = {
        "træ": 5.15508,
        "træ_4x": 5.15508 * 4,
        "iso": 93.0555,
        "mur": 7.2,
        "es": 1,  
        "gips": 28.08
    }.get(material, 1)  # Default to 1 if material not found

    for params in param_combinations:
        # Unpack parameters
        segment_threshold = params['segment_threshold']
        #crop_length = params['crop_length']
        choice1 = params['choice1']
        eps1 = params['eps1']
        min_points1 = params['min_points1']
        choice2 = params['choice2']
        eps2 = params['eps2']
        min_points2 = params['min_points2']
        depth = params["depth"]
        ldt = params['ldt']
        cut_mesh = params["cut_mesh"]
        close_mesh = params["close_mesh"]
        ground_threshold = params['gt']
        overestimation = params['overestimation']
        close_mesh_adv = params['close_mesh_adv']
        edge_threshold = params['edge_threshold']
        ground_level   = params['ground_level']

        # Preprocess
        object_points = preproc(
            data,
            segment=True,
            threshold=segment_threshold,
            crop=False,
            crop_type="sphere",
            center=True,
            n_points=50000
        )


        # Cluster points
        object_cluster = cluster_points(object_points, choice=choice1, eps=eps1, min_points=min_points1)            
        if object_cluster is not None:
            if choice2:
                object_cluster = cluster_points(object_cluster, choice=choice2, eps=eps2, min_points=min_points2)

        # Compute volume
        volume, mesh = compute_poisson_volume(
            object_cluster,
            true_volume=true_volume,
            depth=depth,
            low_density_threshold=ldt,
            cut_mesh=cut_mesh,
            close_mesh=close_mesh,
            close_mesh_adv=close_mesh_adv,            
            ground_threshold=ground_threshold,
            overestimation=overestimation,
            edge_threshold=edge_threshold,
            ground_level=ground_level

        )
        # Calculate error
        error = abs(true_volume - volume)   
        results = {
            "sample": sample,
            "material": material,
            "params": params,
            "volume": volume,
            "error": error
        }

        overall_results.append(results)

import json



# Save overall results
overall_results_file = "results/hyperparam_results.json"
with open(overall_results_file, 'w') as f:
    json.dump(overall_results, f, indent=4)


print(f"Results saved in results directory.")