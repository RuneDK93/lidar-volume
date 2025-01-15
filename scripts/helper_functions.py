import numpy as np
import open3d as o3d
from collections import defaultdict
from scripts.preproc_functions import preproc, cluster_points
from scripts.volume_functions import compute_poisson_volume
import itertools
from itertools import product
import json


def material_settings(material):
    """
    Set true volume and optimized parameters for a given material.

    Parameters:
    material (str): The material type. Valid options are 'træ', 'træ_4x', 'iso', 'mur', 'es', 'gips'.

    Returns:
    tuple: A tuple containing:
        - overestimation (float): The mean fraction by which the model overestimates the volume.
        - ldt (float): Low Density Threshold, determines the percentage of low-density points to remove 
          from the Poisson reconstructed mesh.
        - true_volume (float): The known or estimated true volume for the material in liters.
    """
    # Define true volume for each material
    true_volume_map = {
        "træ": 5.15508,
        "træ_4x": 5.15508 * 4,
        "iso": 93.0555,
        "mur": 7.2,
        "es": 1,  # Estimated based on water volume
        "gips": 28.08
    }

    # Define low density threshold for each material
    ldt = 0.05 if material == "iso" else 0.1

    # Define overestimation factor for each material
    overestimation_map = {
        "træ": 0.9,
        "iso": 0.9,
        "mur": 0.88,
        "gips": 0.65,
        "es": 0.45
    }

    # Optionally add other hyperparameter values to the function.
    # If you add new hyperparameters, remember to add them to the output of this function

    # Define new hyperparameter for each material
    #hyperparam_map = {
    #    "træ":  1,
    #    "iso":  1,
    #    "mur":  1,
    #    "gips": 1,
    #    "es":   1
    #}    

    # Fetch values from the maps
    true_volume = true_volume_map.get(material, None)
    overestimation = overestimation_map.get(material, None)

    if true_volume is None or overestimation is None:
        raise ValueError(f"Unknown material type: {material}")

    return overestimation, ldt, true_volume


def load_hyperparam_search_results(result_file,print_results=False):


    """
    Load and analyze hyperparameter search results from a JSON file.

    This function organizes the errors from a hyperparameter search by material and parameter combinations,
    computes the mean errors for each configuration, and identifies the best hyperparameters for each material
    as well as overall based on the combined mean error.

    Args:
        result_file (str): Path to the JSON file containing search results. Each entry in the JSON should 
                           include 'material', 'params', and 'error'.
        print_results (bool): If True, prints a summary of the best hyperparameters and mean errors.

    Returns:
        params: A dictionary where keys are materials and values are the best hyperparameter configurations
              (as dictionaries) for that material.
    """

    # Load overall results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Organize results by hyperparameters and material
    error_data = defaultdict(lambda: defaultdict(list))

    for entry in results:
        material = entry['material']
        params = tuple(sorted(entry['params'].items()))  # Create a key from sorted params
        error_data[material][params].append(entry['error'])

    # Calculate mean errors for each unique combination of hyperparameters
    mean_errors = {material: {} for material in error_data.keys()}

    for material, params_dict in error_data.items():
        for params, errors in params_dict.items():
            mean_errors[material][params] = np.mean(errors)

    # Find best hyperparameters for each material
    best_params = {}
    for material, param_means in mean_errors.items():
        best_params[material] = min(param_means, key=param_means.get)

    # Calculate combined mean errors across both materials
    combined_mean_errors = defaultdict(list)

    for material, param_means in mean_errors.items():
        for params, mean_error in param_means.items():
            combined_mean_errors[params].append(mean_error)

    # Find the overall best hyperparameters based on combined mean error
    overall_best_params = min(combined_mean_errors, key=lambda x: np.mean(combined_mean_errors[x]))
    overall_best_mean_error = np.mean(combined_mean_errors[overall_best_params])

    # Display results
    if print_results == True:
        print("Mean Errors for Each Material:")
    for material, best_param in best_params.items():

        if material == "træ":
            true_volume = 5.15508 # liters
        if material == "træ_4x":
            true_volume = 5.15508*4 # liters
        if material == 'iso':
            true_volume = 93.0555
        if material == "mur":
            true_volume = 7.2
        if material =="es":
            true_volume = 1 # true volume here is hard to estimate, but probably about 1L from water volume. 
        if material =="gips":
            true_volume = 28.08    


        if print_results == True:
            print(f"{material}: Best Params = {dict(best_param)}, Mean Error = {mean_errors[material][best_param]:.4f}, True Volume = {true_volume}")
    if print_results == True:
        # Overall Best Parameters
        print("\nOverall Best Combination:")
        print(f"Best Params = {dict(overall_best_params)}, Mean Error = {overall_best_mean_error:.4f}")


    params = dict(best_params)

    return params


def sample_results(material, pred_materials, errors_materials, ID, round_number=1):
    """
    Print model results for a sample of materials.

    Parameters:
    material (str): Material name.
    pred_materials (dict): Dictionary of predicted volumes for each material sample.
    errors_materials (dict): Dictionary of error values for each material sample.
    ID: Identifier for the sample to analyze.
    round_number (int): Decimal places to round results to (default is 1).

    Returns:
    None: Prints the results directly.
    """
    def compute_stats(values, true_volume, round_number):
        """Helper function to compute mean, std, and percentage error."""
        mean_value = np.mean(values)
        std_error = np.std(values) / np.sqrt(len(values))
        percent_error_mean = 100 * np.abs(mean_value) / true_volume
        percent_error_std = 100 * std_error / true_volume
        return (
            np.round(mean_value, round_number),
            np.round(std_error, round_number),
            np.round(percent_error_mean, round_number),
            np.round(percent_error_std, round_number),
        )

    # Get true volume
    _, _, true_volume = material_settings(material)
    print(f"Material: {material}\nTrue Volume: {true_volume:.2f} liters")

    # Predictions
    predictions = pred_materials[ID]
    print(f"\nPredictions: {predictions}")
    pred_mean, pred_std, _, _ = compute_stats(predictions, true_volume, round_number)
    print(f"Mean Prediction: {pred_mean}\nStd Error (Prediction): {pred_std}")

    # Absolute Errors
    abs_errors = np.abs(errors_materials[ID])
    abs_mean, abs_std, abs_percent_mean, abs_percent_std = compute_stats(
        abs_errors, true_volume, round_number
    )
    print("\nAbsolute Errors:")
    print(f"Mean: {abs_mean}\nStd Error: {abs_std}")
    print(f"Mean Percent Error: {abs_percent_mean}%\nStd Percent Error: {abs_percent_std}%")

    # Raw Errors
    raw_errors = errors_materials[ID]
    raw_mean, raw_std, raw_percent_mean, raw_percent_std = compute_stats(
        raw_errors, true_volume, round_number
    )
    print("\nMean Errors:")
    print(f"Mean: {raw_mean}\nStd Error: {raw_std}")
    print(f"Mean Percent Error: {raw_percent_mean}%\nStd Percent Error: {raw_percent_std}%")




def samples_run(samples,materials):

    """
    Run volume estimation for multiple samples and materials.

    This function processes 3D models of various materials and samples to compute their estimated volumes 
    and calculate the errors relative to true volumes. It uses preprocessing, clustering, and volume 
    computation steps for each material and sample combination.

    Args:
        samples (list of str): A list of sample names (subdirectory names) to process for each material.
        materials (list of str): A list of material types to analyze, corresponding to directories containing the sample data.

    Returns:
        tuple:
            pred_materials (list of list): Predicted volumes for each sample, organized by material.
            errors_materials (list of list): Volume errors for each sample, organized by material.

    Notes:
        - Each sample is assumed to be stored in the directory structure `Data/{material}/{sample}/textured_output.obj`.
        - Preprocessing, clustering, and volume computation parameters are material-specific, fetched via the `material_settings` function.
        - Errors are calculated as the difference between the predicted and true volumes for each sample.
        - Results are grouped into lists corresponding to each material.
    """


    errors_materials = []
    pred_materials = []
    for material in materials:
        print('Running model on material',material,'(...)')
        # Fetch the parameters for a given material
        overestimation, ldt, _ = material_settings(material)    

        errors_samples = []   
        pred_samples = [] 
        for sample in samples:

            # Træ does not have sample8, so skip that if called
            if material == "træ" and sample == "sample8":
                continue

            data = o3d.io.read_triangle_mesh(f"Data/{material}/{sample}/textured_output.obj")
            object_points     = preproc(data,
                                       segment     = True,
                                       threshold   = 0.02,
                                       crop        = False,
                                       center      = True,
                                       n_points    = 60000)

            # Two-step Cluster points
            object_cluster = cluster_points(object_points,  choice=4, eps=0.1, min_points=10)
            object_cluster = cluster_points(object_cluster, choice=3, eps=0.03, min_points=10)

            # Fetch the parameters for a given material
            overestimation, ldt, true_volume = material_settings(material)

            # Compute volume
            volume, mesh = compute_poisson_volume(
                object_cluster,
                true_volume           = true_volume,
                depth                 = 10,
                low_density_threshold = ldt,
                cut_mesh              = True,
                close_mesh            = False,
                close_mesh_adv        = True,
                ground_level          = 'auto',
                ground_threshold      = 1,
                edge_threshold        = 1,
                overestimation        = overestimation, 
                print_flag            = False)

            # Save results
            errors_samples.append(volume-true_volume)
            pred_samples.append(volume)
        errors_materials.append(errors_samples)
        pred_materials.append(pred_samples)   

    print('Done!')

    return pred_materials, errors_materials