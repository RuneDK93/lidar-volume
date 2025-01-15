import open3d as o3d
import argparse
from scripts.preproc_functions import preproc, cluster_points
from scripts.volume_functions import compute_poisson_volume
from scripts.helper_functions import material_settings

def run_model(file_location, material, true_volume_known):
    # Load the data
    data = o3d.io.read_triangle_mesh(file_location)
    
    # Preprocess the data
    object_points = preproc(data,
                            segment     = True,
                            threshold   = 0.02,
                            crop        = False,
                            center      = True,
                            n_points    = 100000)

    # Two-step cluster the preprocessed points
    object_cluster = cluster_points(object_points,  choice=4, eps=0.1, min_points=10)
    object_cluster = cluster_points(object_cluster, choice=3, eps=0.03, min_points=10)

    # Fetch the optimal parameters for a given material
    overestimation, ldt, true_volume = material_settings(material)

    # If the true volume is not known, set it to None
    if not true_volume_known:
        true_volume = None

    # Compute volume
    volume, mesh = compute_poisson_volume(
        object_cluster,
        true_volume           = true_volume,
        depth                 = 10,
        low_density_threshold = ldt,
        cut_mesh              = True,
        close_mesh            = False,
        close_mesh_adv        = True,
        ground_threshold      = 1,
        ground_level          = 'auto',
        edge_threshold        = 1,
        overestimation        = overestimation, 
        print_flag            = True
    )
    
    return volume

# Main block for handling command-line input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the volume computation model on a LiDAR scan.")
    parser.add_argument("file_location", type=str, help="Path to the LiDAR scan data file (.obj).")
    parser.add_argument("material", type=str, choices=['iso', 'mur', 'træ', 'es','gips'], help="Material type for volume computation.")
    parser.add_argument("--true_volume_known", action="store_true", help="Indicate if the true volume is known for the given file.")

    args = parser.parse_args()

    # Call the function with the provided inputs
    volume = run_model(args.file_location, args.material, args.true_volume_known)

    # Output the computed volume
    if args.true_volume_known:
        print(f"Computed Volume for {args.material} material: {volume:.4f} liters")
    else:
        print(f"Computed Volume for {args.material} material (true volume not provided): {volume:.4f} liters")
