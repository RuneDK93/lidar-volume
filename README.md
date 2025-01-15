# Volume Computation from LiDAR Scans

This repository provides a Python-based solution for computing the volume of objects from LiDAR scans. The workflow consists of two main steps: object identification within the scan and volume estimation using a 3D surface reconstruction technique.

## Features
- **Two-step DBSCAN Clustering:** Identify and isolate the target object from the scanned LiDAR scene.
- **Poisson Surface Reconstruction:** Reconstruct the object's surface and compute the enclosed volume using signed volume calculations.
- Supports multiple material types, each with customizable overestimation parameters to account for air pockets.

---

## Table of Contents
- [Volume Computation from LiDAR Scans](#volume-computation-from-lidar-scans)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Methodology](#methodology)
    - [1. Object Identification](#1-object-identification)
    - [2. Volume Estimation](#2-volume-estimation)
  - [Material Overestimation Factors](#material-overestimation-factors)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Command-Line Usage:](#command-line-usage)
    - [Arguments:](#arguments)
  - [Version Information](#version-information)
  - [License](#license)

---

## Methodology

### 1. Object Identification
Relevant objects in the LiDAR scan are identified using **two-step DBSCAN clustering**, which isolates clusters of points representing the target object.

### 2. Volume Estimation
The volume of the identified object is estimated through:
- **Poisson Reconstruction:** Generates a continuous surface from the clustered points.
- **Signed Volume Computation:** Calculates the volume enclosed by the reconstructed surface.

---

## Material Overestimation Factors
Different materials may contain varying fractions of air pockets within the reconstructed surface. To account for this, the method applies a material-specific **overestimation fraction** scalar derived from training data:

| Material        | Overestimation Fraction |
|-----------------|--------------------------|
| Isolation       | 0.90                     |
| Bricks          | 0.88                     |
| Plaster         | 0.65                     |
| Post-Sorting    | 0.45                     |
| Wood            | 0.90                     |

---

## Requirements
- Python 3.8 or higher
- Libraries:
  - NumPy
  - SciPy
  - Open3D

---

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-url

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt

---

## Usage

You can run the model using the `run_model.py` script. The script takes the LiDAR scan file and material type as inputs and computes the volume of the relevant object in the scan.

### Command-Line Usage:

1. **If the true volume is NOT known:**
   ```bash
   python run_model.py "Data/træ/sample7/textured_output.obj" "træ"


2. **If the true volume is known (i.e. the training data for the model with known volumes):**
   ```bash
   python run_model.py "Data/træ/sample7/textured_output.obj" "træ" --true_volume_known


### Arguments:
- `file_location`: The path to the LiDAR scan data file (e.g., `.obj`).
- `material`: The material type for volume computation. Options include:
  - `iso` (Isolation material)
  - `mur` (Bricks)
  - `træ` (Wood)
  - `es`  (Post-Sorting)
  - `gips` (Plaster)
- `--true_volume_known`: An optional flag to indicate whether the true volume is known for the given file. If provided, the true volume from the training data is used in the computation; otherwise, the script assumes the true volume is unknown and sets it to `None`.

---

## Version Information
This project uses the following library versions for compatibility:

| Library  | Version  |
|----------|----------|
| Python   | 3.8.2    |
| NumPy    | 1.20.1   |
| SciPy    | 1.6.2    |
| Open3D   | 0.10.0.0 |

Make sure to match these versions to avoid unexpected behavior.

---

## License
This project is licensed under the MIT License.
