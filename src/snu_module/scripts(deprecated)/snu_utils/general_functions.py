"""
SNU Integrated Module v2.0
  - General Utility Functions
"""
# Future Import
from __future__ import print_function

# Import Libraries
import numpy as np
import yaml
from itertools import groupby


# Define epsilon
def epsilon():
    return 1e-12


# L2 Distance
def l2_distance_dim2(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# L2 Distance (dimension 3)
def l2_distance_dim3(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


# Get Maximum Tracklet ID among all Tracklets
def get_maximum_id(trks, max_trk_id):
    if len(trks) != 0:
        max_id = -1
        for trk in trks:
            if trk.id >= max_id:
                max_id = trk.id
        if max_id < max_trk_id:
            max_id = max_trk_id
    else:
        max_id = max_trk_id

    return max_id


# Get Maximum Consecutive Values in a List
def get_max_consecutive(src_list, find_val):
    max_count = 0
    for val, sub_list in groupby(src_list):
        sub_list = list(sub_list)

        if val is find_val:
            if len(sub_list) >= max_count:
                max_count = len(sub_list)

    return max_count


# Select Specific Indices from the List
def select_from_list(src_list, sel_indices):
    if len(sel_indices) == 0:
        return []
    else:
        selected_list = list(src_list[i] for i in sel_indices)
        return selected_list


# Exclude Specific Indices from the List
def exclude_from_list(src_list, excl_indices):
    if len(excl_indices) == 0:
        return src_list
    else:
        survived_indices = []
        for src_idx, _ in enumerate(src_list):
            if src_idx not in excl_indices:
                survived_indices.append(src_idx)
        return select_from_list(src_list, survived_indices)


# Search Common Points
def search_common_points(points1, points2):
    common_x, cind_x_1, cind_x_2 = np.intersect1d(points1[:, 0], points2[:, 0], return_indices=True)
    common_y, cind_y_1, cind_y_2 = np.intersect1d(points1[:, 1], points2[:, 1], return_indices=True)

    cind_1 = np.intersect1d(cind_x_1, cind_y_1)
    cind_2 = np.intersect1d(cind_x_2, cind_y_2)

    if len(cind_1) == 0 or len(cind_2) == 0:
        return [], [], []
    else:
        if not np.array_equal(points1[cind_1, :], points2[cind_2, :]):
            assert 0
        else:
            cpoints = points1[cind_1, :]

        return cpoints, cind_1, cind_2


# Search Uncommon Points
def search_uncommon_points(points1, points2):
    _, cind_1, cind_2 = search_common_points(points1, points2)

    excl_points1 = []
    pidx_1 = []
    for pidx, point1 in enumerate(points1):
        if pidx not in cind_1:
            pidx_1.append(pidx)
            excl_points1.append(point1)

    excl_points2 = []
    pidx_2 = []
    for pidx, point2 in enumerate(points2):
        if pidx not in cind_2:
            pidx_2.append(pidx)
            excl_points2.append(point2)

    return np.array(excl_points1).reshape(-1, 2), np.array(excl_points2).reshape(-1, 2), pidx_1, pidx_2


# Generate 2D Gaussian Window
def generate_gaussian_window(row_size, col_size, mean, stdev, min_val=0, max_val=1):
    # Construct 2-D Gaussian Window
    x, y = np.meshgrid(np.linspace(-1, 1, col_size), np.linspace(-1, 1, row_size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mean) ** 2 / (2.0 * stdev ** 2)))

    # cv2.imshow("Unnormalized Gaussian Map", g)

    # Normalize So that the matrix is scaled between "eps" and "1"
    gvec = g.flatten()
    gauss_map = min_val + ((g - np.min(gvec)) * (max_val - min_val) / (np.max(gvec) - np.min(gvec)))

    return gauss_map, min_val, max_val


# Get the number of pixels greater/equal/less than a threshold value
def get_pixel_numbers(patch, pixel_thresh, operator):
    if operator not in ['g', 'ge', 'e', 'le', 'l']:
        assert 0, "operator cannot be defined!"

    # Vectorize Patch to Boost Computation Speed
    patch_vector = patch.reshape(1, -1)
    if operator == 'g':
        return (patch_vector > pixel_thresh).sum()
    elif operator == 'ge':
        return (patch_vector >= pixel_thresh).sum()
    elif operator == 'e':
        return (patch_vector == pixel_thresh).sum()
    elif operator == 'le':
        return (patch_vector <= pixel_thresh).sum()
    else:
        return (patch_vector < pixel_thresh).sum()


# Collect Same Attribute Values from class attributes across multiple classes (CLASS DICTIONARY VERSION)
def gather_values(class_dict, attribute):
    collected_value_dict = {}
    for class_dict_key, value_class in class_dict.items():
        if hasattr(value_class, attribute):
            collected_value_dict[class_dict_key] = getattr(value_class, attribute)

    return collected_value_dict


# Function for Reading Static Camera Parameters
def read_static_cam_param(yaml_file_path, agent_name):
    with open(yaml_file_path, "r") as stream:
        tmp = yaml.safe_load(stream)
    cam_param = np.asarray(tmp[agent_name]["camera_param"])

    return cam_param


# Get Current Computer Screen Geometry
def get_screen_geometry():
    import Tkinter as tk
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry_str = str(root.winfo_geometry())
    root.destroy()
    geometry_dict = {
        "curr_monitor": {
            "width_pixels": int(geometry_str.split("x")[0]),
            "height_pixels": int(geometry_str.split("x")[1].split("+")[0]),
        },
        "margin_length": {
            "left_pixels": int(geometry_str.split("x")[1].split("+")[1]),
            "top_pixels": int(geometry_str.split("x")[1].split("+")[2]),
        },
    }
    return geometry_dict


# Colormap
def colormap(colormap_bin):
    # Calculate Spectrum Ratio
    spectrum_ratio = 255*6/colormap_bin

    # Initialize Starting Color
    red, green, blue = 255, 0, 0

    # Initialize Colorbar List
    colorbar = []

    # Get Colorbar Step
    cstep = int(round(spectrum_ratio))

    # Loop for colorbar steps
    for color_idx in range(0, 255*6+1, cstep):
        if 1 < color_idx <= 255:
            blue += cstep
        elif 255 < color_idx <= 255*2:
            red -= cstep
        elif 255*2 < color_idx <= 255*3:
            green += cstep
        elif 255*3 < color_idx <= 255*4:
            blue -= cstep
        elif 255*4 < color_idx <= 255*5:
            red += cstep
        elif 255*5 < color_idx <= 255*6:
            green -= cstep

        # Append to Colorbar
        colorbar.append((red/255.0, green/255.0, blue/255.0))

    return colorbar


# Make Difference Matrix for a (list/array), output is a numpy matrix
def array_difference_matrix(array):
    """
    for input array: [a11, a12, a13]
    make output difference matrix as follows
    -------------
    output matrix: [(a11-a11), (a11-a12), (a11-a13),
                    (a12-a11), (a12-a12), (a12-a13),
                    (a13-a11), (a13-a12), (a13-a13)]
    """
    if type(array) == list:
        array = np.asarray(array)
    assert (array.ndim == 1), "Input Array must be one-dimensional"

    # Get Array Length and Convert array into column vector
    N = array.shape[0]
    array_column = array.reshape(N, 1)

    # Construct Difference Matrix
    return np.tile(array_column, N) - np.tile(array_column.T, (N, 1))
