import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from PIL import Image
import cv2

####### GLOBAL PARAMS FOR TUNING #######
GLOBAL_KERNEL_SIZE = 5
GLOBAL_SIGMA = 1

GLOBAL_HARRIS_RESPONSE_THRESHOLD = 0.08

GLOBAL_ALPHA = 0.04

GLOBAL_MATCH_RATIO_THRESHOLD = 0.75

GLOBAL_RADIUS = 50
GLOBAL_TOP_PERCENTILE = 10
########################################

# Gauss kernel generation code from HW1
def gauss_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))
    g /= g.sum()  # Normalize
    return g

def plot_interest_points(image, x, y):
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='r', s=10)
    plt.show()

def get_interest_points(image, feature_width):

    # STEP 1: Calculate the gradient (partial derivatives on two directions).

    # Blur first
    gauss_filter = gauss_kernel(GLOBAL_KERNEL_SIZE, GLOBAL_SIGMA)
    image = cv2.filter2D(image, -1, gauss_filter)

    x_gradient = filters.sobel_v(image)
    y_gradient = filters.sobel_h(image)

    # STEP 2: Apply Gaussian filter with appropriate sigma. Using filters for this just for consistency across dimensions

    gauss_x = filters.gaussian(x_gradient, sigma=1)
    gauss_y = filters.gaussian(y_gradient, sigma=1)
    gauss_xy = filters.gaussian(x_gradient * y_gradient, sigma=1)

    # STEP 3: Calculate Harris cornerness score for all pixels.

    det = gauss_x * gauss_y - gauss_xy ** 2
    trace_squared = (gauss_x + gauss_y) ** 2
    harris_response = det - GLOBAL_ALPHA * trace_squared

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    coordinates = feature.peak_local_max(harris_response, min_distance=5, threshold_rel=GLOBAL_HARRIS_RESPONSE_THRESHOLD)
    xs = coordinates[:, 1]
    ys = coordinates[:, 0]

    # Attempt at Adaptive Non-Max Suppression

    # coords = np.array([xs, ys]).T
    # sorted_coords = coords[np.argsort(harris_response[ys, xs])]
    # distances = np.sqrt(np.sum((sorted_coords[:, None] - sorted_coords[None]) ** 2, axis=-1))

    # final_coords = []

    # for i, point in enumerate(sorted_coords):
    #     current_distances = distances[i]
    #     within_radius_indices = np.where(current_distances < GLOBAL_RADIUS)[0]
    #     harris_within_radius = harris_response[sorted_coords[within_radius_indices, 1], sorted_coords[within_radius_indices, 0]]
    #     if len(harris_within_radius) > 0:
    #         threshold = np.percentile(harris_within_radius, 100 - GLOBAL_TOP_PERCENTILE)
    #         top_indices = within_radius_indices[harris_within_radius >= threshold]
    #         if i in top_indices:
    #             final_coords.append(point)

    # # Convert final_coords back to numpy array if needed and separate into xs and ys
    # final_coords = np.array(final_coords)
    # xs = final_coords[:, 0]
    # ys = final_coords[:, 1]

    return xs, ys

def get_features(image, x, y, feature_width):

    gauss_filter = gauss_kernel(GLOBAL_KERNEL_SIZE, GLOBAL_SIGMA)

    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.

    # Take in the image, convert to greyscale using PIL

    smoothed = cv2.filter2D(image, -1, gauss_filter)
    plt.imshow(smoothed, cmap='gray')
    x_gradient = filters.sobel_v(smoothed)
    y_gradient = filters.sobel_h(smoothed)

    # STEP 2: Decompose the gradient vectors to magnitude and direction.

    magnitude = np.sqrt(x_gradient ** 2 + y_gradient ** 2)
    direction = np.arctan2(y_gradient, x_gradient)

    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors.

    descriptors = []

    grid_width = 4
    hist_bin_width = 2*np.pi / 8

    for point_x, point_y in zip(x, y):
        hist = np.zeros(128)

        # For each feature, determine the patch we are considering to create the feature descriptor

        patch_x_start = int(point_x - feature_width / 2)
        patch_x_end = patch_x_start + feature_width

        patch_y_start = int(point_y - feature_width / 2)
        patch_y_end = patch_y_start + feature_width

        # Do 16 times for each cell in our 4x4 grid inside the patch

        start_index = 0

        for i in range(grid_width):
            for j in range(grid_width):

                cell_x_start = max(patch_x_start + i * (feature_width // grid_width), 0)
                cell_x_end = min(cell_x_start + (feature_width // grid_width), image.shape[1])

                cell_y_start = max(patch_y_start + j * (feature_width // grid_width), 0)
                cell_y_end = min(cell_y_start + (feature_width // grid_width), image.shape[0])

                tmp_hist = np.zeros(8)
                for x in range(cell_x_start, cell_x_end):
                    for y in range(cell_y_start, cell_y_end):
                        bin_index = int(direction[y, x] / hist_bin_width)
                        tmp_hist[bin_index] += magnitude[y, x]
                hist[start_index:start_index+8] = tmp_hist
                start_index += 8

        # Normalize the histogram to unit length then add it to the descriptors
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        else:
            hist = np.zeros_like(hist)

        descriptors.append(hist)

    features = np.array(descriptors, dtype=np.float32)
    return features

def match_features(im1_features, im2_features):
    # For extra credit you can implement spatial verification of matches.

    '''
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    matches_list = []
    confidences_list = []

    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
        # Sum the squared differences, then square root - Euclidean distance

    # Fix the dimensions of the two feature arrays to be able to calculate distances
    if im1_features.ndim == 1:
        im1_features = im1_features[np.newaxis, :]
    if im2_features.ndim == 1:
        im2_features = im2_features[np.newaxis, :]

    distances_squared = np.sum((im1_features[:, np.newaxis, :] - im2_features[np.newaxis, :, :]) ** 2, axis=2)
    distances = np.sqrt(distances_squared)

    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.

    for i in range(len(distances) - 1):
        sorted_indices = np.argsort(distances[i])
        closest_index = sorted_indices[0]
        second_closest_index = sorted_indices[1]

        ratio = distances[i][closest_index] / distances[i][second_closest_index]

        if ratio < GLOBAL_MATCH_RATIO_THRESHOLD:
            matches_list.append([i, closest_index])
            confidences_list.append(ratio)

    matches = np.array(matches_list)
    confidences = np.array(confidences_list)

    return matches, confidences