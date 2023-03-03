from math import radians
from random import randrange

from numpy import array, arange, concatenate, dot, where, unique, zeros, sin, cos
from numpy.random import rand
from cv2 import blur, imread, cvtColor, COLOR_BGR2GRAY
from cv2 import copyMakeBorder, BORDER_CONSTANT, threshold, THRESH_BINARY, findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE, drawContours


def new_drop_image(drop_profile, img_size_pix, rotation, drop_scale, noise, 
                   rel_capillary_height, above_apex, delta_s):

    # Grab drop radius
    relative_drop_radius = drop_profile[:, 0].max()
    
    # Add capillary tip
    if rel_capillary_height != 0:
        max_z_index = drop_profile[:, 1].argmax()
        xf, zf = drop_profile[max_z_index, :]
        z_new = arange(zf, zf + relative_drop_radius * rel_capillary_height, delta_s)
        x_new = array([xf] * len(z_new))
        cap = array(list(zip(x_new, z_new)))
        drop_profile = concatenate((drop_profile, cap))

    # Cutoff drop below apex
    if above_apex:
        max_x_index = drop_profile[:, 0].argmax()
        drop_profile = drop_profile[max_x_index:, :]

    # Mirror drop
    drop_z = drop_profile[:, 1]
    mirror_x = - drop_profile[:, 0]
    mirrored_drop_profile = array(list(zip(mirror_x, drop_z)))
    drop_profile = concatenate((drop_profile, mirrored_drop_profile))

    # Rotate drop
    rotation = radians(rotation)
    rotation_matrix = array([[cos(rotation), sin(rotation)], [-sin(rotation), cos(rotation)]])
    drop_profile = dot(drop_profile, rotation_matrix.T)
    relative_drop_radius = relative_drop_radius * cos(rotation)

    # Scale drop to image size
    drop_profile[:, 0] = drop_profile[:, 0] - drop_profile[:, 0].min()
    drop_profile[:, 1] = drop_profile[:, 1] - drop_profile[:, 1].min()
    max_length_scale = max(drop_profile.max(axis=0))
    scaler_factor = img_size_pix * drop_scale / max_length_scale
    drop_profile = drop_profile * scaler_factor
    relative_drop_radius = relative_drop_radius * scaler_factor / img_size_pix

    # Drop duplicate rows
    drop_profile = drop_profile.astype(int)
    drop_profile = unique(drop_profile, axis=0)

    # Flip image axis
    drop_profile[:, 1] = - drop_profile[:, 1]
    drop_profile[:, 1] = drop_profile[:, 1] - drop_profile[:, 1].min()

    # Shift drop in image
    x_shift = randrange(0, img_size_pix - drop_profile[:, 0].max())
    z_shift = randrange(0, img_size_pix - drop_profile[:, 1].max())
    drop_profile[:, 0] = drop_profile[:, 0] + x_shift
    drop_profile[:, 1] = drop_profile[:, 1] + z_shift

    # Draw drop points onto image
    drop_image = zeros((img_size_pix, img_size_pix))
    for i in range(len(drop_profile)):
        x, z = drop_profile[i, :]
        drop_image[z, x] = 255
    
    # Add salt and pepper noise
    if noise != 0:
        rand_matrix = rand(img_size_pix, img_size_pix)
        rand_matrix = where(rand_matrix < noise, 1, 0)
        rand_matrix = rand_matrix * 255
        drop_image = drop_image + rand_matrix
        drop_image = where(drop_image > 255, 255, drop_image)

    # Blur image
    drop_image = blur(drop_image, (3, 3))

    # Invert image
    drop_image = 255 - drop_image
        
    return relative_drop_radius, drop_image


def extract_drop_profile(img_path, thres):

    # Read img
    img = imread(str(img_path))
    img = cvtColor(img, COLOR_BGR2GRAY)

    # Threshold image
    _, img = threshold(img, thres, 255, THRESH_BINARY)

    # Find contours
    contours, hierarchy = findContours(img, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # Find largest contours (In terms of number of pixels in contour)
    largest_index = 0
    largest = 0
    for i, c in enumerate(contours):
        len_c = len(c)
        if len_c + largest:
            largest = len_c
            largest_index = i

    # Get points from contour
    contour = contours[largest_index]
    points = []
    for point in contour:
        point = point[0]
        points.append([point[0], point[1]])
    points = array(points)

    # Drop border pixels
    points = points[points[:, 0] > 0]
    points = points[points[:, 1] > 0]
    points = points[points[:, 0] < points[:, 0].max()]
    points = points[points[:, 1] < points[:, 1].max()]

    # Normalize points from drop radius
    points = points / points[:, 0].max()

    # Flip axis
    points[:, 1] = - points[:, 1]
    points[:, 1] = points[:, 1] - points[:, 1].min()

    # Cut drop off at min point in x-axis
    idx, _ = where(points == points[:, 1].min())
    cutoff_index = min(idx)
    points = points[:cutoff_index]

    return points
