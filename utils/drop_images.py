from math import radians
from random import randrange

from numpy import array, arange, concatenate, dot, where, zeros, sin, cos, unique
from numpy.random import rand
from cv2 import blur


def new_drop_image(drop_profile, img_size_pix, rotation, drop_scale, noise, 
                   rel_capillary_height, above_apex, delta_s, shift=True):

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

    # Scale drop to image size
    drop_profile[:, 0] = drop_profile[:, 0] - drop_profile[:, 0].min()
    drop_profile[:, 1] = drop_profile[:, 1] - drop_profile[:, 1].min()
    max_length_scale = max(drop_profile.max(axis=0))
    scaler_factor = (img_size_pix - 1) * drop_scale / max_length_scale
    drop_profile = drop_profile * scaler_factor

    relative_drop_radius = relative_drop_radius * cos(rotation)
    relative_drop_radius = relative_drop_radius * scaler_factor / img_size_pix

    # Drop duplicate rows
    drop_profile = drop_profile.astype(int)
    drop_profile = unique(drop_profile, axis=0)

    # Flip image axis
    drop_profile[:, 1] = - drop_profile[:, 1]
    drop_profile[:, 1] = drop_profile[:, 1] - drop_profile[:, 1].min()

    # Shift drop in image
    if shift:
        x_shift = randrange(0, img_size_pix - drop_profile[:, 0].max())
        z_shift = randrange(0, img_size_pix - drop_profile[:, 1].max())
    else:
        x_shift = (img_size_pix - drop_profile[:, 0].max()) / 2
        z_shift = (img_size_pix - drop_profile[:, 1].max()) / 2

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
