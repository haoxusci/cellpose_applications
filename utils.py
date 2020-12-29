#!/usr/bin/env python
# coding: utf-8
import numpy as np
import time, os, sys
import imageio
from cellpose import models

use_GPU = models.use_gpu()


def segment_imgs(imgs, channels, model_type="cyto"):
    """
    Args:
        imgs (list): list of image arrays or paths.
        channels (list): list of channel specifications for imgs

    Note:
        # define CHANNELS to run segementation on
        # grayscale=0, R=1, G=2, B=3
        # channels = [cytoplasm, nucleus]
        # if NUCLEUS channel does not exist, set the second channel to 0
        # channels = [0,0]
        # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
        # channels = [0,0] # IF YOU HAVE GRAYSCALE
        # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

        # or if you have different types of channels in each image
        # channels = [[2,3], [0,0], [0,0]]
    """
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu=use_GPU, model_type=model_type)
    assert all(isinstance(item, str) for item in imgs) or all(
        isinstance(item, np.ndarray) for item in imgs
    ), "imgs is not a list of image paths or array"

    if isinstance(imgs[0], str):
        imgs = [imageio.imread(img) for _, img in enumerate(imgs)]

    assert len(imgs) == len(
        channels
    ), "imgs and channels should have same length"
    assert all(
        isinstance(item, list) for item in channels
    ), "channels should be a list of list"

    masks, flows, styles, diams = model.eval(
        imgs, diameter=None, flow_threshold=None, channels=channels
    )

    # return masks, flows, styles, diams
    return {"masks": masks, "flows": flows, "styles": styles, "diams": diams}


def read_to_rgb(fov_chs_files):
    """
    Read image files to rgb img array.
    
    Args:
        fov_chs_files (dict): a dict speicifying the image path for each channel
        
    
    Returns:
        list: return rgb image array and channel of list stating the cytoplasm and nuclei channel settings
    
    Example:
        fov_chs_files = {
            'nuclei': '/home/haoxu/data/test_data_20201218/40x/r08c22f01p01-ch1sk1fk1fl1.tiff',
            'er': '/home/haoxu/data/test_data_20201218/40x/r08c22f01p01-ch3sk1fk1fl1.tiff',
            'virus': '/home/haoxu/data/test_data_20201218/40x/r08c22f01p01-ch2sk1fk1fl1.tiff'
        }
    """
    fov_chs_imgs = {}
    for key, file in fov_chs_files.items():
        fov_chs_imgs[key] = imageio.imread(file)
        #print(key, item)
    fov_chs_rgb = np.array(
        [
            fov_chs_imgs['virus'],
            fov_chs_imgs['er'],
            fov_chs_imgs['nuclei']
        ]
    )
    fov_chs_rgb = np.transpose(fov_chs_rgb, (1, 2, 0))
    channel = [2, 3]

    return fov_chs_rgb, channel

def quantify_fov(img, mask):
    """
    Quantify virus img signal by using mask files
    
    Args:
        img (numpy array): 2D image array to be quantified.
        mask (numpy array): 2D image array with each cell being deassignated with a unique number.
        
    Returns:
        list: list of quantification for single cell.
              Speicifically, it follows this pattern-(cell_idx, cell_size, cell_integ)
    """
    if not mask.dtype == 'uint16':
        mask = mask.astype('uint16')
    cell_indexes = np.unique(mask)
    
    # create an empty list to host data
    fov_quantify = []

    if len(cell_indexes) < 2:
        print('No cells segmentated!')
        return
    
    for cell_idx in cell_indexes:
        if cell_idx == 0:
            continue
        current_cell_info = {}
        
        cell_mask_bool = (
                    mask == cell_idx
                )
        # get cell size
        cell_size = np.count_nonzero(cell_mask_bool)
        # retain selected cells in the img
        cell_in_img = img[cell_mask_bool]
        # get signal integration
        cell_integ = np.sum(cell_in_img)
        
        #current_cell_info['cell_size'] = cell_size
        #current_cell_info['cell_integ'] = cell_integ
        fov_quantify.append(
            (cell_idx, cell_size, cell_integ)
        )
        
    return fov_quantify