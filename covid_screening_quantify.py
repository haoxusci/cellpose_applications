import os
from glob import glob
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import segment_imgs, read_to_rgb, quantify_fov

CHS_MAPPING = {
    'nuclei': 'ch1',
    'er': 'ch2',
    'virus': 'ch3'
}


def sort_wells_fovs(image_folder):
    """
    sort the wells and fovs in an image folder
    
    Args:
        image_folder (str): path of image folder, which hosts all the images.
        
    Returns:
        tuple: wells and fovs list
    """
    files = os.listdir(image_folder)
    files = list(
        filter(
            lambda item:
                item if item.endswith('.tiff') and not item.startswith('.') else None,
            files
        )
    )
    fovs = [file[:9] for file in files]
    fovs = list(set(fovs))
    fovs.sort()
    wells = [fov[:6] for fov in fovs]
    wells = list(set(wells))
    wells.sort()
    
    return wells, fovs, files


def quantify_all_fovs(image_folder, processed_folder=None, save_mask=True, quantify_directly=True):
    """
    Quantify all the fovs in image folder.
    
    Args:
        image_folder (str): path to folder where hosts all the images.
        save_mask (bool): to save mask as file or not.
        quantify_directly (bool): to quantify the cell data directly or not.
    
    Returns:
        str: the path of processed folder.
    """
    wells, fovs, files = sort_wells_fovs(image_folder)
    # create processed folder if to save files
    if not processed_folder:
        processed_folder = image_folder + '_processed'
    if save_mask or quantify_directly:
        os.makedirs(processed_folder, exist_ok=True)

    for _, well_id in enumerate(wells):
        # get all the fovs for the well
        well_fovs = list(
            filter(
                lambda fov: fov if fov[:6] == well_id else None, fovs
            )
        )
        well_fovs.sort()
        for _, well_fov in enumerate(well_fovs):
            # set the output for fov_data_output
            fov_data_output = []
            # get all the files for a fov
            fov_files = list(
                filter(
                    lambda file: file if well_fov in file else None, files
                )
            )
            fov_files.sort()
            # this part is a simple assertation to make sure only one plane in zstack.
            # Otherwise, changes are needed fot this pipeline
            planes = [fov_file.split('-')[0] for fov_file in fov_files]
            planes = set(planes)
            assert len(planes) == 1, "This pipeline requires single plane image, but multiple planes/z-stack images found!"

            #CHS_MAPPING
            nuclei_file = list(
                filter(
                    lambda fov_file:
                        fov_file if fov_file.split('-')[1][:3] == CHS_MAPPING['nuclei'] else None,
                    fov_files
                )
            )
            er_file = list(
                filter(
                    lambda fov_file:
                        fov_file if fov_file.split('-')[1][:3] == CHS_MAPPING['er'] else None,
                    fov_files
                )
            )
            virus_file = list(
                filter(
                    lambda fov_file:
                        fov_file if fov_file.split('-')[1][:3] == CHS_MAPPING['virus'] else None,
                    fov_files
                )
            )
            assert len(nuclei_file) == len(er_file) == len(virus_file) == 1, "{} should have only one file for each channel!".format(well_fov)

            fov_chs_files = {
                'nuclei': os.path.join(image_folder, nuclei_file[0]),
                'er': os.path.join(image_folder, er_file[0]),
                'virus': os.path.join(image_folder, virus_file[0])
            }

            # read image files to rgb img array with virus in r, er in g, nuclei in b
            fov_chs_rgb, channel =  read_to_rgb(fov_chs_files)

            # segment the cell images
            results = segment_imgs([fov_chs_rgb], [channel])
            mask = results['masks'][0]
            # save mask data to file, if save_mask is true
            if save_mask:
                mask = np.asarray(mask, dtype='uint16')
                mask_file = os.path.join(processed_folder, well_fov + '_mask.png')
                imageio.imwrite(mask_file, mask, format='PNG-FI')

            # skip the quantificaiton if quantify_directly is false
            if not quantify_directly:
                continue
            # calculate the intensity
            fov_quantify = quantify_fov(fov_chs_rgb[:,:,0], mask)
            # if no cells, continue
            if not len(fov_quantify):
                continue

            # write data to list, and then to csv file
            for cell_data in fov_quantify:
                cell_idx, cell_size, cell_integ, cell_mean = cell_data

                cell_info = {}
                cell_info['well_id'] = well_id
                cell_info['well_fovs'] = well_fov
                cell_info['nuclei_file'] = fov_chs_files['nuclei']
                cell_info['er_file'] = fov_chs_files['er']
                cell_info['virus_file'] = fov_chs_files['virus']
                if save_mask:
                    cell_info['mask_file'] = mask_file
                cell_info['cell_idx'] = cell_idx
                cell_info['cell_size'] = cell_size
                cell_info['cell_integ'] = cell_integ
                cell_info['cell_mean'] = cell_mean
                # append the data to fov_data_output
                fov_data_output.append(cell_info)
            # save the data from each fov to csv file
            df = pd.DataFrame(fov_data_output)
            csv_file = os.path.join(processed_folder, well_fov + '_quantify.csv')

            df.to_csv(csv_file, index=False)
    
    return processed_folder


def combine_all_quantify(processed_folder, combine_file=True):
    """
    Combine fov quantify csv into one combined csv file.
    
    Args:
        processed_folder (str): the path of processed folder.
        combine_file (bool): save combined csv if true, else not.
    """
    quantify_csvs = glob(processed_folder + '/*_quantify.csv')
    quantiy_csvs = list(
        filter(
            lambda quantify_csv:
                quantify_csv if not os.path.basename(quantify_csv).startswith('.') else None,
            quantify_csvs
        )
    )
    quantiy_csvs.sort()
    dfs = []
    for _, quantify_csv in enumerate(quantify_csvs):
        try:
            df = pd.read_csv(quantify_csv)
            dfs.append(df)
        except Exception as e:
            print(f'{quantify_csv} cannot be successfully read!')
    df = pd.concat(dfs)
    if combine_file:
        df.to_csv(
            '/home/haoxu/data/test_data_20201218/40x_processed/quantify_all.csv',
            index=False
        )
    else:
        return df