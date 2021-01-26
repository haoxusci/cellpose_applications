import os
from glob import glob
import imageio
import numpy as np
import pandas as pd
from imageio import imread
from matplotlib import pyplot as plt
import seaborn as sns
from skimage import segmentation, morphology, exposure
from cellpose_applications.utils import segment_imgs, read_to_rgb

CHS_MAPPING = {"nuclei": "ch1", "er": "ch3", "virus": "ch2"}


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
            lambda item: item
            if item.endswith(".tiff") and not item.startswith(".")
            else None,
            files,
        )
    )
    fovs = [file[:9] for file in files]
    fovs = list(set(fovs))
    fovs.sort()
    wells = [fov[:6] for fov in fovs]
    wells = list(set(wells))
    wells.sort()

    return wells, fovs, files


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
    if not mask.dtype == "uint16":
        mask = mask.astype("uint16")
    cell_indexes = np.unique(mask)

    # create an empty list to host data
    fov_quantify = []

    if len(cell_indexes) < 2:
        print("No cells segmentated!")
        return fov_quantify

    for cell_idx in cell_indexes:
        if cell_idx == 0:
            continue
        current_cell_info = {}

        cell_mask_bool = mask == cell_idx
        # get cell size
        cell_size = np.count_nonzero(cell_mask_bool)
        # retain selected cells in the img
        cell_in_img = img[cell_mask_bool]
        # get signal integration
        cell_integ = np.sum(cell_in_img)
        cell_mean = np.mean(cell_in_img)
        array_size = cell_in_img.size
        last4percentmean = np.mean(
            cell_in_img[np.argsort(cell_in_img)[int(array_size * 0.96) :]]
        ).astype(int)

        # current_cell_info['cell_size'] = cell_size
        # current_cell_info['cell_integ'] = cell_integ
        fov_quantify.append(
            (cell_idx, cell_size, cell_integ, cell_mean, last4percentmean)
        )

    return fov_quantify


def quantify_all_fovs(
    image_folder, processed_folder=None, save_mask=True, quantify_directly=True
):
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
        processed_folder = image_folder + "_processed"
    if save_mask or quantify_directly:
        os.makedirs(processed_folder, exist_ok=True)

    for _, well_id in enumerate(wells):
        # get all the fovs for the well
        well_fovs = list(
            filter(lambda fov: fov if fov[:6] == well_id else None, fovs)
        )
        well_fovs.sort()
        for _, well_fov in enumerate(well_fovs):
            # set the output for fov_data_output
            fov_data_output = []
            # get all the files for a fov
            fov_files = list(
                filter(lambda file: file if well_fov in file else None, files)
            )
            fov_files.sort()
            # this part is a simple assertation to make sure only one plane in zstack.
            # Otherwise, changes are needed fot this pipeline
            planes = [fov_file.split("-")[0] for fov_file in fov_files]
            planes = set(planes)
            assert (
                len(planes) == 1
            ), "This pipeline requires single plane image, but multiple planes/z-stack images found!"

            # CHS_MAPPING
            nuclei_file = list(
                filter(
                    lambda fov_file: fov_file
                    if fov_file.split("-")[1][:3] == CHS_MAPPING["nuclei"]
                    else None,
                    fov_files,
                )
            )
            er_file = list(
                filter(
                    lambda fov_file: fov_file
                    if fov_file.split("-")[1][:3] == CHS_MAPPING["er"]
                    else None,
                    fov_files,
                )
            )
            virus_file = list(
                filter(
                    lambda fov_file: fov_file
                    if fov_file.split("-")[1][:3] == CHS_MAPPING["virus"]
                    else None,
                    fov_files,
                )
            )
            assert (
                len(nuclei_file) == len(er_file) == len(virus_file) == 1
            ), "{} should have only one file for each channel!".format(
                well_fov
            )

            fov_chs_files = {
                "nuclei": os.path.join(image_folder, nuclei_file[0]),
                "er": os.path.join(image_folder, er_file[0]),
                "virus": os.path.join(image_folder, virus_file[0]),
            }

            # read image files to rgb img array with virus in r, er in g, nuclei in b
            fov_chs_rgb, channel = read_to_rgb(fov_chs_files)

            # segment the cell images
            results = segment_imgs([fov_chs_rgb], [channel])
            mask = results["masks"][0]
            # save mask data to file, if save_mask is true
            if save_mask:
                mask = np.asarray(mask, dtype="uint16")
                mask_file = os.path.join(
                    processed_folder, well_fov + "_mask.png"
                )
                imageio.imwrite(mask_file, mask, format="PNG-FI")

            # skip the quantificaiton if quantify_directly is false
            if not quantify_directly:
                continue
            # calculate the intensity
            fov_quantify = quantify_fov(fov_chs_rgb[:, :, 0], mask)
            # if no cells, continue
            if not len(fov_quantify):
                continue

            # write data to list, and then to csv file
            for cell_data in fov_quantify:
                (
                    cell_idx,
                    cell_size,
                    cell_integ,
                    cell_mean,
                    last4percentmean,
                ) = cell_data

                cell_info = {}
                cell_info["well_id"] = well_id
                cell_info["well_fov"] = well_fov
                cell_info["nuclei_file"] = fov_chs_files["nuclei"]
                cell_info["er_file"] = fov_chs_files["er"]
                cell_info["virus_file"] = fov_chs_files["virus"]
                if save_mask:
                    cell_info["mask_file"] = mask_file
                cell_info["cell_idx"] = cell_idx
                cell_info["cell_size"] = cell_size
                cell_info["cell_integ"] = cell_integ
                cell_info["cell_mean"] = cell_mean
                cell_info["last4percentmean"] = last4percentmean
                # append the data to fov_data_output
                fov_data_output.append(cell_info)
            # save the data from each fov to csv file
            df = pd.DataFrame(fov_data_output)
            csv_file = os.path.join(
                processed_folder, well_fov + "_quantify.csv"
            )

            df.to_csv(csv_file, index=False)

    return processed_folder


def combine_all_quantify(processed_folder, combine_file=True):
    """
    Combine fov quantify csv into one combined csv file.

    Args:
        processed_folder (str): the path of processed folder.
        combine_file (bool): save combined csv if true, else not.
    """
    quantify_csvs = glob(processed_folder + "/*_quantify.csv")
    quantiy_csvs = list(
        filter(
            lambda quantify_csv: quantify_csv
            if not os.path.basename(quantify_csv).startswith(".")
            else None,
            quantify_csvs,
        )
    )
    quantiy_csvs.sort()
    dfs = []
    for _, quantify_csv in enumerate(quantify_csvs):
        try:
            df = pd.read_csv(quantify_csv)
            dfs.append(df)
        except Exception as e:
            print(f"{quantify_csv} cannot be successfully read!")
    df = pd.concat(dfs)
    if combine_file:
        df.to_csv(
            os.path.join(processed_folder, "quantify_all.csv"),
            index=False,
        )
    else:
        return df


def write_conditions(quantify_csv_file, condition_xlsx_file):
    """write conditions to quantify_all_file.
    Args:
        quantify_csv_file (str): quantify csv file path.
        condition_xlsx_file (str): conditions xlsx file path.
    """
    df = pd.read_csv(quantify_csv_file)
    # get the conditions
    conditions_data = pd.read_excel(condition_xlsx_file)
    letters = "abcdefghijklmnop".upper()

    df = df.assign(**dict.fromkeys(["Compound"], np.nan))
    for item in conditions_data["Destination Well"]:
        row = letters.index(item[0]) + 1
        column = int(item[1:])
        well_id = f"r{row:02d}c{column:02d}"
        condition_rows = conditions_data.loc[
            conditions_data["Destination Well"] == item
        ]
        if len(condition_rows):
            try:
                condition = condition_rows.iloc[0].Compound
                df.loc[df.well_id == well_id, "Compound"] = condition
            except:
                print(item)
                print(condition_rows)
    # save again the file
    df.to_csv(quantify_csv_file, index=False)


def adjust_image(img):
    """Adjust image intensity for better visualization.
    Args:
        img (numpy array): image array of 3d, e.g., (1080, 1080, 3)

    Returns:
        numpy array: image array of 3d after intensity adjustment.
    """
    # img = np.stack([image['red'], image['green'], image['blue']], axis=2)
    # img = img/img.max(axis=(0,1))
    # img = rescale(img, 0.5, anti_aliasing=True)
    dtype = img.dtype
    p2, p99 = np.percentile(img, (0, 99.9))
    img = exposure.rescale_intensity(img, in_range=(p2, p99))
    img = img.astype(dtype)

    return img


def add_contours(mask_img, img, ch):
    """Add contours on img on a rgb image.
    Args:
        mask_image (numpy array): 2d image array, with each cell being attributed with one unique value.
        img (numpy array): 3d image array, rgb image.
        ch (int): designate the channel index that shows the cell contours.

    Returns:
        numpy array: image array with cell contours being added onto the image.
    """
    max_value = img.max()
    dtype = img.dtype

    cell_contours = segmentation.find_boundaries(mask_img)
    cell_contours = morphology.binary_dilation(
        cell_contours, selem=np.full((2, 2), 1)
    )
    cell_contours_invert = np.invert(cell_contours)
    img_ch = img[:, :, ch]
    img_ch = img_ch * cell_contours_invert
    img_ch = img_ch.astype(dtype)
    # print(img.dtype)
    contours = cell_contours.astype(dtype) * max_value
    img[:, :, ch] = img_ch + contours

    return img


def select_cells(mask_image, cell_indice):
    """get boolean mask for seleted cells.
    Args:
        mask_image (numpy array): 2d image array, with each cell being attributed with one unique value.
        cell_indice (list): list of values that indicates the cells to be retain.

    Returns:
        numpy array: mask of boolean values for selected cells.
    """
    dtype = mask_image.dtype
    mask_selected = np.full_like(mask_image, 0, dtype=dtype)
    for i, cell in enumerate(cell_indice):
        cell_mask = mask_image * (mask_image == cell)
        mask_selected = np.where(
            mask_selected > cell_mask, mask_selected, cell_mask
        )

    return mask_selected


def show_infections(df, well_fov, dim_er=False, contour=True, show_er=True):
    """Show cell infections.
    Args:
        df (pandas DataFrame): pandas dataframe with infected cells indicated.
        well_fov (str): well fov info.

    Returns:
        numpy array: image array with infected and noninfected cells marked.
    """
    cell_examples = df.loc[df.well_fov == well_fov]
    if not len(cell_examples):
        raise Exception("No cell sample found, try a different fov!")
    infected_cells = df.loc[
        (df.well_fov == well_fov) & (df.Infected == 1)
    ].cell_idx.to_list()
    noninfected_cells = df.loc[
        (df.well_fov == well_fov) & (df.Infected == 0)
    ].cell_idx.to_list()

    fov_example = df.loc[df.well_fov == well_fov].iloc[0]

    nuclei_image = imread(fov_example.nuclei_file)
    if not show_er:
        er_image = np.full_like(nuclei_image, 0)
    else:
        er_image = imread(fov_example.er_file)
    virus_image = imread(fov_example.virus_file)

    img = np.stack([virus_image, er_image, nuclei_image], axis=2)
    img = adjust_image(img)
    if dim_er and show_er:
        img[:, :, 1] = img[:, :, 1] // 1.5
    if not contour:
        return img
    mask_image = imread(fov_example.mask_file)
    mask_infected = select_cells(mask_image, infected_cells)
    mask_noninfected = select_cells(mask_image, noninfected_cells)
    img = add_contours(mask_infected, img, ch=0)
    img = add_contours(mask_noninfected, img, ch=2)

    return img


def calc_infection(df, df_well_ids, well_id):
    """Calculate the infection rate for each well.
    Args:
        df (pandas DataFrame): pandas dataframe with infected cells indicated.
        df_well_ids (list): available well_ids in df dataframe.
        well_id (str): well id info.

    Returns:
        dict: cell_count, infected_cell_count, infection_rate.
    """
    if not well_id in df_well_ids:
        # print(f"there is no cell detected in {well_id}")
        # if no cells found, give infection rate as -50
        return {
            "cell_count": 0,
            "infected_cell_count": 0,
            "infection_rate": -50,
        }
    else:
        well_cells = df.loc[df.well_id == well_id]
        cell_count = len(well_cells)
        infected_cell_count = len(well_cells.loc[well_cells.Infected == 1])
        infection_rate = round(infected_cell_count / cell_count * 100, 2)

        return {
            "cell_count": cell_count,
            "infected_cell_count": infected_cell_count,
            "infection_rate": infection_rate,
        }


def plate_infection(processed_folder, infection_threshold=3000):
    """Calculate all the infection rate in the plate.
    Args:
        processed_folder (str): the path of processed folder.
        infection_threshold (int): threshold value for cell infection.

    Returns:
        dict: cell infection numbers
    """
    df = pd.read_csv(os.path.join(processed_folder, "quantify_all.csv"))
    # define if cell is infected
    df = df.assign(**dict.fromkeys(["Infected"], 0))
    df.loc[df.last4percentmean > infection_threshold, "Infected"] = 1

    df_well_ids = df.well_id.to_list()
    df_well_ids.sort()
    df_well_ids = set(df_well_ids)

    row_number = 16
    column_number = 24

    cell_counts = []
    infected_cell_counts = []
    infection_rates = []
    for row in range(1, row_number + 1):
        for column in range(1, column_number + 1):
            well_id = f"r{row:02d}c{column:02d}"
            infection_info = calc_infection(df, df_well_ids, well_id)
            cell_count, infected_cell_count, infection_rate = (
                infection_info["cell_count"],
                infection_info["infected_cell_count"],
                infection_info["infection_rate"],
            )

            cell_counts.append(cell_count)
            infected_cell_counts.append(infected_cell_count)
            infection_rates.append(infection_rate)
    cell_counts = np.array(cell_counts)
    cell_counts = cell_counts.reshape((row_number, column_number))
    infected_cell_counts = np.array(infected_cell_counts)
    infected_cell_counts = infected_cell_counts.reshape(
        (row_number, column_number)
    )
    infection_rates = np.array(infection_rates)
    infection_rates = infection_rates.reshape((row_number, column_number))

    return {
        "row_number": row_number,
        "column_number": column_number,
        "cell_counts": cell_counts,
        "infected_cell_counts": infected_cell_counts,
        "infection_rates": infection_rates,
    }


def plate_plots(processed_folder, infection_threshold=3000):
    """Plot cell counts, infected cell counts, infection rates for a plate of processed folder
    Args:
        processed_folder (str): the path of processed folder.
        infection_threshold (int): threshold value for cell infection.
    """
    infection_data = plate_infection(processed_folder, infection_threshold)
    sns.set_theme()
    data = infection_data["infection_rates"]
    mask = np.zeros_like(data)
    mask = np.where(data >= 0, mask, True)

    xticklabels = [
        str(i) for i in range(1, 1 + infection_data["column_number"])
    ]
    yticklabels = "abcdefghijklmnop"[: infection_data["row_number"]].upper()
    yticklabels = list(yticklabels)
    for item in infection_data.keys():
        if item in ["row_number", "column_number"]:
            continue
        legend_label = item.replace("_", " ")
        if "rate" in item:
            legend_label += " (%)"
            # vmax = 100
        # else:
        # vmax=None
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            infection_data[item],
            vmin=0,
            vmax=None,
            center=0,
            linewidths=0.5,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            mask=mask,
            cbar_kws={"label": legend_label, "shrink": 0.5},
        )
        plt.yticks(rotation=0)
        plot_folder = os.path.join(processed_folder, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        plt.savefig(
            os.path.join(plot_folder, item + "_plate_map.png"),
            bbox_inches="tight",
            dpi=100,
        )

    ## scatter plots
    df = pd.read_csv(os.path.join(processed_folder, "quantify_all.csv"))
    compounds = list(set(df.Compound.to_list()))
    df_compounds = pd.DataFrame
    calculate_items = [
        "cell_counts",
        "infected_cell_counts",
        "infection_rates",
    ]
    compounds_infections = []
    for compound in compounds:
        wells_selected = list(set(df.loc[df.Compound == compound].well_id))
        for well_selected in wells_selected:
            compound_infection = {}
            row_index = int(well_selected[1:3]) - 1
            column_index = int(well_selected[4:]) - 1
            compound_infection["well_id"] = well_selected
            for _, calculate_item in enumerate(calculate_items):
                compound_infection[calculate_item] = infection_data[
                    calculate_item
                ][row_index][column_index]
            compound_infection["Compound"] = compound
            compounds_infections.append(compound_infection)

    df2 = pd.DataFrame(compounds_infections)
    df2.to_csv(
        os.path.join(processed_folder, "infection_data.csv"), index=False
    )
    sns.set_theme(style="ticks", color_codes=True)

    for _, calculate_item in enumerate(calculate_items):
        plt.figure(figsize=(10, 20))
        sns.catplot(
            x="Compound", y=calculate_item, kind="box", data=df2, aspect=2
        )
        plt.xticks(rotation=45)
        ylabel = calculate_item.replace("_", " ")
        if ylabel == "infection rates":
            ylabel = "infection rates (%)"
        plt.ylabel(ylabel)
        plt.savefig(
            os.path.join(plot_folder, calculate_item + "_bbox.png"),
            bbox_inches="tight",
            dpi=100,
        )
