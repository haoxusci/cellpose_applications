import pandas as pd
from matplotlib import pyplot as plt
from covid_screening_quantify import quantify_all_fovs, combine_all_quantify

if __name__ == "__main__":
    """run image segmentation and quantification"""
    image_folder = "/home/haoxu/data/Images"
    quantify_all_fovs(image_folder)
    processed_folder = "/home/haoxu/data/Images_processed"

    combine_all_quantify(processed_folder)
