import numpy as np
import pandas as pd
import os
import glob
import torch
from torch.utils.data import Dataset
from astropy.wcs import WCS
from astropy.io import fits
from PIL.TiffTags import TAGS
import PIL.Image as pillow
from io import StringIO


def read_wcs(file):
    """
    Read WCS information from TIF files with WCS information embedded in header.
    
    Args:
        file: PIL Image file object
        
    Returns:
        astropy.wcs.WCS object
    """
    meta_dict = {TAGS[key]: file.tag[key] for key in file.tag_v2}
    
    long_header_str = meta_dict['ImageDescription'][0]
    line_length = 80

    # Splitting the string into lines of 80 characters
    lines = [long_header_str[i:i+line_length] for i in range(0, len(long_header_str), line_length)]
    
    # Join the lines with newline characters to form a properly formatted header string
    corrected_header_str = "\n".join(lines)

    # Use an IO stream to mimic a file
    header_stream = StringIO(corrected_header_str)

    # Read the header using astropy.io.fits
    header = fits.Header.fromtextfile(header_stream)
    wcs = WCS(header)
    
    return wcs


def prepare_dataframe(image_dir, truth_table_path, dataset_type='megadeep500'):
    """
    Prepare dataframe matching cluster filenames with BCG coordinates.
    
    Args:
        image_dir: Directory containing .tif images
        truth_table_path: Path to CSV file with cluster information
        dataset_type: Either 'megadeep500' or 'SPT3G_1500d' to handle different CSV formats
        
    Returns:
        pandas DataFrame with columns: Filename, BCG RA, BCG Dec
    """
    input_clusters_df = pd.read_csv(truth_table_path)
    
    cluster_names_list = list(input_clusters_df.loc[:]['Cluster name'])
    
    if dataset_type == 'SPT3G_1500d':
        ra_list = list(input_clusters_df.loc[:]['RA'])
        dec_list = list(input_clusters_df.loc[:]['Dec'])
    else:
        ra_list = list(input_clusters_df.loc[:]['BCG RA'])
        dec_list = list(input_clusters_df.loc[:]['BCG Dec'])
    
    cluster_filenames = []
    new_ra_list = []
    new_dec_list = []
    
    for full_filename in glob.glob(image_dir + '*.tif'):
        filename = os.path.basename(full_filename)
        cluster_name = filename.split('_')[0]
        if cluster_name in cluster_names_list:
            # Find all indices for this cluster name to handle multiple entries correctly
            indices = [i for i, name in enumerate(cluster_names_list) if name == cluster_name]
            for index in indices:
                cluster_filenames.append(filename)
                new_ra_list.append(ra_list[index])
                new_dec_list.append(dec_list[index])
    
    df_dict = {
        'Filename': cluster_filenames,
        'BCG RA': new_ra_list,
        'BCG Dec': new_dec_list
    }
    
    dataframe = pd.DataFrame(df_dict)
    
    return dataframe


class BCGDataset(Dataset):
    """
    Dataset class for loading astronomical images and BCG coordinates.
    """
    
    def __init__(self, image_dir, dataframe):
        """
        Args:
            image_dir: Directory containing .tif image files
            dataframe: DataFrame with Filename, BCG RA, BCG Dec columns
        """
        self.image_dir = image_dir
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row['Filename']
        bcg_ra = row['BCG RA']
        bcg_dec = row['BCG Dec']
        
        # Load image
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image using PIL
        pil_image = pillow.open(image_path)
        
        # Extract the desired frame from multi-frame TIFF (frame 1, not 0)
        pil_image.seek(0) 
        image_array = np.asarray(pil_image)
        
        # Get WCS information
        wcs = read_wcs(pil_image)
        
        # Convert RA/Dec to pixel coordinates
        x_pixel, y_pixel = wcs.world_to_pixel_values(bcg_ra, bcg_dec)
        
        # Y-coordinate inversion to match image coordinate system
        y_pixel_corrected = 512 - y_pixel
        
        pil_image.close()
        
        bcg_coords = np.array([x_pixel, y_pixel_corrected])
        
        # Basic validity checks
        if np.any(np.isnan(bcg_coords)) or np.any(np.isinf(bcg_coords)):
            print(f"Warning: Invalid coordinates for {filename}: {bcg_coords}")
        
        return {
            'image': image_array,
            'BCG': bcg_coords,
            'filename': filename
        }