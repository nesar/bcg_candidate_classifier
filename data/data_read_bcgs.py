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


def prepare_bcg_dataframe(csv_path):
    """
    Prepare dataframe from BCG CSV with coordinates already calculated.
    
    Args:
        csv_path: Path to CSV file with BCG information including x,y coordinates
        
    Returns:
        pandas DataFrame with columns: Filename, BCG RA, BCG Dec, x, y, Cluster z, BCG Probability, delta_mstar_z
    """
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_columns = ['Filename', 'BCG RA', 'BCG Dec', 'x', 'y', 'Cluster z', 'BCG Probability', 'delta_mstar_z']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter out entries with invalid coordinates
    valid_mask = (~pd.isna(df['x'])) & (~pd.isna(df['y'])) & \
                 (df['x'] >= 0) & (df['x'] <= 512) & \
                 (df['y'] >= 0) & (df['y'] <= 512)
    
    df_clean = df[valid_mask].copy()
    
    print(f"Loaded {len(df_clean)} valid BCG entries from {len(df)} total entries")
    
    return df_clean


class BCGDataset(Dataset):
    """
    Dataset class for loading astronomical images and BCG coordinates with additional features.
    """
    
    def __init__(self, image_dir, dataframe, include_additional_features=True):
        """
        Args:
            image_dir: Directory containing .tif image files
            dataframe: DataFrame with BCG information including coordinates
            include_additional_features: Whether to include redshift and delta_mstar_z as features
        """
        self.image_dir = image_dir
        self.dataframe = dataframe.reset_index(drop=True)
        self.include_additional_features = include_additional_features
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row['Filename']
        bcg_ra = row['BCG RA']
        bcg_dec = row['BCG Dec']
        x_pixel = row['x']
        y_pixel = row['y']
        cluster_z = row['Cluster z']
        bcg_probability = row['BCG Probability']
        delta_mstar_z = row['delta_mstar_z']
        
        # Load image
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image using PIL
        pil_image = pillow.open(image_path)
        
        # Extract the desired frame from multi-frame TIFF (frame 1, not 0)
        pil_image.seek(0) 
        image_array = np.asarray(pil_image)
        
        pil_image.close()
        
        bcg_coords = np.array([x_pixel, y_pixel])
        
        # Basic validity checks
        if np.any(np.isnan(bcg_coords)) or np.any(np.isinf(bcg_coords)):
            print(f"Warning: Invalid coordinates for {filename}: {bcg_coords}")
        
        result = {
            'image': image_array,
            'BCG': bcg_coords,
            'filename': filename,
            'bcg_ra': bcg_ra,
            'bcg_dec': bcg_dec,
            'cluster_z': cluster_z,
            'bcg_probability': bcg_probability,
            'delta_mstar_z': delta_mstar_z
        }
        
        # Add additional features if requested
        if self.include_additional_features:
            result['additional_features'] = np.array([cluster_z, delta_mstar_z])
        
        return result


# MultiScaleBCGDataset removed - use separate 2.2 or 3.8 arcmin datasets instead


# Utility function to create datasets
def create_bcg_datasets(dataset_type='2p2arcmin', split_ratio=0.8, random_seed=42):
    """
    Create train/test datasets for BCG data.
    
    Args:
        dataset_type: Either '2p2arcmin' or '3p8arcmin'
        split_ratio: Fraction of data to use for training
        random_seed: Random seed for reproducible splits
        
    Returns:
        train_dataset, test_dataset
    """
    np.random.seed(random_seed)
    
    # Single scale dataset
    if dataset_type == '2p2arcmin':
        image_dir = '/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/2p2arcmin/'
        csv_path = '/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/bcgs_2p2arcmin_with_coordinates.csv'
    elif dataset_type == '3p8arcmin':
        image_dir = '/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/3p8arcmin/'
        csv_path = '/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/bcgs_3p8arcmin_with_coordinates.csv'
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use '2p2arcmin' or '3p8arcmin'")
    
    df = prepare_bcg_dataframe(csv_path)
    full_dataset = BCGDataset(image_dir, df)
    
    # Create train/test split
    n_total = len(full_dataset)
    n_train = int(n_total * split_ratio)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"Created train dataset with {len(train_dataset)} samples")
    print(f"Created test dataset with {len(test_dataset)} samples")
    
    return train_dataset, test_dataset