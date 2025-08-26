import pandas as pd
import numpy as np
import os
import glob
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


def create_bcg_coordinates_csv():
    """
    Create CSV files with x,y coordinates for both 2.2 and 3.8 arcmin images
    """
    # Read truth table
    truth_table_path = '/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/bcgs_parsed_for_truth_table_post_human_inspect.csv'
    input_df = pd.read_csv(truth_table_path)
    
    # Process both image directories
    directories = {
        '2p2arcmin': '/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/2p2arcmin/',
        '3p8arcmin': '/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/3p8arcmin/'
    }
    
    for dir_name, image_dir in directories.items():
        print(f"Processing {dir_name} images...")
        
        # Get all .tif files in directory
        image_files = glob.glob(os.path.join(image_dir, '*.tif'))
        
        results = []
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            cluster_name = filename.split('_')[0]
            
            # Find matching entries in truth table
            matching_rows = input_df[input_df['Cluster name'] == cluster_name]
            
            if len(matching_rows) == 0:
                print(f"Warning: No truth table entry found for {cluster_name}")
                continue
            
            # Load image to get WCS
            try:
                pil_image = pillow.open(image_path)
                pil_image.seek(0)
                wcs = read_wcs(pil_image)
                pil_image.close()
                
                # Process each matching row (could be multiple BCGs per cluster)
                for _, row in matching_rows.iterrows():
                    bcg_ra = row['BCG RA']
                    bcg_dec = row['BCG Dec']
                    
                    # Convert RA/Dec to pixel coordinates
                    x_pixel, y_pixel = wcs.world_to_pixel_values(bcg_ra, bcg_dec)
                    
                    # Y-coordinate inversion to match image coordinate system
                    y_pixel_corrected = 512 - y_pixel
                    
                    # Create result row
                    result_row = row.to_dict()
                    result_row['Filename'] = filename
                    result_row['x'] = x_pixel
                    result_row['y'] = y_pixel_corrected
                    
                    results.append(result_row)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        output_path = f'/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/bcgs_{dir_name}_with_coordinates.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Saved {len(results_df)} entries to {output_path}")


if __name__ == "__main__":
    create_bcg_coordinates_csv()