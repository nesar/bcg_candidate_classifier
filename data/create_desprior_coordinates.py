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


def create_desprior_coordinates_csv():
    """
    Create CSV files with x,y coordinates for DESprior candidates in both 2.2 and 3.8 arcmin images
    """
    # Read prior locations
    prior_locations_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/prior_locations.csv'
    prior_df = pd.read_csv(prior_locations_path)
    
    print(f"Loaded {len(prior_df)} candidate locations for {len(prior_df['cluster'].unique())} clusters")
    
    # Process both image directories
    directories = {
        '2p2arcmin': '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/2p2arcmin/',
        '3p8arcmin': '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin/'
    }
    
    for dir_name, image_dir in directories.items():
        print(f"\nProcessing DESprior candidates for {dir_name} images...")
        
        # Get all .tif files in directory
        image_files = glob.glob(os.path.join(image_dir, '*.tif'))
        print(f"Found {len(image_files)} image files")
        
        results = []
        clusters_processed = 0
        candidates_processed = 0
        candidates_outside_image = 0
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            cluster_name = filename.split('_')[0]
            
            # Find matching entries in prior locations
            matching_candidates = prior_df[prior_df['cluster'] == cluster_name]
            
            if len(matching_candidates) == 0:
                print(f"Warning: No prior candidates found for {cluster_name}")
                continue
            
            # Load image to get WCS
            try:
                pil_image = pillow.open(image_path)
                pil_image.seek(0)
                wcs = read_wcs(pil_image)
                pil_image.close()
                
                clusters_processed += 1
                
                # Process each candidate for this cluster
                for _, candidate in matching_candidates.iterrows():
                    candidate_ra = candidate['ra']
                    candidate_dec = candidate['dec']
                    delta_mstar = candidate['delta_mstar']
                    starflag = candidate['starflag']
                    
                    # Convert RA/Dec to pixel coordinates
                    x_pixel, y_pixel = wcs.world_to_pixel_values(candidate_ra, candidate_dec)
                    
                    # Y-coordinate inversion to match image coordinate system
                    y_pixel_corrected = 512 - y_pixel
                    
                    # Check if candidate is within image bounds
                    is_inside = (0 <= x_pixel <= 512) and (0 <= y_pixel_corrected <= 512)
                    
                    if not is_inside:
                        candidates_outside_image += 1
                    
                    # Create result row
                    result_row = {
                        'cluster': cluster_name,
                        'filename': filename,
                        'candidate_ra': candidate_ra,
                        'candidate_dec': candidate_dec,
                        'x': x_pixel,
                        'y': y_pixel_corrected,
                        'delta_mstar': delta_mstar,
                        'starflag': starflag,
                        'is_inside_image': is_inside
                    }
                    
                    results.append(result_row)
                    candidates_processed += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        output_path = f'/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_{dir_name}_with_coordinates.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"Results for {dir_name}:")
        print(f"  - Clusters processed: {clusters_processed}")
        print(f"  - Candidates processed: {candidates_processed}")
        print(f"  - Candidates outside image bounds: {candidates_outside_image}")
        print(f"  - Candidates inside image bounds: {candidates_processed - candidates_outside_image}")
        print(f"  - Saved to: {output_path}")
        
        # Show some statistics per cluster
        if len(results_df) > 0:
            candidates_per_cluster = results_df.groupby('cluster').size()
            inside_candidates_per_cluster = results_df[results_df['is_inside_image']].groupby('cluster').size()
            
            print(f"  - Average candidates per cluster: {candidates_per_cluster.mean():.1f}")
            print(f"  - Average inside candidates per cluster: {inside_candidates_per_cluster.mean():.1f}")
            print(f"  - Min/Max candidates per cluster: {candidates_per_cluster.min()}/{candidates_per_cluster.max()}")


if __name__ == "__main__":
    create_desprior_coordinates_csv()