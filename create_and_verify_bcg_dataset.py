#!/usr/bin/env python3
"""
BCG Dataset Creator and Verifier

This script consolidates coordinate generation and comprehensive data analysis for BCG datasets.
It creates coordinate files, analyzes data matching, and ensures the ML pipeline has clean datasets.

Features:
1. Generates BCG coordinates from truth table
2. Generates DESprior candidate coordinates 
3. Comprehensive data matching analysis
4. Creates clean, matched datasets for ML training
5. Detailed reporting and statistics

Usage:
    python create_and_verify_bcg_dataset.py [--regenerate] [--analysis-only]
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from collections import defaultdict
from astropy.wcs import WCS
from astropy.io import fits
from PIL.TiffTags import TAGS
import PIL.Image as pillow
from io import StringIO


class BCGDatasetManager:
    """Comprehensive BCG dataset management and analysis"""
    
    def __init__(self):
        # Data paths
        self.truth_table_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_parsed_for_truth_table_post_human_inspect.csv' 
        # self.prior_locations_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/prior_locations.csv' #pre-purge
        self.prior_locations_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/prior_locations_more_info_PURGED_111925.csv' #removed a bunch of candidates
        self.image_dirs = {
            '2p2arcmin': '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/2p2arcmin/',
            '3p8arcmin': '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin/'
        }
        
        # Output paths
        self.output_dir = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/'
        self.bcg_coord_files = {
            '2p2arcmin': os.path.join(self.output_dir, 'bcgs_2p2arcmin_with_coordinates.csv'),
            '3p8arcmin': os.path.join(self.output_dir, 'bcgs_3p8arcmin_with_coordinates.csv')
        }
        self.desprior_coord_files = {
            '2p2arcmin': os.path.join(self.output_dir, 'desprior_candidates_2p2arcmin_with_coordinates.csv'),
            '3p8arcmin': os.path.join(self.output_dir, 'desprior_candidates_3p8arcmin_with_coordinates.csv')
        }
        self.clean_bcg_files = {
            '2p2arcmin': os.path.join(self.output_dir, 'bcgs_2p2arcmin_clean_matched.csv'),
            '3p8arcmin': os.path.join(self.output_dir, 'bcgs_3p8arcmin_clean_matched.csv')
        }
        # self.clean_desprior_files = { #pre-purge
        #     '2p2arcmin': os.path.join(self.output_dir, 'desprior_candidates_2p2arcmin_clean_matched.csv'),
        #     '3p8arcmin': os.path.join(self.output_dir, 'desprior_candidates_3p8arcmin_clean_matched.csv')
        # }

        self.clean_desprior_files = { #post-purge
            '2p2arcmin': os.path.join(self.output_dir, 'desprior_candidates_2p2arcmin_purge_matched.csv'),
            '3p8arcmin': os.path.join(self.output_dir, 'desprior_candidates_3p8arcmin_purge_matched.csv')
        }
        
        # Analysis storage
        self.analysis_results = {}
    
    def read_wcs(self, file):
        """Read WCS information from TIF files with WCS information embedded in header."""
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
    
    def load_raw_data(self):
        """Load all raw data sources for analysis."""
        print("Loading raw data sources...")
        
        # Load truth table
        truth_df = pd.read_csv(self.truth_table_path)
        print(f"  ✓ Truth table: {len(truth_df)} entries, {truth_df['Cluster name'].nunique()} unique clusters")
        
        # Load prior locations
        prior_df = pd.read_csv(self.prior_locations_path)
        print(f"  ✓ Prior locations: {len(prior_df)} candidates, {prior_df['cluster'].nunique()} unique clusters")
        
        # Get image files
        image_files = {}
        for scale, image_dir in self.image_dirs.items():
            files = glob.glob(os.path.join(image_dir, '*.tif'))
            clusters = [os.path.basename(f).split('_')[0] for f in files]
            image_files[scale] = {
                'files': files,
                'clusters': clusters,
                'unique_clusters': list(set(clusters))
            }
            print(f"  ✓ {scale} images: {len(files)} files, {len(image_files[scale]['unique_clusters'])} unique clusters")
        
        self.raw_data = {
            'truth_table': truth_df,
            'prior_locations': prior_df,
            'image_files': image_files
        }
        
        return self.raw_data
    
    def analyze_overlaps(self):
        """Analyze overlaps between different data sources."""
        print("\nAnalyzing data overlaps...")
        
        # Get unique cluster sets
        truth_clusters = set(self.raw_data['truth_table']['Cluster name'].unique())
        prior_clusters = set(self.raw_data['prior_locations']['cluster'].unique())
        
        overlaps = {}
        
        for scale in ['2p2arcmin', '3p8arcmin']:
            image_clusters = set(self.raw_data['image_files'][scale]['unique_clusters'])
            
            overlaps[scale] = {
                'truth_image': len(truth_clusters.intersection(image_clusters)),
                'prior_image': len(prior_clusters.intersection(image_clusters)),
                'truth_only': truth_clusters - image_clusters,
                'image_only': image_clusters - truth_clusters,
                'prior_only_in_images': prior_clusters.intersection(image_clusters) - truth_clusters,
                'total_images': len(image_clusters)
            }
            
            print(f"  {scale}:")
            print(f"    Truth table ∩ Images: {overlaps[scale]['truth_image']} matches")
            print(f"    Prior locations ∩ Images: {overlaps[scale]['prior_image']} matches")
            print(f"    Images without truth: {len(overlaps[scale]['image_only'])}")
            print(f"    Images with only prior (no truth): {len(overlaps[scale]['prior_only_in_images'])}")
        
        # Cross-scale analysis
        clusters_2p2 = set(self.raw_data['image_files']['2p2arcmin']['unique_clusters'])
        clusters_3p8 = set(self.raw_data['image_files']['3p8arcmin']['unique_clusters'])
        
        overlaps['cross_scale'] = {
            '2p2_in_3p8': len(clusters_2p2.intersection(clusters_3p8)),
            'total_2p2': len(clusters_2p2),
            '3p8_only': clusters_3p8 - clusters_2p2
        }
        
        print(f"  Cross-scale:")
        print(f"    2.2 arcmin ∩ 3.8 arcmin: {overlaps['cross_scale']['2p2_in_3p8']} matches")
        print(f"    3.8 arcmin only: {len(overlaps['cross_scale']['3p8_only'])}")
        
        self.analysis_results['overlaps'] = overlaps
        return overlaps
    
    def create_bcg_coordinates(self, scale):
        """Create BCG coordinates for given scale."""
        print(f"\nGenerating BCG coordinates for {scale}...")
        
        truth_df = self.raw_data['truth_table']
        image_dir = self.image_dirs[scale]
        image_files = glob.glob(os.path.join(image_dir, '*.tif'))
        
        results = []
        processed = 0
        no_match = 0
        errors = 0
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            cluster_name = filename.split('_')[0]
            
            # Find matching entries in truth table
            matching_rows = truth_df[truth_df['Cluster name'] == cluster_name]
            
            if len(matching_rows) == 0:
                no_match += 1
                continue
            
            # Load image to get WCS
            try:
                pil_image = pillow.open(image_path)
                pil_image.seek(0)
                wcs = self.read_wcs(pil_image)
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
                
                processed += 1
                    
            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                errors += 1
                continue
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.bcg_coord_files[scale], index=False)
        
        stats = {
            'total_images': len(image_files),
            'processed': processed,
            'no_match': no_match,
            'errors': errors,
            'coordinate_entries': len(results_df)
        }
        
        print(f"  ✓ Processed {processed} images, {no_match} no match, {errors} errors")
        print(f"  ✓ Generated {len(results_df)} coordinate entries")
        print(f"  ✓ Saved to: {self.bcg_coord_files[scale]}")
        
        return results_df, stats
    
    def create_desprior_coordinates(self, scale):
        """Create DESprior candidate coordinates for given scale."""
        print(f"\nGenerating DESprior coordinates for {scale}...")
        
        prior_df = self.raw_data['prior_locations']
        image_dir = self.image_dirs[scale]
        image_files = glob.glob(os.path.join(image_dir, '*.tif'))
        
        results = []
        clusters_processed = 0
        candidates_processed = 0
        candidates_outside_image = 0
        no_match = 0
        errors = 0
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            cluster_name = filename.split('_')[0]
            
            # Find matching entries in prior locations
            matching_candidates = prior_df[prior_df['cluster'] == cluster_name]
            
            if len(matching_candidates) == 0:
                no_match += 1
                continue
            
            # Load image to get WCS
            try:
                pil_image = pillow.open(image_path)
                pil_image.seek(0)
                wcs = self.read_wcs(pil_image)
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
                print(f"    Error processing {filename}: {e}")
                errors += 1
                continue
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.desprior_coord_files[scale], index=False)
        
        candidates_inside = candidates_processed - candidates_outside_image
        
        stats = {
            'total_images': len(image_files),
            'clusters_processed': clusters_processed,
            'no_match': no_match,
            'errors': errors,
            'candidates_processed': candidates_processed,
            'candidates_inside': candidates_inside,
            'candidates_outside': candidates_outside_image,
            'inside_percentage': candidates_inside / candidates_processed * 100 if candidates_processed > 0 else 0
        }
        
        print(f"  ✓ Processed {clusters_processed} clusters, {no_match} no match, {errors} errors")
        print(f"  ✓ Generated {candidates_processed} candidate coordinates")
        print(f"  ✓ Inside image bounds: {candidates_inside} ({stats['inside_percentage']:.1f}%)")
        print(f"  ✓ Outside image bounds: {candidates_outside_image}")
        print(f"  ✓ Saved to: {self.desprior_coord_files[scale]}")
        
        return results_df, stats
    
    def create_clean_datasets(self):
        """Create clean, matched datasets for ML training."""
        print("\nCreating clean datasets for ML training...")
        
        for scale in ['2p2arcmin', '3p8arcmin']:
            print(f"\n  Processing {scale}:")
            
            # Load coordinate files
            bcg_df = pd.read_csv(self.bcg_coord_files[scale])
            desprior_df = pd.read_csv(self.desprior_coord_files[scale])
            
            # Get clusters that have both BCG truth data AND images
            bcg_clusters = set(bcg_df['Cluster name'].unique())
            desprior_clusters = set(desprior_df['cluster'].unique())
            
            # Clean BCG dataset - only clusters that have valid images and coordinates
            clean_bcg_df = bcg_df.copy()
            
            # Clean DESprior dataset - only inside image bounds and valid clusters
            clean_desprior_df = desprior_df[
                (desprior_df['is_inside_image'] == True) & 
                (desprior_df['cluster'].isin(bcg_clusters))  # Only clusters with BCG truth
            ].copy()
            
            # Additional filtering for pristine ML dataset
            # Remove any clusters with coordinate errors (outside reasonable bounds)
            valid_bcg_mask = (
                (clean_bcg_df['x'] >= 0) & (clean_bcg_df['x'] <= 512) &
                (clean_bcg_df['y'] >= 0) & (clean_bcg_df['y'] <= 512) &
                (~clean_bcg_df['x'].isna()) & (~clean_bcg_df['y'].isna())
            )
            clean_bcg_df = clean_bcg_df[valid_bcg_mask]
            
            # Remove DESprior candidates for clusters that don't have valid BCG coordinates
            final_valid_clusters = set(clean_bcg_df['Cluster name'].unique())
            clean_desprior_df = clean_desprior_df[
                clean_desprior_df['cluster'].isin(final_valid_clusters)
            ]
            
            # Save clean datasets
            clean_bcg_df.to_csv(self.clean_bcg_files[scale], index=False)
            clean_desprior_df.to_csv(self.clean_desprior_files[scale], index=False)
            
            # Statistics
            bcg_clusters_clean = len(clean_bcg_df['Cluster name'].unique())
            desprior_clusters_clean = len(clean_desprior_df['cluster'].unique())
            
            print(f"    Clean BCG entries: {len(clean_bcg_df)} ({bcg_clusters_clean} clusters)")
            print(f"    Clean DESprior entries: {len(clean_desprior_df)} ({desprior_clusters_clean} clusters)")
            print(f"    BCG clusters with DESprior candidates: {desprior_clusters_clean}")
            print(f"    ✓ Saved clean BCG to: {self.clean_bcg_files[scale]}")
            print(f"    ✓ Saved clean DESprior to: {self.clean_desprior_files[scale]}")
            
            # Store stats
            self.analysis_results[f'{scale}_clean'] = {
                'bcg_entries': len(clean_bcg_df),
                'bcg_clusters': bcg_clusters_clean,
                'desprior_entries': len(clean_desprior_df),
                'desprior_clusters': desprior_clusters_clean,
                'clusters_with_both': desprior_clusters_clean
            }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BCG DATASET ANALYSIS REPORT")
        print("="*80)
        
        # Raw data summary
        print("\n### RAW DATA SOURCES ###")
        truth_df = self.raw_data['truth_table']
        prior_df = self.raw_data['prior_locations']
        
        print(f"Truth Table: {len(truth_df):,} BCG entries across {truth_df['Cluster name'].nunique():,} unique clusters")
        print(f"Prior Locations: {len(prior_df):,} DESprior candidates across {prior_df['cluster'].nunique():,} unique clusters")
        
        for scale in ['2p2arcmin', '3p8arcmin']:
            image_data = self.raw_data['image_files'][scale]
            print(f"{scale.upper()} Images: {len(image_data['files']):,} image files ({len(image_data['unique_clusters']):,} unique clusters)")
        
        # Overlap analysis
        print("\n### OVERLAP ANALYSIS ###")
        overlaps = self.analysis_results['overlaps']
        
        for scale in ['2p2arcmin', '3p8arcmin']:
            scale_data = overlaps[scale]
            print(f"{scale.upper()}:")
            print(f"  Truth Table ∩ Images: {scale_data['truth_image']:,} matching clusters")
            print(f"  Prior Locations ∩ Images: {scale_data['prior_image']:,} matching clusters")
            print(f"  Images without truth: {len(scale_data['image_only']):,}")
            print(f"  Images with only prior: {len(scale_data['prior_only_in_images']):,}")
        
        cross_scale = overlaps['cross_scale']
        print(f"Cross-scale: {cross_scale['2p2_in_3p8']:,} clusters in both scales ({len(cross_scale['3p8_only']):,} only in 3.8)")
        
        # Coordinate generation results
        print("\n### COORDINATE FILES GENERATED ###")
        for scale in ['2p2arcmin', '3p8arcmin']:
            bcg_df = pd.read_csv(self.bcg_coord_files[scale])
            desprior_df = pd.read_csv(self.desprior_coord_files[scale])
            
            print(f"{scale.upper()}:")
            print(f"  BCG coordinates: {len(bcg_df):,} entries")
            print(f"  DESprior coordinates: {len(desprior_df):,} candidates")
            
            if len(desprior_df) > 0:
                inside_candidates = len(desprior_df[desprior_df['is_inside_image'] == True])
                outside_candidates = len(desprior_df) - inside_candidates
                inside_pct = inside_candidates / len(desprior_df) * 100
                print(f"    Inside image bounds: {inside_candidates:,} ({inside_pct:.1f}%)")
                print(f"    Outside image bounds: {outside_candidates:,}")
        
        # Clean dataset summary
        print("\n### CLEAN DATASETS FOR ML TRAINING ###")
        for scale in ['2p2arcmin', '3p8arcmin']:
            clean_stats = self.analysis_results[f'{scale}_clean']
            print(f"{scale.upper()}:")
            print(f"  Clean BCG entries: {clean_stats['bcg_entries']:,} ({clean_stats['bcg_clusters']:,} clusters)")
            print(f"  Clean DESprior entries: {clean_stats['desprior_entries']:,} ({clean_stats['desprior_clusters']:,} clusters)")
            print(f"  Clusters with both BCG truth & DESprior candidates: {clean_stats['clusters_with_both']:,}")
        
        # Data quality insights
        print("\n### DATA QUALITY INSIGHTS ###")
        
        # Calculate coverage percentages
        total_2p2_images = len(self.raw_data['image_files']['2p2arcmin']['files'])
        total_3p8_images = len(self.raw_data['image_files']['3p8arcmin']['files'])
        
        clean_2p2_clusters = self.analysis_results['2p2arcmin_clean']['bcg_clusters']
        clean_3p8_clusters = self.analysis_results['3p8arcmin_clean']['bcg_clusters']
        
        print(f"1. BCG Truth Coverage: {clean_2p2_clusters}/{total_2p2_images} (2.2) and {clean_3p8_clusters}/{total_3p8_images} (3.8) images have valid BCG coordinates")
        
        # DESprior coverage
        desprior_2p2 = self.analysis_results['2p2arcmin_clean']['desprior_clusters']
        desprior_3p8 = self.analysis_results['3p8arcmin_clean']['desprior_clusters']
        
        print(f"2. DESprior Coverage: {desprior_2p2}/{clean_2p2_clusters} (2.2) and {desprior_3p8}/{clean_3p8_clusters} (3.8) BCG clusters have DESprior candidates")
        
        print(f"3. Multiple BCGs: Some clusters have multiple BCG entries (expected for complex systems)")
        print(f"4. Coordinate Bounds: All clean datasets have coordinates within valid image bounds")
        print(f"5. ML Ready: Clean datasets contain only matched, validated entries for training")
        
        # File locations
        print("\n### GENERATED FILES ###")
        print("Raw coordinate files:")
        for scale in ['2p2arcmin', '3p8arcmin']:
            print(f"  {os.path.basename(self.bcg_coord_files[scale])}")
            print(f"  {os.path.basename(self.desprior_coord_files[scale])}")
        
        print("Clean ML-ready files:")
        for scale in ['2p2arcmin', '3p8arcmin']:
            print(f"  {os.path.basename(self.clean_bcg_files[scale])}")
            print(f"  {os.path.basename(self.clean_desprior_files[scale])}")
        
        print("\n" + "="*80)
        print("✅ Dataset creation and verification completed successfully!")
        print("✅ Clean datasets are ready for ML training pipeline.")
        print("="*80)
    
    def run_full_analysis(self, regenerate_coords=True):
        """Run complete dataset creation and analysis pipeline."""
        print("Starting comprehensive BCG dataset analysis...")
        
        # Load raw data
        self.load_raw_data()
        
        # Analyze overlaps
        self.analyze_overlaps()
        
        if regenerate_coords:
            # Generate coordinate files
            for scale in ['2p2arcmin', '3p8arcmin']:
                self.create_bcg_coordinates(scale)
                self.create_desprior_coordinates(scale)
        else:
            print("\nSkipping coordinate regeneration (using existing files)")
        
        # Create clean datasets
        self.create_clean_datasets()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        return self.analysis_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='BCG Dataset Creator and Verifier')
    parser.add_argument('--regenerate', action='store_true', 
                       help='Regenerate coordinate files (default: True)')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run analysis only, skip coordinate generation')
    parser.add_argument('--no-regenerate', action='store_true',
                       help='Skip coordinate regeneration, use existing files')
    
    args = parser.parse_args()
    
    # Determine whether to regenerate coordinates
    regenerate_coords = True
    if args.analysis_only or args.no_regenerate:
        regenerate_coords = False
    
    # Create and run analysis
    manager = BCGDatasetManager()
    results = manager.run_full_analysis(regenerate_coords=regenerate_coords)
    
    print(f"\n📊 Analysis complete! Check the generated files:")
    print(f"   Raw coordinates: bcgs_*_with_coordinates.csv, desprior_candidates_*_with_coordinates.csv")
    print(f"   Clean ML datasets: *_clean_matched.csv files")
    print(f"\n🚀 The clean datasets are ready for use in train.py, test.py, and enhanced_full_run.py")


if __name__ == "__main__":
    main()