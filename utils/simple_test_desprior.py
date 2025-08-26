#!/usr/bin/env python3
"""
Simple test script for DESprior candidate system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from data.candidate_dataset_bcgs import create_desprior_candidate_dataset_from_files
        print("  ✓ Successfully imported create_desprior_candidate_dataset_from_files")
        
        from data.data_read_bcgs import create_bcg_datasets, prepare_bcg_dataframe
        print("  ✓ Successfully imported data reading functions")
        
        import pandas as pd
        print("  ✓ pandas imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_csv_files():
    """Test that CSV files exist and are readable"""
    print("\nTesting CSV files...")
    
    files_to_check = [
        '/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/bcgs_2p2arcmin_with_coordinates.csv',
        '/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/desprior_candidates_2p2arcmin_with_coordinates.csv'
    ]
    
    try:
        import pandas as pd
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"  ✓ {os.path.basename(file_path)}: {len(df)} rows")
            else:
                print(f"  ✗ File not found: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading CSV files: {e}")
        return False

def test_small_dataset():
    """Test creating a very small dataset"""
    print("\nTesting small dataset creation...")
    
    try:
        from data.data_read_bcgs import prepare_bcg_dataframe
        
        # Test just the dataframe preparation with heavy filtering to get small dataset
        df = prepare_bcg_dataframe(
            '/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/data/bcgs_2p2arcmin_with_coordinates.csv',
            z_range=(0.5, 0.6),  # Very narrow range
            delta_mstar_z_range=(-2.0, -1.5)  # Very narrow range
        )
        
        print(f"  ✓ Created filtered dataframe with {len(df)} entries")
        
        if len(df) > 0:
            print(f"  ✓ Sample columns: {list(df.columns)}")
            print(f"  ✓ Redshift range: {df['Cluster z'].min():.3f} - {df['Cluster z'].max():.3f}")
            print(f"  ✓ delta_mstar_z range: {df['delta_mstar_z'].min():.3f} - {df['delta_mstar_z'].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run simple tests"""
    print("Simple DESprior System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_csv_files,
        test_small_dataset
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    print("\n" + "="*40)
    if success:
        print("✓ Simple tests passed! DESprior system is ready.")
    else:
        print("✗ Some tests failed.")
    
    sys.exit(0 if success else 1)