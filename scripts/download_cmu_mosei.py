import os
import sys
from mmsdk import mmdatasdk

# Directory to store the dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CMU_MOSEI')
os.makedirs(DATA_DIR, exist_ok=True)

# Define feature files
csd_files = {
    'glove_vectors': 'CMU_MOSEI_TimestampedWordVectors.csd',
    'COVAREP': 'CMU_MOSEI_COVAREP.csd',
    'OpenFace_2': 'CMU_MOSEI_VisualOpenFace2.csd',
    'FACET 4.2': 'CMU_MOSEI_VisualFacet42.csd',
    'All Labels': 'CMU_MOSEI_Labels.csd'
}

def main():
    print(f"Preparing to download CMU-MOSEI features to: {DATA_DIR}")
    
    try:
        # Create recipe dictionary
        recipe = {}
        for key, filename in csd_files.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                print(f"File already exists: {filename}")
                recipe[key] = file_path
            else:
                print(f"Need to download: {filename}")
                recipe[key] = f"http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/{key.lower()}/{filename}"
        
        # Download and load all files
        print("\nStarting download and load features...")
        dataset = mmdatasdk.mmdataset(recipe)
        print("Dataset loaded successfully!")
        
        # Print available computational sequence keys
        print("\nAvailable computational sequence keys:")
        keys = list(dataset.computational_sequences.keys())
        print(keys)
        
        # Use correct alignment key
        align_key = 'All Labels'
        if align_key in dataset.computational_sequences:
            print(f"\nUsing '{align_key}' for alignment...")
            try:
                dataset.align(align_key)
                print("Alignment completed!")
                
                # Print alignment statistics
                print("\nAlignment statistics:")
                for key in dataset.computational_sequences.keys():
                    data = dataset.computational_sequences[key].data
                    print(f"{key}: {len(data)} segments")
                
                print("\nCMU-MOSEI dataset is ready!")
                
            except Exception as e:
                print(f"\nError during alignment: {str(e)}")
                sys.exit(1)
        else:
            print(f"\nError: Alignment key '{align_key}' not found. Available keys: {keys}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 