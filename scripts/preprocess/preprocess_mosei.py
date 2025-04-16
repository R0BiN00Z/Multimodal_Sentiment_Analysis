import os
import numpy as np
import torch
from mmsdk import mmdatasdk
from tqdm import tqdm

class MOSEIPreprocessor:
    def __init__(self, data_dir):
        """
        Initialize the MOSEI preprocessor
        
        Args:
            data_dir (str): Directory containing the CMU-MOSEI dataset
        """
        self.data_dir = data_dir
        self.aligned_dir = os.path.join(data_dir, "aligned")
        os.makedirs(self.aligned_dir, exist_ok=True)
        
        # Define feature files
        self.csd_files = {
            'glove_vectors': 'CMU_MOSEI_TimestampedWordVectors.csd',
            'COVAREP': 'CMU_MOSEI_COVAREP.csd',
            'OpenFace_2': 'CMU_MOSEI_VisualOpenFace2.csd',
            'FACET 4.2': 'CMU_MOSEI_VisualFacet42.csd',
            'All Labels': 'CMU_MOSEI_Labels.csd'
        }
        
    def load_dataset(self):
        """Load the CMU-MOSEI dataset"""
        print("Loading CMU-MOSEI dataset...")
        
        # Create recipe dictionary
        recipe = {}
        for key, filename in self.csd_files.items():
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                recipe[key] = file_path
            else:
                raise FileNotFoundError(f"Missing file: {filename}")
        
        # Load dataset
        dataset = mmdatasdk.mmdataset(recipe)
        print("Dataset loaded successfully!")
        return dataset
    
    def align_features(self, dataset):
        """Align features using 'All Labels' as the alignment key"""
        print("Aligning features...")
        
        try:
            # Align features
            dataset.align('All Labels', collapse_functions=[np.mean])
            print("Features aligned successfully!")
            return dataset
        except Exception as e:
            print(f"Error during alignment: {str(e)}")
            raise
    
    def extract_features(self, dataset):
        """Extract features from the aligned dataset"""
        print("Extracting features...")
        
        features = {
            'text': [],
            'audio': [],
            'visual': [],
            'labels': []
        }
        
        # Get computational sequences
        text_seq = dataset.computational_sequences['glove_vectors']
        audio_seq = dataset.computational_sequences['COVAREP']
        visual_seq = dataset.computational_sequences['OpenFace_2']
        label_seq = dataset.computational_sequences['All Labels']
        
        # Extract features for each video
        for video_id in tqdm(dataset.computational_sequences['All Labels'].keys()):
            try:
                # Extract text features
                text_data = text_seq[video_id]['features']
                text_features = np.mean(text_data, axis=0)
                
                # Extract audio features
                audio_data = audio_seq[video_id]['features']
                audio_features = np.mean(audio_data, axis=0)
                
                # Extract visual features
                visual_data = visual_seq[video_id]['features']
                visual_features = np.mean(visual_data, axis=0)
                
                # Extract labels
                label_data = label_seq[video_id]['features']
                labels = np.mean(label_data, axis=0)
                
                # Store features
                features['text'].append(text_features)
                features['audio'].append(audio_features)
                features['visual'].append(visual_features)
                features['labels'].append(labels)
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def save_features(self, features, output_dir):
        """Save extracted features to disk"""
        print("Saving features...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each feature type
        for key, data in features.items():
            output_path = os.path.join(output_dir, f"{key}_features.npy")
            np.save(output_path, data)
            print(f"Saved {key} features to {output_path}")
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        try:
            # Load dataset
            dataset = self.load_dataset()
            
            # Align features
            aligned_dataset = self.align_features(dataset)
            
            # Extract features
            features = self.extract_features(aligned_dataset)
            
            # Save features
            self.save_features(features, self.aligned_dir)
            
            print("Preprocessing completed successfully!")
            
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise

def main():
    # Set data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'CMU_MOSEI')
    
    # Initialize preprocessor
    preprocessor = MOSEIPreprocessor(data_dir)
    
    # Run preprocessing
    preprocessor.preprocess()

if __name__ == "__main__":
    main() 