import os
import json
import librosa
import pandas as pd

"""Build train and valid manifest.json files for ASR training in the NeMo framework for the CMU dataset, originally called 'cmu_hf.zip'
NOTE: this formatting of the cleaned CMU dataset is for ASR training only!
"""

# Function to build a manifest
def build_manifest(src_data_path, manifest_path):
    with open(manifest_path, 'w') as fout:
        df = pd.read_csv(os.path.join(src_data_path, '..', 'metadata.csv'))
        DS_TYPE = src_data_path.split('/')[-1]
        for i, row in df.iterrows():
            filepath = row['file_name'].strip()
            transcript = row['sentence'].strip()
            if filepath.split('/')[0] in DS_TYPE:
                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": os.path.join(src_data_path, filepath.split('/')[-1]),
                    "duration": librosa.core.get_duration(filename=os.path.join(src_data_path, filepath.split('/')[-1])),
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')

                
if __name__ == "__main__":
    # Building Manifests
    data_dir_root = '/workspace/datasets/cmu_hf'
    print("******")
    # creating test manifest
    test_manifest_path = os.path.join(data_dir_root, 'test_manifest.json') # to be created
    # if manifest does not yet exist
    if not os.path.isfile(test_manifest_path):
        test_path = os.path.join(data_dir_root, 'test')
        build_manifest(test_path, test_manifest_path)
        print("Test manifest created.")
    # creating train manifest
    train_manifest_path = os.path.join(data_dir_root, 'train_manifest.json') # to be created
    # if manifest does not yet exist
    if not os.path.isfile(train_manifest_path):
        train_path = os.path.join(data_dir_root, 'train')
        build_manifest(train_path, train_manifest_path)
        print("Training manifest created.")
    print("***Done***")
    # Manifest filepaths, use these to override using Hydra
    TRAIN_MANIFEST = train_manifest_path
    TEST_MANIFEST = test_manifest_path