import os
import json
import string
import librosa

"""Build train and valid manifest.json files for ASR training in the NeMo framework for the MyST dataset, originally called 'myst_w2v2_asr.zip'
NOTE: this formatting of the cleaned MyST dataset is for ASR training only!
NOTE: this script also works for pfe_16khz (PFStar child audio dataset cleaned and preprocessed for ASR).
"""

# Function to build a manifest
def build_manifest(src_data_path, manifest_path):
    with open(manifest_path, 'w') as fout:
        for dirpath, subdirs, filenames in os.walk(src_data_path, topdown=False):
            if not subdirs:
                # get set of unique <transcript, audio> pairs
                ids = set(list(map(lambda x: x[:-4], filenames)))
                for id in ids:
                    # get transcript
                    with open(os.path.join(dirpath, id+'.txt'),'r') as f:
                        transcript = f.readline().strip()
                        transcript = " ".join(transcript.translate(str.maketrans('', '', string.punctuation)).split())  #removes all the punctuations and extra whitespace
                    # Write the metadata to the manifest
                    metadata = {
                        "audio_filepath": os.path.join(dirpath, id+'.wav'),
                        "duration": librosa.core.get_duration(filename=os.path.join(dirpath, id+'.wav')),
                        "text": transcript
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')

                
if __name__ == "__main__":
    # Building Manifests
    data_dir_root = '/workspace/datasets/LibriSpeech/test-clean'
    print("******")
    # creating train manifest
    train_manifest_path = os.path.join(data_dir_root, 'train_manifest.json') # to be created
    # if manifest does not yet exist
    if not os.path.isfile(train_manifest_path):
        train_path = os.path.join(data_dir_root, 'train')
        build_manifest(train_path, train_manifest_path)
        print("Training manifest created.")
        
    # creating test manifest
    test_manifest_path = os.path.join(data_dir_root, 'test_manifest.json') # to be created
    # if manifest does not yet exist
    if not os.path.isfile(test_manifest_path):
        test_path = os.path.join(data_dir_root, 'test')
        build_manifest(test_path, test_manifest_path)
        print("Test manifest created.")
    print("***Done***") 
    # Manifest filepaths, use these to override using Hydra
    TRAIN_MANIFEST = train_manifest_path
    TEST_MANIFEST = test_manifest_path