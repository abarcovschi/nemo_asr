import os
import json
import librosa

"""
Build manifest.json file for transcribing a diarization tool's output.
(all diarization tools used in the speech-augmentation pipeline output a common format, so this script will work for each tool).
Diarization options described here: https://github.com/C3Imaging/speech-augmentation#speaker-diarization
"""


def build_manifest(src_data_path, manifest_path):
    """
    Builds a manifest JSON file, which will list the wav files that need transcribing.
    For each wav file, add a line in the following format: {"audio_filepath": /path/to/audio.wav, "duration": time in secs}
    """
    with open(manifest_path, 'w') as fout:
        for dirpath, _, filenames in os.walk(src_data_path, topdown=False):
            for filename in filenames:
                if filename.endswith(".wav"):
                    # Write the metadata to the manifest.
                    metadata = {
                        "audio_filepath": os.path.join(dirpath, filename),
                        "duration": librosa.core.get_duration(filename=os.path.join(dirpath, filename)),
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')

                
if __name__ == "__main__":
    data_dir_root = '/workspace/datasets/LibriTTS_test_whisper'
    manifest_path = os.path.join(data_dir_root, 'manifest.json') # to be created.
    # if manifest does not yet exist.
    if not os.path.isfile(manifest_path):
        build_manifest(data_dir_root, manifest_path)
        print(f"{manifest_path} manifest file created.")
