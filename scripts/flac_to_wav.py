import os
from pydub import AudioSegment

for dirpath, _, filenames in os.walk('/workspace/datasets/LibriSpeech', topdown=False):
        # get list of speech files and corresponding transcripts from a single folder
        speech_files = []
        # loop through all files found
        for filename in filenames:
            if filename.endswith('.flac'):
                speech_files.append(os.path.join(dirpath, filename))

        if speech_files:
            for speech_file in speech_files:
                audio = AudioSegment.from_file(speech_file, "flac")
                audio.export(speech_file.replace("flac", "wav"), format="wav")
                os.remove(speech_file)