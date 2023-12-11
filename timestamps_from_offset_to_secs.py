"""
Convert from word-level timestamp offset values, returned in the json file after running 'transcribe_speech.py' 
with 'compute_timestamps=True' command on a Conformer model to seconds values
and save output in the same format of time alignment output as with wav2vec2 and Whisper models from the following scripts:
https://github.com/C3Imaging/speech-augmentation/blob/main/wav2vec2_forced_alignment_libri.py
https://github.com/C3Imaging/speech-augmentation/blob/main/whisper_time_alignment.py
"""

import argparse
from itertools import groupby
import yaml
import os
import json
import logging
import time
import sys


class Config(object):  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of nested elements, e.g. cfg.get_config("meta/dataset_name")"""
    
    def __init__(self, config_path):
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())
    
    def get(self, path=None, default=None):
        """Parses the string, e.g. 'experiment/training/batch_size' by splitting it into a list and recursively accessing the nested sub-dictionaries."""
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default


def setup_logging(arg_folder, filename, console=False, filemode='w+'):
    """Set up logging to a logfile and optionally to the console also, if console param is True."""
    if not os.path.exists(arg_folder): os.makedirs(arg_folder, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # track INFO logging events (default was WARNING)
    root_logger.handlers = []  # clear handlers
    root_logger.addHandler(logging.FileHandler(os.path.join(arg_folder, filename), filemode))  # handler to log to file
    root_logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))  # log level and message

    if console:
        root_logger.addHandler(logging.StreamHandler(sys.stdout))  # handler to log to console
        root_logger.handlers[1].setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))


def main(args):
    # create config object from yaml file.
    cfg = Config(os.path.join(args.model_folder_path, "hparams.yaml"))
    # get the seconds duration of a timestep at output of the Conformer
    time_stride = 4 * cfg.get("cfg/preprocessor/window_stride")

    # load the transcripts file and loop through the predictions
    with open(args.predictions_file_path, 'r') as fin:
        for line in fin:
            # load a line from the predictions.json file as a dict.
            item = json.loads(line)

            logging.info(f"starting to process {item['audio_filepath']}")

            # get all keys from the item dict.
            keys = list(item.keys())

            if "hypothesis" in ' '.join(keys):
                # all beam search hypotheses were returned in the predictions.json file.

                # leave only "hypothesisX_xxx" keys in the dict.
                keys.pop(0) # remove "audio_filepath"
                keys.pop(0) # remove "duration"

                # group keys into separate lists according to their hypothesis number.
                each_word = sorted([x.split('_') for x in keys])

                # group by the hypothesis number.
                grouped = [list(value) for key, value in groupby(each_word, lambda x: x[0])]

                hypothesis_keys = []
                for group in grouped:
                    temp = []
                    for i in range(len(group)):
                        temp.append("_".join(group[i]))
                    hypothesis_keys.append(temp)

                for i in range(len(hypothesis_keys)):
                    # NOTE:
                    # hypothesis_keys[i][0] = 'hypothesis{i}_pred_text'
                    # hypothesis_keys[i][1] = 'hypothesis{i}_timestamps_char'
                    # hypothesis_keys[i][2] = 'hypothesis{i}_timestamps_word'

                    # open a file for storing the i hypotheses for all audio files.
                    with open(os.path.join(args.out_dir, f"hypotheses{i+1}_of_{len(hypothesis_keys)}.json"), 'a') as fout:
                        # item_out is a JSON line to write to output file for hypotheses i.
                        item_out = dict()

                        # create unique id of audio sample by including leaf folder in the id.
                        temp = item['audio_filepath'].split('/')[-2:] # [0] = subfolder, [1] = ____.wav
                        temp[-1] = temp[-1].split('.wav')[0] # remove '.wav'
                        id = '/'.join(temp)

                        item_out['wav_path'] = item['audio_filepath']
                        item_out['id'] = id
                        item_out['pred_txt'] = item[hypothesis_keys[i][0]]

                        vals = list()
                        # word-level timestamps for hypothesis i for audio file of this line.
                        word_timestamps = item[hypothesis_keys[i][2]]

                        for stamp in word_timestamps:
                            word_dict = dict()

                            start = stamp['start_offset'] * time_stride
                            stop = stamp['end_offset'] * time_stride
                            word = stamp['word']

                            word_dict['word'] = word
                            word_dict['start_time'] = start
                            word_dict['end_time'] = stop

                            vals.append(word_dict)

                        item_out['timestamps_word'] = vals
                        fout.write(json.dumps(item_out) + "\n")

            else:
                # just the best hypothesis was returned in the predictions.json file.
                with open(os.path.join(args.out_dir, 'best_hypotheses.json'), 'a') as fout:
                    item_out = dict()

                    # create unique id of audio sample by including leaf folder in the id.
                    temp = item['audio_filepath'].split('/')[-2:] # [0] = subfolder, [1] = ____.wav
                    temp[-1] = temp[-1].split('.wav')[0] # remove '.wav'
                    id = '/'.join(temp)

                    item_out['wav_path'] = item['audio_filepath']
                    item_out['id'] = id
                    item_out['pred_txt'] = item['pred_text']

                    vals = list()
                    word_timestamps = item['timestamps_word']

                    for stamp in word_timestamps:
                        word_dict = dict()

                        start = stamp['start_offset'] * time_stride
                        stop = stamp['end_offset'] * time_stride
                        word = stamp['word']

                        word_dict['word'] = word
                        word_dict['start_time'] = start
                        word_dict['end_time'] = stop

                        vals.append(word_dict)

                    item_out['timestamps_word'] = vals
                    fout.write(json.dumps(item_out) + "\n")

            logging.info(f"finished processing {item['audio_filepath']}")


if __name__ == "__main__":
    # parse program arguments
    parser = argparse.ArgumentParser(
        description="Convert from word-level timestamp offset values, returned in the resultant JSON file after running 'transcribe_speech_custom.py' with timestamps enabled on a Conformer model, to seconds values and save each hypothesis as a separate JSON file.")
    parser.add_argument("--model_folder_path", type=str, default=None, required=True,
                        help="Path to the folder that contains the trained .nemo model, which is stored in the 'checkpoints/' subfolder of this specified folder.")
    parser.add_argument("--predictions_file_path", type=str, default=None, required=True,
                        help="Path to the JSON file containing timestamped transcript predictions outputted by the NeMo ASR model after running 'transcribe_speech_custom.py' on it.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a new output folder to create, where results will be saved.")
    
    args = parser.parse_args()

    # setup logging to both console and logfile
    setup_logging(args.out_dir, 'time_alignment_conversion.log', console=True, filemode='w')

    # start timing how long it takes to run script
    tic = time.perf_counter()

    # log the command that started the script
    logging.info(f"Started script via: python {' '.join(sys.argv)}")

    main(args)

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")