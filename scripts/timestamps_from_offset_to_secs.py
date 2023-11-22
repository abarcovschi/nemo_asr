# convert from word-level timestamp offset values, returned in the json file after running transcribe_speech.py with compute_timestamps=True on a Conformer model, to seconds values.

import argparse
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


def main():
    # get the seconds duration of a timestep at output of the Conformer
    time_stride = 4 * cfg.get("cfg/preprocessor/window_stride")

    # load the transcripts file and loop through the predictions
    with open(args.predictions_file_path, 'r') as fin:
        for line in fin:
            item = json.loads(line)

            logging.info(f"starting to process {item['audio_filepath']}")
            audio_folder = '/'.join(item['audio_filepath'].split('/')[:-1])
            wav_name = item['audio_filepath'].split('/')[-1].split(".wav")[0]

            # create NEMO_ALIGNS_ output subfolder in the same folder as where multiple audiofiles for a particular speaker reside.
            cur_root_dir = os.path.join(audio_folder, out_dir)
            if not os.path.exists(cur_root_dir): 
                os.makedirs(cur_root_dir, exist_ok=True)
                logging.info(f"{cur_root_dir} folder created.")

            # create output subfolder for a single audio file in the particular speaker folder.
            cur_out_dir = os.path.join(cur_root_dir, wav_name)
            if not os.path.exists(cur_out_dir): 
                os.makedirs(cur_out_dir, exist_ok=True)
                logging.info(f"{cur_out_dir} folder created.")

            word_timestamps = item['timestamps_word']

            # write the timestamps for that audio file into a txt file in the cur_out_dir subfolder
            with open(os.path.join(cur_out_dir, 'alignments.txt'), 'w') as f:
                f.write("word_label,start_time,stop_time\n") # time is in seconds
                for stamp in word_timestamps:
                    start = stamp['start_offset'] * time_stride
                    stop = stamp['end_offset'] * time_stride
                    word = stamp['word']
                    # for each word detected, save to file the {label, start time, stop time} as a CSV line
                    f.write(f"{word},{start:.2f},{stop:.2f}\n")
            logging.info(f"finished processing {item['audio_filepath']}")


if __name__ == "__main__":
    # parse program arguments
    parser = argparse.ArgumentParser(
        description="Convert from word-level timestamp offset values, returned in the json file after running transcribe_speech.py with compute_timestamps=True on a Conformer model, to seconds values and save to file word-by-word.")
    parser.add_argument("--model_folder_path", type=str, default=None, required=True,
                        help="Path to the folder that contains the trained nemo model, which is stored in the 'checkpoints/' subfolder.")
    parser.add_argument("--predictions_file_path", type=str, default=None, required=True,
                        help="Path to the file containing timestamped predictions outputted by the model after running 'transcribe_speech.py' on it.")
    parser.add_argument("--out_folder_name", type=str, default='a_nemo_model_alignments',
                help="Name of the output folder, useful to differentiate runs.")
    
    global args, cfg
    args = parser.parse_args()
    # create config object from yaml file.
    cfg = Config(os.path.join(args.model_folder_path, "hparams.yaml"))
    # setup folder structure variables
    global out_dir, NEMO_ALIGNS_PREFIX
    WHISPER_ALIGNS_PREFIX = "NEMO_ALIGNS_"
    out_dir = WHISPER_ALIGNS_PREFIX + args.out_folder_name # the output folder to be created in folders where there are audio files

    # setup logging to both console and logfile
    setup_logging(os.path.join('/'.join(args.predictions_file_path.split('/')[:-1]), args.predictions_file_path.split('/')[-1].split(".json")[0]), 'time_alignment_conversion.log', console=True)

    # start timing how long it takes to run script
    tic = time.perf_counter()

    # log the command that started the script
    logging.info(f"Started script via: python {' '.join(sys.argv)}")
    main()

    toc = time.perf_counter()
    logging.info(f"Finished processing in {time.strftime('%H:%M:%Ss', time.gmtime(toc - tic))}")
    
