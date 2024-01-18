# NeMo ASR Experiments

This project involves finetuning and extracting time alignments from Automatic Speech Recognition (ASR) models using the Nvidia NeMo framework.

# Tasks

## Finetuning NeMo Models

The following guide describes the steps required to finetune the [Conformer-Transducer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_transducer_large) models of different sizes (see list of finetuneable models [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html#english)).<br />

1. Create `train_manifest.json` and `test_manifest.json` files from a dataset using one of the `scripts/prepare_manifest_X.py` scripts depending on the structure of the dataset. For example, the [`scripts/prepare_manifest_myst.py`](https://github.com/abarcovschi/nemo_asr/blob/main/scripts/prepare_manifest_myst.py) script is used for the MyST dataset.
2. Create SPE+unigram tokenizer model from the train manifest file using [process_asr_text_tokenizer.py](https://github.com/abarcovschi/nemo_asr/blob/main/process_asr_text_tokenizer.py) script, called as follows:
```
python process_asr_text_tokenizer.py --manifest="/workspace/datasets/myst_w2v2_asr/train_manifest.json" --data_root=tokenizers/myst_w2v2_asr --vocab_size=1024 --tokenizer="spe" --spe_type="unigram" --log 
```
The tokenizer model will be saved to the `--data_root` folder location.<br />

**NOTE:** Check the .vocab file of the tokenizer subfolder. The decoder must get a vocab of size 1024, so if the vocab size is less, instead of "unigram" use "bpe" for the `--spe_type` to create a vocabulary of size 1024.

3. Download the non-finetuned base model in .nemo format from NGC using wget.
4. Then fine-tuning is run with a modified version of the original Conformer-Transducer BPE .yaml config file via the following command (example below is for XLarge model):
```
python3 nemo_asr/speech_to_text_rnnt_bpe_custom.py --config-path=configs --config-name=conformer_transducer_bpe_xlarge_custom1 name=<experiment_output_folder_name> model.train_ds.batch_size=4 model.validation_ds.batch_size=4 model.test_ds.batch_size=4 trainer.devices=[1,2,3] trainer.accelerator=gpu +init_from_nemo_model=non-finetuned/models/stt_en_conformer_transducer_xlarge.nemo ~model.encoder.stochastic_depth_drop_prob ~model.encoder.stochastic_depth_mode ~model.encoder.stochastic_depth_start_layer
```
The script will look for the config .yaml file located at `--config-path`/`--config-name`.yaml

## Generating Transcriptions (Inference)

1. If needed, to create a .nemo model from .ckpt (using the .ckpt that gave the lowest WER on the valid dataset), run: [`scripts/ckpt_to_nemo_asr.py`](https://github.com/abarcovschi/nemo_asr/blob/main/scripts/ckpt_to_nemo_asr.py). But no need to do this if setting the following lines in the finetuning .yaml config file under the `exp_manager->checkpoint_callback_params` section:
```yaml
save_best_model: true # saves best model as nemo file instead of the default last checkpoint
always_save_nemo: True # saves the checkpoints as nemo files instead of PTL checkpoints (last checkpoint is saved by default)
```
2.
```
python transcribe_speech.py model_path=/path/to/checkpoint.nemo dataset_manifest=/workspace/datasets/myst_w2v2_asr/test_manifest.json output_filename=/path/to/predictions.json batch_size=8 cuda=1 amp=True
```
The WER will be printed in the terminal. <br />
Different decoders may be used other than the default greedy decoder, e.g. ALSD variant of beam search with beam length of 5 and with timestamps:
```
python transcribe_speech.py
model_path=/path/to/checkpoint.nemo
dataset_manifest=/workspace/datasets/myst_w2v2_asr/test_manifest.json 
output_filename=/path/to/predictions.json
batch_size=8
cuda=1 
amp=True
decoder_type='rnnt'
rnnt_decoding.strategy='alsd'
rnnt_decoding.beam.beam_size=5
compute_timestamps=True
```

**OPTIONS:** <br />
- `decoder_type`: (`'rnnt'` -> will use rnnt_decoding object, `'ctc'` -> will use ctc_decoding object) will set `asr_model.cfg.decoding.model_type` to one of these.
- `rnnt_decoding.strategy`: (`'greedy'`, `'greedy_batch'` **(default)**, `'beam'`, `'alsd'`, `'tsd'`) will set `asr_model.cfg.decoding.strategy` to one of these.
- `rnnt_decoding.beam.beam_size`: will set `asr_model.cfg.decoding.beam.beam_size` to a new value.
- `rnnt_decoding.rnnt_timestamp_type`: (`'all'`, `'char'`, `'word'`) will set `asr_model.cfg.decoding.rnnt_timestamp_type` to one of these. <br />
	same options if using `ctc_decoding.ctc_timestamp_type`
- `compute_timestamps`: (False **(default)**, True) will set `asr_model.cfg.decoding.preserve_alignments` to `True` or `False`. <br />

**NOTES:**
- Any command line argument will be processed as a field of the `cfg` object in the script, e.g.  setting `rnnt_decoding.strategy='tsd'` will mean in the script it is accessed via `cfg.rnnt_decoding.strategy`.
- In the script, the `asr_model.cfg.decoding` property is set to the `cfg.rnnt_decoding` dict object if on the command line we use `decoder_type='rnnt'`, otherwise it is set to the `cfg.ctc_decoding` dict object if we use `decoder_type='ctc'` on the command line.
- If using `decoder_type='ctc'`, use `ctc_decoding` on the command line instead of `rnnt_decoding` for specifying further options.
- Generating time alignments is only possible with `rnnt_decoding.strategy='greedy'/'greedy_batch'` or `ctc_decoding.strategy='greedy '/'greedy_batch'`.

## Generating Time Alignments for Multiple Transcriptions

1. Create a JSON manifest file containing a list of audio files to transcribe. To create such a manifest file, take inspiration from [prepare_manifest_Wearable_Audio_Diarized.py](https://github.com/abarcovschi/nemo_asr/blob/main/scripts/prepare_manifest_Wearable_Audio_Diarized.py).

2. Run the script `transcribe_speech.py`:
Example: `python transcribe_speech.py model_path=/path/to/model/folder/checkpints/model.nemo dataset_manifest=/path/to/manifest_to_transcribe.json output_filename=/path/to/output_predictions.json batch_size=8 cuda=-1 amp=True decoder_type='rnnt' compute_timestamps=True rnnt_decoding.strategy='alsd' rnnt_decoding.beam.beam_size=5`<br />
**NOTES:**
  - To create char and word-level time alignments for the generated transcriptions, you must include the **`compute_timestamps=True`** flag.
  - `decoder_type` may be `'rnnt'` or `'ctc'`, depending on your model (will set `asr_model.cfg.decoding.model_type` to `rnnt` or `ctc`).
  - If using `'rnnt'`, use `'rnnt_decoding'` argument for setting further RNNT-specific decoding parameters.
  - If using `'ctc'`, use `'ctc_decoding'` argument for setting further CTC-specific decoding parameters.
  - `rnnt_decoding.strategy` may be `'greedy'`, `'greedy_batch'`, `'beam'`, `'alsd'` or `'tsd'` (will set `asr_model.cfg.decoding.strategy` to one of these - default='greedy_batch').
    - Same options if using `ctc_decoding.strategy`.
  - `rnnt_decoding.rnnt_timestamp_type` may be `'all'`, `'char'` or `'word'` (will set `asr_model.cfg.decoding.rnnt_timestamp_type` to one of these - default='all').
    - Same options if using `ctc_decoding.ctc_timestamp_type`.

 3. Run the script `timestamps_from_offset_to_secs.py --model_folder_path /path/to/model/folder/ --predictions_file_path /path/to/output_predictions.json --out_dir /path/to/new/folder/for/results` to create output `hypotheses.json` JSON files containing transcripts with word-level time alignments (if --compute_timestamps=True was used) for each audio file listed in the input manifest file used in step 2.<br />
 **NOTE:** the `--model_folder_path` argument is the folder in which the `hparams.yaml` file is located for the .nemo model used in step 2. The .nemo model is usually located in the /checkpoints subdirectory of this folder.
<br /><br />

**UPDATE:** The script `transcribe_speech.py` can only be used to create word-level time alignments for the **best** hypothesis from the ASR transcription process.<br />
To be able to create **multiple** JSON output files, one for **each** hypothesis rank outputted by the beam-search decoding process for the audio files, use `transcribe_speech_custom.py` script with the following command-line arguments:
- decoder_type='rnnt'
- compute_timestamps=True
- rnnt_decoding.strategy='beam'
- rnnt_decoding.preserve_alignments=True (**NOTE:** this param doesn't seem to actually influence time alignments).
- rnnt_decoding.beam.return_best_hypothesis=False
- [optionally, change the beam size (number of hypotheses outputted) with: rnnt_decoding.beam.beam_size=N]<br />
You can then run `timestamps_from_offset_to_secs.py` as before.<br /><br />

To return just the best hypotheses, you can run `transcribe_speech_custom.py` with `rnnt_decoding.beam.return_best_hypothesis=False`, and to optionally include time alignments information add `compute_timestamps=True`<br />
Then run `timestamps_from_offset_to_secs.py` as before to create a single `best_hypotheses.json` file.

### Output Formats

Each JSON line in the output file created by `transcribe_speech_custom.py` will be formatted as a Python Dict. An example output for a `hypothesis.json` output file **with** timestamps if `rnnt_decoding.beam.beam_size=3` was used, ran on a `manifest.json` file with 2 audio files, is the following:
```json
{"audio_filepath": "/workspace/datasets/LibriTTS_test/121/127105/121-127105-0011.wav", "duration": 5.78, "hypothesis1_pred_text": "she was the most agreable woman i've never known in her position she would have been worth of any whatever", "hypothesis1_timestamps_char": [{"char": ["she"], "start_offset": 7, "end_offset": 8}, {"char": ["was"], "start_offset": 11, "end_offset": 12}, {"char": ["the"], "start_offset": 16, "end_offset": 17}, {"char": ["most"], "start_offset": 19, "end_offset": 20}, {"char": ["a"], "start_offset": 25, "end_offset": 26}, {"char": ["g"], "start_offset": 28, "end_offset": 29}, {"char": ["re"], "start_offset": 30, "end_offset": 31}, {"char": ["a"], "start_offset": 33, "end_offset": 34}, {"char": ["ble"], "start_offset": 36, "end_offset": 37}, {"char": ["w"], "start_offset": 40, "end_offset": 41}, {"char": ["o", "m"], "start_offset": 44, "end_offset": 45}, {"char": ["an"], "start_offset": 45, "end_offset": 46}, {"char": ["i"], "start_offset": 47, "end_offset": 48}, {"char": ["'", "ve"], "start_offset": 50, "end_offset": 51}, {"char": ["never"], "start_offset": 54, "end_offset": 55}, {"char": ["know"], "start_offset": 60, "end_offset": 61}, {"char": ["n"], "start_offset": 64, "end_offset": 65}, {"char": ["in"], "start_offset": 68, "end_offset": 69}, {"char": ["her"], "start_offset": 72, "end_offset": 73}, {"char": ["po"], "start_offset": 76, "end_offset": 77}, {"char": ["s"], "start_offset": 79, "end_offset": 80}, {"char": ["it"], "start_offset": 81, "end_offset": 82}, {"char": ["ion"], "start_offset": 83, "end_offset": 84}, {"char": ["she"], "start_offset": 93, "end_offset": 94}, {"char": ["would"], "start_offset": 98, "end_offset": 99}, {"char": ["have"], "start_offset": 102, "end_offset": 103}, {"char": ["be", "en"], "start_offset": 105, "end_offset": 106}, {"char": ["wor"], "start_offset": 108, "end_offset": 109}, {"char": ["th"], "start_offset": 114, "end_offset": 115}, {"char": ["of"], "start_offset": 118, "end_offset": 119}, {"char": ["any"], "start_offset": 121, "end_offset": 122}, {"char": ["whatever"], "start_offset": 128, "end_offset": 129}], "hypothesis1_timestamps_word": [{"word": "she", "start_offset": 7, "end_offset": 11}, {"word": "was", "start_offset": 11, "end_offset": 16}, {"word": "the", "start_offset": 16, "end_offset": 19}, {"word": "most", "start_offset": 19, "end_offset": 25}, {"word": "agreable", "start_offset": 25, "end_offset": 40}, {"word": "woman", "start_offset": 40, "end_offset": 47}, {"word": "i've", "start_offset": 47, "end_offset": 54}, {"word": "never", "start_offset": 54, "end_offset": 60}, {"word": "known", "start_offset": 60, "end_offset": 68}, {"word": "in", "start_offset": 68, "end_offset": 72}, {"word": "her", "start_offset": 72, "end_offset": 76}, {"word": "position", "start_offset": 76, "end_offset": 93}, {"word": "she", "start_offset": 93, "end_offset": 98}, {"word": "would", "start_offset": 98, "end_offset": 102}, {"word": "have", "start_offset": 102, "end_offset": 105}, {"word": "been", "start_offset": 105, "end_offset": 108}, {"word": "worth", "start_offset": 108, "end_offset": 118}, {"word": "of", "start_offset": 118, "end_offset": 121}, {"word": "any", "start_offset": 121, "end_offset": 128}, {"word": "whatever", "start_offset": 128, "end_offset": 129}], "hypothesis2_pred_text": "she was the most agreable woman i've never known in her position she would have been worth of any whatever", "hypothesis2_timestamps_char": [{"char": ["she"], "start_offset": 7, "end_offset": 8}, {"char": ["was"], "start_offset": 11, "end_offset": 12}, {"char": ["the"], "start_offset": 16, "end_offset": 17}, {"char": ["most"], "start_offset": 19, "end_offset": 20}, {"char": ["a"], "start_offset": 25, "end_offset": 26}, {"char": ["g"], "start_offset": 28, "end_offset": 29}, {"char": ["re"], "start_offset": 30, "end_offset": 31}, {"char": ["a"], "start_offset": 33, "end_offset": 34}, {"char": ["ble"], "start_offset": 36, "end_offset": 37}, {"char": ["w"], "start_offset": 40, "end_offset": 41}, {"char": ["o", "m"], "start_offset": 44, "end_offset": 45}, {"char": ["an"], "start_offset": 45, "end_offset": 46}, {"char": ["i"], "start_offset": 47, "end_offset": 48}, {"char": ["'", "ve"], "start_offset": 50, "end_offset": 51}, {"char": ["never"], "start_offset": 54, "end_offset": 55}, {"char": ["know"], "start_offset": 60, "end_offset": 61}, {"char": ["n"], "start_offset": 64, "end_offset": 65}, {"char": ["in"], "start_offset": 68, "end_offset": 69}, {"char": ["her"], "start_offset": 72, "end_offset": 73}, {"char": ["po"], "start_offset": 76, "end_offset": 77}, {"char": ["s"], "start_offset": 80, "end_offset": 81}, {"char": ["it"], "start_offset": 81, "end_offset": 82}, {"char": ["ion"], "start_offset": 83, "end_offset": 84}, {"char": ["she"], "start_offset": 93, "end_offset": 94}, {"char": ["would"], "start_offset": 98, "end_offset": 99}, {"char": ["have"], "start_offset": 102, "end_offset": 103}, {"char": ["be", "en"], "start_offset": 105, "end_offset": 106}, {"char": ["wor"], "start_offset": 108, "end_offset": 109}, {"char": ["th"], "start_offset": 114, "end_offset": 115}, {"char": ["of"], "start_offset": 118, "end_offset": 119}, {"char": ["any"], "start_offset": 121, "end_offset": 122}, {"char": ["whatever"], "start_offset": 128, "end_offset": 129}], "hypothesis2_timestamps_word": [{"word": "she", "start_offset": 7, "end_offset": 11}, {"word": "was", "start_offset": 11, "end_offset": 16}, {"word": "the", "start_offset": 16, "end_offset": 19}, {"word": "most", "start_offset": 19, "end_offset": 25}, {"word": "agreable", "start_offset": 25, "end_offset": 40}, {"word": "woman", "start_offset": 40, "end_offset": 47}, {"word": "i've", "start_offset": 47, "end_offset": 54}, {"word": "never", "start_offset": 54, "end_offset": 60}, {"word": "known", "start_offset": 60, "end_offset": 68}, {"word": "in", "start_offset": 68, "end_offset": 72}, {"word": "her", "start_offset": 72, "end_offset": 76}, {"word": "position", "start_offset": 76, "end_offset": 93}, {"word": "she", "start_offset": 93, "end_offset": 98}, {"word": "would", "start_offset": 98, "end_offset": 102}, {"word": "have", "start_offset": 102, "end_offset": 105}, {"word": "been", "start_offset": 105, "end_offset": 108}, {"word": "worth", "start_offset": 108, "end_offset": 118}, {"word": "of", "start_offset": 118, "end_offset": 121}, {"word": "any", "start_offset": 121, "end_offset": 128}, {"word": "whatever", "start_offset": 128, "end_offset": 129}], "hypothesis3_pred_text": "she was the most agreable woman i've never known in her position she would have been worth of any whatever", "hypothesis3_timestamps_char": [{"char": ["she"], "start_offset": 7, "end_offset": 8}, {"char": ["was"], "start_offset": 11, "end_offset": 12}, {"char": ["the"], "start_offset": 16, "end_offset": 17}, {"char": ["most"], "start_offset": 19, "end_offset": 20}, {"char": ["a"], "start_offset": 25, "end_offset": 26}, {"char": ["g"], "start_offset": 28, "end_offset": 29}, {"char": ["re"], "start_offset": 30, "end_offset": 31}, {"char": ["a"], "start_offset": 33, "end_offset": 34}, {"char": ["ble"], "start_offset": 36, "end_offset": 37}, {"char": ["w"], "start_offset": 40, "end_offset": 41}, {"char": ["o", "m"], "start_offset": 44, "end_offset": 45}, {"char": ["an"], "start_offset": 45, "end_offset": 46}, {"char": ["i"], "start_offset": 47, "end_offset": 48}, {"char": ["'", "ve"], "start_offset": 50, "end_offset": 51}, {"char": ["never"], "start_offset": 54, "end_offset": 55}, {"char": ["know"], "start_offset": 60, "end_offset": 61}, {"char": ["n"], "start_offset": 64, "end_offset": 65}, {"char": ["in"], "start_offset": 68, "end_offset": 69}, {"char": ["her"], "start_offset": 72, "end_offset": 73}, {"char": ["po"], "start_offset": 76, "end_offset": 77}, {"char": ["s"], "start_offset": 79, "end_offset": 80}, {"char": ["it"], "start_offset": 81, "end_offset": 82}, {"char": ["ion"], "start_offset": 83, "end_offset": 84}, {"char": ["she"], "start_offset": 93, "end_offset": 94}, {"char": ["would"], "start_offset": 98, "end_offset": 99}, {"char": ["have"], "start_offset": 102, "end_offset": 103}, {"char": ["be", "en"], "start_offset": 105, "end_offset": 106}, {"char": ["wor"], "start_offset": 109, "end_offset": 110}, {"char": ["th"], "start_offset": 114, "end_offset": 115}, {"char": ["of"], "start_offset": 118, "end_offset": 119}, {"char": ["any"], "start_offset": 121, "end_offset": 122}, {"char": ["whatever"], "start_offset": 128, "end_offset": 129}], "hypothesis3_timestamps_word": [{"word": "she", "start_offset": 7, "end_offset": 11}, {"word": "was", "start_offset": 11, "end_offset": 16}, {"word": "the", "start_offset": 16, "end_offset": 19}, {"word": "most", "start_offset": 19, "end_offset": 25}, {"word": "agreable", "start_offset": 25, "end_offset": 40}, {"word": "woman", "start_offset": 40, "end_offset": 47}, {"word": "i've", "start_offset": 47, "end_offset": 54}, {"word": "never", "start_offset": 54, "end_offset": 60}, {"word": "known", "start_offset": 60, "end_offset": 68}, {"word": "in", "start_offset": 68, "end_offset": 72}, {"word": "her", "start_offset": 72, "end_offset": 76}, {"word": "position", "start_offset": 76, "end_offset": 93}, {"word": "she", "start_offset": 93, "end_offset": 98}, {"word": "would", "start_offset": 98, "end_offset": 102}, {"word": "have", "start_offset": 102, "end_offset": 105}, {"word": "been", "start_offset": 105, "end_offset": 109}, {"word": "worth", "start_offset": 109, "end_offset": 118}, {"word": "of", "start_offset": 118, "end_offset": 121}, {"word": "any", "start_offset": 121, "end_offset": 128}, {"word": "whatever", "start_offset": 128, "end_offset": 129}]}
{"audio_filepath": "/workspace/datasets/LibriTTS_test/121/127105/121-127105-0034.wav", "duration": 7.41, "hypothesis1_pred_text": "it sounded sounded and all the more so because of his main condition which was", "hypothesis1_timestamps_char": [{"char": ["it"], "start_offset": 8, "end_offset": 9}, {"char": ["sound"], "start_offset": 13, "end_offset": 14}, {"char": ["ed"], "start_offset": 19, "end_offset": 20}, {"char": ["sound"], "start_offset": 49, "end_offset": 50}, {"char": ["ed"], "start_offset": 55, "end_offset": 56}, {"char": ["and"], "start_offset": 87, "end_offset": 88}, {"char": ["all"], "start_offset": 96, "end_offset": 97}, {"char": ["the"], "start_offset": 101, "end_offset": 102}, {"char": ["more"], "start_offset": 105, "end_offset": 106}, {"char": ["so"], "start_offset": 111, "end_offset": 112}, {"char": ["because"], "start_offset": 117, "end_offset": 118}, {"char": ["of"], "start_offset": 121, "end_offset": 122}, {"char": ["his"], "start_offset": 125, "end_offset": 126}, {"char": ["ma"], "start_offset": 128, "end_offset": 129}, {"char": ["in"], "start_offset": 131, "end_offset": 132}, {"char": ["con"], "start_offset": 135, "end_offset": 136}, {"char": ["d"], "start_offset": 138, "end_offset": 139}, {"char": ["it"], "start_offset": 140, "end_offset": 141}, {"char": ["ion"], "start_offset": 143, "end_offset": 144}, {"char": ["which"], "start_offset": 155, "end_offset": 156}, {"char": ["was"], "start_offset": 168, "end_offset": 169}], "hypothesis1_timestamps_word": [{"word": "it", "start_offset": 8, "end_offset": 13}, {"word": "sounded", "start_offset": 13, "end_offset": 49}, {"word": "sounded", "start_offset": 49, "end_offset": 87}, {"word": "and", "start_offset": 87, "end_offset": 96}, {"word": "all", "start_offset": 96, "end_offset": 101}, {"word": "the", "start_offset": 101, "end_offset": 105}, {"word": "more", "start_offset": 105, "end_offset": 111}, {"word": "so", "start_offset": 111, "end_offset": 117}, {"word": "because", "start_offset": 117, "end_offset": 121}, {"word": "of", "start_offset": 121, "end_offset": 125}, {"word": "his", "start_offset": 125, "end_offset": 128}, {"word": "main", "start_offset": 128, "end_offset": 135}, {"word": "condition", "start_offset": 135, "end_offset": 155}, {"word": "which", "start_offset": 155, "end_offset": 168}, {"word": "was", "start_offset": 168, "end_offset": 169}], "hypothesis2_pred_text": "it sounded sounded and all the more so because of his main condition which was", "hypothesis2_timestamps_char": [{"char": ["it"], "start_offset": 8, "end_offset": 9}, {"char": ["sound"], "start_offset": 13, "end_offset": 14}, {"char": ["ed"], "start_offset": 19, "end_offset": 20}, {"char": ["sound"], "start_offset": 49, "end_offset": 50}, {"char": ["ed"], "start_offset": 55, "end_offset": 56}, {"char": ["and"], "start_offset": 87, "end_offset": 88}, {"char": ["all"], "start_offset": 96, "end_offset": 97}, {"char": ["the"], "start_offset": 101, "end_offset": 102}, {"char": ["more"], "start_offset": 105, "end_offset": 106}, {"char": ["so"], "start_offset": 111, "end_offset": 112}, {"char": ["because"], "start_offset": 117, "end_offset": 118}, {"char": ["of"], "start_offset": 121, "end_offset": 122}, {"char": ["his"], "start_offset": 125, "end_offset": 126}, {"char": ["ma"], "start_offset": 128, "end_offset": 129}, {"char": ["in"], "start_offset": 131, "end_offset": 132}, {"char": ["con"], "start_offset": 135, "end_offset": 136}, {"char": ["d"], "start_offset": 138, "end_offset": 139}, {"char": ["it"], "start_offset": 140, "end_offset": 141}, {"char": ["ion"], "start_offset": 143, "end_offset": 144}, {"char": ["which"], "start_offset": 158, "end_offset": 159}, {"char": ["was"], "start_offset": 168, "end_offset": 169}], "hypothesis2_timestamps_word": [{"word": "it", "start_offset": 8, "end_offset": 13}, {"word": "sounded", "start_offset": 13, "end_offset": 49}, {"word": "sounded", "start_offset": 49, "end_offset": 87}, {"word": "and", "start_offset": 87, "end_offset": 96}, {"word": "all", "start_offset": 96, "end_offset": 101}, {"word": "the", "start_offset": 101, "end_offset": 105}, {"word": "more", "start_offset": 105, "end_offset": 111}, {"word": "so", "start_offset": 111, "end_offset": 117}, {"word": "because", "start_offset": 117, "end_offset": 121}, {"word": "of", "start_offset": 121, "end_offset": 125}, {"word": "his", "start_offset": 125, "end_offset": 128}, {"word": "main", "start_offset": 128, "end_offset": 135}, {"word": "condition", "start_offset": 135, "end_offset": 158}, {"word": "which", "start_offset": 158, "end_offset": 168}, {"word": "was", "start_offset": 168, "end_offset": 169}], "hypothesis3_pred_text": "it sounded sounded and all the more so because of his main condition which was", "hypothesis3_timestamps_char": [{"char": ["it"], "start_offset": 8, "end_offset": 9}, {"char": ["sound"], "start_offset": 13, "end_offset": 14}, {"char": ["ed"], "start_offset": 19, "end_offset": 20}, {"char": ["sound"], "start_offset": 49, "end_offset": 50}, {"char": ["ed"], "start_offset": 55, "end_offset": 56}, {"char": ["and"], "start_offset": 87, "end_offset": 88}, {"char": ["all"], "start_offset": 96, "end_offset": 97}, {"char": ["the"], "start_offset": 101, "end_offset": 102}, {"char": ["more"], "start_offset": 105, "end_offset": 106}, {"char": ["so"], "start_offset": 111, "end_offset": 112}, {"char": ["because"], "start_offset": 117, "end_offset": 118}, {"char": ["of"], "start_offset": 121, "end_offset": 122}, {"char": ["his"], "start_offset": 125, "end_offset": 126}, {"char": ["ma"], "start_offset": 128, "end_offset": 129}, {"char": ["in"], "start_offset": 131, "end_offset": 132}, {"char": ["con"], "start_offset": 135, "end_offset": 136}, {"char": ["d"], "start_offset": 138, "end_offset": 139}, {"char": ["it"], "start_offset": 140, "end_offset": 141}, {"char": ["ion"], "start_offset": 143, "end_offset": 144}, {"char": ["which"], "start_offset": 156, "end_offset": 157}, {"char": ["was"], "start_offset": 168, "end_offset": 169}], "hypothesis3_timestamps_word": [{"word": "it", "start_offset": 8, "end_offset": 13}, {"word": "sounded", "start_offset": 13, "end_offset": 49}, {"word": "sounded", "start_offset": 49, "end_offset": 87}, {"word": "and", "start_offset": 87, "end_offset": 96}, {"word": "all", "start_offset": 96, "end_offset": 101}, {"word": "the", "start_offset": 101, "end_offset": 105}, {"word": "more", "start_offset": 105, "end_offset": 111}, {"word": "so", "start_offset": 111, "end_offset": 117}, {"word": "because", "start_offset": 117, "end_offset": 121}, {"word": "of", "start_offset": 121, "end_offset": 125}, {"word": "his", "start_offset": 125, "end_offset": 128}, {"word": "main", "start_offset": 128, "end_offset": 135}, {"word": "condition", "start_offset": 135, "end_offset": 156}, {"word": "which", "start_offset": 156, "end_offset": 168}, {"word": "was", "start_offset": 168, "end_offset": 169}]}
```
Each JSON line in the new output files created by `timestamps_from_offset_to_secs.py` will be formatted as a Python Dict. An example output for a `hypotheses1_of_3.json` file (which will contain the best hypotheses with word-level timestamps formatted in seconds for each audio file) created from the previous `hypothesis.json`, is the following:
```json
{"wav_path": "/workspace/datasets/LibriTTS_test/121/127105/121-127105-0011.wav", "id": "127105/121-127105-0011", "pred_txt": "she was the most agreable woman i've never known in her position she would have been worth of any whatever", "timestamps_word": [{"word": "she", "start_time": 0.28, "end_time": 0.44}, {"word": "was", "start_time": 0.44, "end_time": 0.64}, {"word": "the", "start_time": 0.64, "end_time": 0.76}, {"word": "most", "start_time": 0.76, "end_time": 1.0}, {"word": "agreable", "start_time": 1.0, "end_time": 1.6}, {"word": "woman", "start_time": 1.6, "end_time": 1.8800000000000001}, {"word": "i've", "start_time": 1.8800000000000001, "end_time": 2.16}, {"word": "never", "start_time": 2.16, "end_time": 2.4}, {"word": "known", "start_time": 2.4, "end_time": 2.72}, {"word": "in", "start_time": 2.72, "end_time": 2.88}, {"word": "her", "start_time": 2.88, "end_time": 3.04}, {"word": "position", "start_time": 3.04, "end_time": 3.72}, {"word": "she", "start_time": 3.72, "end_time": 3.92}, {"word": "would", "start_time": 3.92, "end_time": 4.08}, {"word": "have", "start_time": 4.08, "end_time": 4.2}, {"word": "been", "start_time": 4.2, "end_time": 4.32}, {"word": "worth", "start_time": 4.32, "end_time": 4.72}, {"word": "of", "start_time": 4.72, "end_time": 4.84}, {"word": "any", "start_time": 4.84, "end_time": 5.12}, {"word": "whatever", "start_time": 5.12, "end_time": 5.16}]}
{"wav_path": "/workspace/datasets/LibriTTS_test/121/127105/121-127105-0034.wav", "id": "127105/121-127105-0034", "pred_txt": "it sounded sounded and all the more so because of his main condition which was", "timestamps_word": [{"word": "it", "start_time": 0.32, "end_time": 0.52}, {"word": "sounded", "start_time": 0.52, "end_time": 1.96}, {"word": "sounded", "start_time": 1.96, "end_time": 3.48}, {"word": "and", "start_time": 3.48, "end_time": 3.84}, {"word": "all", "start_time": 3.84, "end_time": 4.04}, {"word": "the", "start_time": 4.04, "end_time": 4.2}, {"word": "more", "start_time": 4.2, "end_time": 4.44}, {"word": "so", "start_time": 4.44, "end_time": 4.68}, {"word": "because", "start_time": 4.68, "end_time": 4.84}, {"word": "of", "start_time": 4.84, "end_time": 5.0}, {"word": "his", "start_time": 5.0, "end_time": 5.12}, {"word": "main", "start_time": 5.12, "end_time": 5.4}, {"word": "condition", "start_time": 5.4, "end_time": 6.2}, {"word": "which", "start_time": 6.2, "end_time": 6.72}, {"word": "was", "start_time": 6.72, "end_time": 6.76}]}
```

This format is consistent with the format in the output files created by the other ASR alignment scripts described in the [C3Imaging/speech-augmentation](https://github.com/C3Imaging/speech-augmentation#time-aligned-predictions-and-forced-alignment) project.<br />

If there was only one hypothesis per audio in the `hypothesis.json` file outputted by `transcribe_speech_custom.py`, then `timestamps_from_offset_to_secs.py` will output a single `best_hypotheses.json` file.
