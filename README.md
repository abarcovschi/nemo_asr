# NeMo ASR Experiments

This project involves finetuning and extracting time alignments from Automatic Speech Recognition (ASR) models using the Nvidia NeMo framework.

# Tasks

## Generating Time Alignments for Transcriptions

1. Create a JSON manifest file containing a list of audio files to transcribe. To create such a manifest file, take inspiration from [prepare_manifest_Wearable_Audio_Diarized.py](https://github.com/abarcovschi/nemo_asr/blob/main/scripts/prepare_manifest_Wearable_Audio_Diarized.py).

2. Run the script `transcribe_speech.py`:
Example: `python transcribe_speech.py model_path=/path/to/model.nemo dataset_manifest=/path/to/manifest_to_transcribe.json output_filename=/path/to/output_predictions.json batch_size=8 cuda=-1 amp=True decoder_type='rnnt' compute_timestamps=True rnnt_decoding.strategy='alsd' rnnt_decoding.beam.beam_size=5`<br />
**NOTES:**
  - To create char and word-level time alignments for the generated transcriptions, you must include the **`compute_timestamps=True`** flag.
  - `decoder_type` may be `'rnnt'` or `'ctc'`, depending on your model (will set `asr_model.cfg.decoding.model_type` to `rnnt` or `ctc`).
  - If using `'rnnt'`, use `'rnnt_decoding'` argument for setting further RNNT-specific decoding parameters.
  - If using `'ctc'`, use `'ctc_decoding'` argument for setting further CTC-specific decoding parameters.
  - `rnnt_decoding.strategy` may be `'greedy'`, `'greedy_batch'`, `'beam'`, `'alsd'` or `'tsd'` (will set `asr_model.cfg.decoding.strategy` to one of these).
    - Same options if using `ctc_decoding.strategy`.
  - `rnnt_decoding.rnnt_timestamp_type` may be `'all'`, `'char'` or `'word'` (will set `asr_model.cfg.decoding.rnnt_timestamp_type` to one of these).
    - Same options if using `ctc_decoding.ctc_timestamp_type`.

 3. Run the script `scripts/timestamps_from_offset_to_secs.py --predictions_file_path /path/to/output_predictions.json` to create output 'alignments.txt' CSV files containing transcripts with word-level time alignments for each audio file listed in the input manifest file used in step 2.
<br />Each line in the new output files will have 3 values, separated by commas: word, start time (sec), end time (sec). This format is consistent with the 'alignments.txt' files created by the other alignment scripts described in the [C3Imaging/speech-augmentation](https://github.com/C3Imaging/speech-augmentation#time-aligned-predictions-and-forced-alignment) project.
<br /><br />

**UPDATE:** The script `transcribe_speech.py` can only be used to create word-level time alignments for the **best** hypothesis from the ASR transcription process. To be able to create **multiple** alignments.txt output files, one for **each** hypothesis outputted by the beam-search decoding process, use `transcribe_speech_custom.py` script with the following command-line arguments:
- decoder_type='rnnt'
- compute_timestamps=True
- rnnt_decoding.strategy='beam'
- rnnt_decoding.beam.return_best_hypothesis=False
- [optionally, change the beam size (number of hypotheses outputted) with: rnnt_decoding.beam.beam_size=N]<br />
You can then run `scripts/timestamps_from_offset_to_secs.py` as before.


