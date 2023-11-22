# NeMo ASR Experiments

This project involves finetuning and extracting time alignments from Automatic Speech Recognition (ASR) models using the Nvidia NeMo framework.

# Tasks

## Generating Time Alignments for Transcriptions

0. Create a JSON manifest file containing a list of audio files to transcribe.

1. First, run the script `transcribe_speech.py`:

Example: `python transcribe_speech.py model_path=/path/to/model.nemo dataset_manifest=/path/to/to_transcribe.json output_filename=/path/to/predictions.json batch_size=8 cuda=-1 amp=True decoder_type='rnnt' compute_timestamps=True rnnt_decoding.strategy='alsd' rnnt_decoding.beam.beam_size=5`

**NOTES:**
- To create char and word-level time alignments for the generated transcriptions, you must include the **`compute_timestamps=True`** flag.
- `decoder_type` may be `'rnnt'` or `'ctc'`, depending on your model (will set `asr_model.cfg.decoding.model_type` to `rnnt` or `ctc`).
- If using `'rnnt'`, use `'rnnt_decoding'` argument for setting further RNNT-specific decoding parameters.
- If using `'ctc'`, use `'ctc_decoding'` argument for setting further CTC-specific decoding parameters.
- `rnnt_decoding.strategy` may be `'greedy'`, `'greedy_batch'`, `'alsd'` or `'tsd'` (will set `asr_model.cfg.decoding.strategy` to one of these).
- `rnnt_decoding.rnnt_timestamp_type` may be `'all'`, `'char'` or `'word'` (will set `asr_model.cfg.decoding.rnnt_timestamp_type` to one of these).
  - Same options if using `ctc_decoding.ctc_timestamp_type`.

 2. Secondly, run the script `scripts/timestamps_from_offset_to_secs.py` to create word-level time alignments for each audio file listed in the manifest file used in step 1.
<br />Each line in the new files will have 3 values, separated by commas: word, start time (sec), end time (sec). 


