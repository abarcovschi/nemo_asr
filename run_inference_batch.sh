#!/bin/bash

declare -a arr_dataset=(
	"/workspace/datasets/LibriSpeech/test-clean/test_manifest.json"
	"/workspace/datasets/LibriSpeech/test-other/test_manifest.json")

declare -a arr_model=(
	"/workspace/nemo/non-finetuned/models/stt_en_conformer_transducer_small.nemo"
	"/workspace/nemo/non-finetuned/models/stt_en_conformer_transducer_medium.nemo"
	"/workspace/nemo/non-finetuned/models/stt_en_conformer_transducer_large.nemo"
    "/workspace/nemo/non-finetuned/models/stt_en_conformer_transducer_xlarge.nemo"
    "/workspace/nemo/non-finetuned/models/stt_en_conformer_transducer_xxlarge.nemo")

arr_model_len=${#arr_model[@]}
arr_dataset_len=${#arr_dataset[@]}

# use for loop to read all values and indexes
ix=0
for (( i=0; i<${arr_model_len}; i++ ));
do
	for (( j=0; j<${arr_dataset_len}; j++ ));
	do
        echo "${ix}) ${arr_model[$i]} ${arr_dataset[$j]}" >> run_inference_batch.log
        echo "${arr_model[$i]} ${arr_dataset[$j]}"  >> whisper_out4.log
        python3 transcribe_speech.py model_path="${arr_model[$i]}" dataset_manifest="${arr_dataset[$j]}" output_filename="temp.json" | grep -e 'wer' >> run_inference_batch.log
        ((ix++))
	done
done
