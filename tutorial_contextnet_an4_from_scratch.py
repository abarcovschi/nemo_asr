import os
import wget
import tarfile 
import subprocess 
import glob
import json
import librosa
from omegaconf import OmegaConf, open_dict

# create configs/ folder
if not os.path.exists("configs/"): os.makedirs("configs")
# download config YAML file for Nemo model
url = 'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/contextnet_rnnt/contextnet_rnnt.yaml'
if not os.path.exists("configs/contextnet_rnnt.yaml"): wget.download(url, out="configs/")

# create scripts/ folder
if not os.path.exists("scripts/"): os.makedirs("scripts")

# download training dataset
url = 'https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/dataset_processing/process_an4_data.py'
if not os.path.exists("scripts/process_an4_data.py"): wget.download(url, out="scripts/")

data_dir = "datasets/"

if not os.path.exists(data_dir):
  os.makedirs(data_dir)

## AN4 training dataset
# Download the dataset. This will take a few moments...
print("******")
if not os.path.exists(data_dir + 'an4_sphere.tar.gz'):
    an4_url = 'https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz'  # for the original source, please visit http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz 
    an4_path = wget.download(an4_url, data_dir)
    print(f"Dataset downloaded at: {an4_path}")
else:
    print("Tarfile already exists.")
    an4_path = data_dir + 'an4_sphere.tar.gz'

if not os.path.exists(data_dir + '/an4/'):
    # Untar and convert .sph to .wav (using sox)
    tar = tarfile.open(an4_path)
    tar.extractall(path=data_dir)

    print("Converting .sph to .wav...")
    sph_list = glob.glob(data_dir + 'an4/**/*.sph', recursive=True)
    for sph_path in sph_list:
        wav_path = sph_path[:-4] + '.wav'
        cmd = ["sox", sph_path, wav_path]
        subprocess.run(cmd)

print("Finished conversion.\n******")

# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path):
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin:
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find('(')-1].lower()
                transcript = transcript.replace('<s>', '').replace('</s>', '')
                transcript = transcript.strip()

                file_id = line[line.find('(')+1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    data_dir, wav_path,
                    file_id[file_id.find('-')+1 : file_id.rfind('-')],
                    file_id + '.wav')

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')

# Building Manifests
print("******")
train_transcripts = os.path.join(data_dir, 'an4/etc/an4_train.transcription')
train_manifest = os.path.join(data_dir, 'an4/train_manifest.json')
if not os.path.isfile(train_manifest):
    build_manifest(train_transcripts, train_manifest, 'an4/wav/an4_clstk')
    print("Training manifest created.")

test_transcripts = os.path.join(data_dir, 'an4/etc/an4_test.transcription')
test_manifest = os.path.join(data_dir, 'an4/test_manifest.json')
if not os.path.isfile(test_manifest):
    build_manifest(test_transcripts, test_manifest, 'an4/wav/an4test_clstk')
    print("Test manifest created.")
print("***Done***") 
# Manifest filepaths
TRAIN_MANIFEST = train_manifest
TEST_MANIFEST = test_manifest

# ------------ build tokenizer ----------------
if not os.path.exists("scripts/process_asr_text_tokenizer.py"):
  wget.download("https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/tokenizers/process_asr_text_tokenizer.py", out="scripts/")
  
VOCAB_SIZE = 32  # can be any value above 29
TOKENIZER_TYPE = "spe"  # can be wpe or spe
SPE_TYPE = "unigram"  # can be bpe or unigram

# ------------------------------------------------------------------- #
os.system('rm -r tokenizers/')

if not os.path.exists("tokenizers"):
  os.makedirs("tokenizers")
  
subprocess.call(["python", "scripts/process_asr_text_tokenizer.py", "--manifest", TRAIN_MANIFEST, "--data_root", "tokenizers", "--tokenizer", TOKENIZER_TYPE, "--spe_type", SPE_TYPE, "--no_lower_case", "--log", "--vocab_size", str(VOCAB_SIZE)])

# Tokenizer path
if TOKENIZER_TYPE == 'spe':
  TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_{SPE_TYPE}_v{VOCAB_SIZE}")
  TOKENIZER_TYPE_CFG = "bpe"
else:
  TOKENIZER = os.path.join("tokenizers", f"tokenizer_wpe_v{VOCAB_SIZE}")
  TOKENIZER_TYPE_CFG = "wpe"

config = OmegaConf.load("configs/contextnet_rnnt.yaml")

config.model.encoder.jasper = config.model.encoder.jasper[:5]
config.model.encoder.jasper[-1].filters = '${model.model_defaults.enc_hidden}'

# print out the train and validation configs to know what needs to be changed
print(OmegaConf.to_yaml(config.model.train_ds))

config.model.train_ds.manifest_filepath = TRAIN_MANIFEST
config.model.validation_ds.manifest_filepath = TEST_MANIFEST
config.model.test_ds.manifest_filepath = TEST_MANIFEST

print(OmegaConf.to_yaml(config.model.tokenizer))

config.model.tokenizer.dir = TOKENIZER
config.model.tokenizer.type = TOKENIZER_TYPE_CFG

print(OmegaConf.to_yaml(config.model.optim))

# Finally, let's remove logging of samples and the warmup since the dataset is small (similar to CTC models)
config.model.log_prediction = False
config.model.optim.sched.warmup_steps = None

print(OmegaConf.to_yaml(config.model.spec_augment))

config.model.spec_augment.freq_masks = 0
config.model.spec_augment.time_masks = 0

# Two lines to enable the fused batch step
config.model.joint.fuse_loss_wer = True
config.model.joint.fused_batch_size = 16  # this can be any value (preferably less than model.*_ds.batch_size)

# We will also reduce the hidden dimension of the joint and the prediction networks to preserve some memory
config.model.model_defaults.pred_hidden = 64
config.model.model_defaults.joint_hidden = 64

# Use just 128 filters across the model to speed up training and reduce parameter count
config.model.model_defaults.filters = 128

import torch
from pytorch_lightning import Trainer

if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'gpu'

EPOCHS = 50

# --------------- CREATE TRANSDUCER MODEL ----------------

# Initialize a Trainer for the Transducer model
trainer = Trainer(devices=1, accelerator=accelerator, max_epochs=EPOCHS,
                  enable_checkpointing=False, logger=False,
                  log_every_n_steps=5, check_val_every_n_epoch=10)

# Import the Transducer Model
import nemo.collections.asr as nemo_asr

# Build the model
model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)

model.summarize()

# Prepare NeMo's Experiment manager to handle checkpoint saving and logging for us
from nemo.utils import exp_manager

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/',
    name=f"Transducer-Model",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

exp_config = OmegaConf.structured(exp_config)

logdir = exp_manager.exp_manager(trainer, exp_config)

# Train the model
trainer.fit(model)
trainer.test(model)
a = 1