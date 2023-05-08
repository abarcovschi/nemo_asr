# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_rnnt_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

"""
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def freeze_layers(module, l_name, all_except_l_name=False):
    """Freeze specified layers of module whose parameter name contains 'l_name' substring."""

    # if all_except_l_name=True, only train the specified layers, freezing all other layers.
    # normalisation layers are not frozen.
    if all_except_l_name:
        for name, param in module.named_parameters():
            # freeze every layer of module that is not 'l_name' and not a normalisation layer.
            if param.requires_grad and l_name not in name and 'norm' not in name:
                param.requires_grad = False
    else:
        if "norm" in l_name:
            logging.warning("WARNING:: Will not freeze normalisation layers!!!")
        for name, param in module.named_parameters():
            # freeze the specified 'l_name' layers of the module
            if param.requires_grad and l_name in name and 'norm' not in name:
                param.requires_grad = False


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    
    # import a self-supervised pre-trained-only model
    # import nemo.collections.asr as nemo_asr
    # ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(model_name='ssl_en_conformer_large')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # # Initialize the weights of the model from another model, if provided via config
    # asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    
    # define model
    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)


    # load ssl checkpoint
    # asr_model.load_state_dict(ssl_model.state_dict(), strict=False)
    # del ssl_model

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # freeze layers
    layers_to_freeze = [0,1,2,3,4,5,6,7,8]
    # for i in layers_to_freeze:
        # freeze_layers(module=asr_model.encoder, l_name=f'layers.{i}')
    freeze_layers(module=asr_model.encoder, l_name='feed_forward', all_except_l_name=True)
    
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer(devices=1, accelerator='gpu')
        if asr_model.prepare_test(trainer):
            logging.warning("test freezes after training on line 'trainer.test(asr_model)'")
            # trainer.test(asr_model)
            pass


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
