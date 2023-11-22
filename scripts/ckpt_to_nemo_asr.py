from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch

ckpt_path = "/workspace/projects/nemo_asr/nemo_experiments/NEW_Conformer-Transducer-BPE-Small_MyST_onlyFFlayersEnc_noam/2023-05-30_20-56-27/checkpoints/NEW_Conformer-Transducer-BPE-Small_MyST_onlyFFlayersEnc_noam--val_wer=0.1694-epoch=516.ckpt"
nemo_path = "/workspace/projects/nemo_asr/nemo_experiments/NEW_Conformer-Transducer-BPE-Small_MyST_onlyFFlayersEnc_noam/2023-05-30_20-56-27/checkpoints/NEW_Conformer-Transducer-BPE-Small_MyST_onlyFFlayersEnc_noam--val_wer=0.1694-epoch=516.nemo"

model = EncDecRNNTBPEModel.load_from_checkpoint(ckpt_path) 
checkpoint = torch.load(ckpt_path) 
model.load_state_dict(checkpoint['state_dict'], strict=True) 
model.save_to(nemo_path) 