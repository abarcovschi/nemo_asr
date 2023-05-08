from nemo.collections.asr.models import EncDecRNNTBPEModel
import torch

ckpt_path = "/workspace/nemo/nemo_experiments/Conformer-Transducer-BPE-Large_MyST_lr1_onlyFFlayersEnc_noamhold/2023-05-02_20-54-18/checkpoints/Conformer-Transducer-BPE-Large_MyST_lr1_onlyFFlayersEnc_noamhold--val_wer=0.1433-epoch=135.ckpt"
nemo_path = "/workspace/nemo/nemo_experiments/Conformer-Transducer-BPE-Large_MyST_lr1_onlyFFlayersEnc_noamhold/2023-05-02_20-54-18/checkpoints/Conformer-Transducer-BPE-Large_MyST_lr1_onlyFFlayersEnc_noamhold--val_wer=0.1433-epoch=135.nemo"

model = EncDecRNNTBPEModel.load_from_checkpoint(ckpt_path) 
checkpoint = torch.load(ckpt_path) 
model.load_state_dict(checkpoint['state_dict'], strict=True) 
model.save_to(nemo_path) 