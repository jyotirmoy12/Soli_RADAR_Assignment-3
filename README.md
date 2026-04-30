
# Soli Radar Gesture Recognition — Adversarial Training
## Requirements
pip install torch h5py scipy numpy matplotlib
## Run Training
1. Upload SoliData.zip 
2. Open soli_gan.ipynb in Google Colab
3. Run cells 1–13 top to bottom
## Run Testing Only
```python
import torch
ckpt = torch.load("full_fold0_best.pt")
C.load_state_dict(ckpt['classifier'])
C.eval()
