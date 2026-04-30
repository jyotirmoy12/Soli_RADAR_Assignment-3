# Soli Radar Gesture Recognition — Adversarial Training

## Requirements
pip install torch h5py scipy numpy matplotlib


## Run Training
1. Place `SoliData.zip`
2. Open `soli_gan.ipynb`
3. Run **Cells 1 to 13** top to bottom

## Run Testing
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
C = GestureClassifier(n_classes=11, n_subjects=10).to(device)
ckpt = torch.load("full_fold0_best.pt", map_location=device)
C.load_state_dict(ckpt['classifier'])
C.eval()
# x: tensor of shape (1, 1, 32, 16, 16)
pred = C.predict(x.to(device)).argmax(dim=1).item()


## Pretrained Model
[ full_fold1_best.pt]
