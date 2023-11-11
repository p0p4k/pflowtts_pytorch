from pflow.hifigan.meldataset import mel_spectrogram
import torch

audio = torch.randn(2,1, 1000)
mels = mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
print(mels.shape)