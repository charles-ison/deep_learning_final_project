from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T

resample_rate = 16000

# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v0", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0",trust_remote_code=True)

input_audio = torch.randn(10080).cuda()
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
print(outputs)