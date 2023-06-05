from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T

resample_rate = 24000

# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

input_audio = torch.randn(10080).cuda()
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)

all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
last_hidden_state = outputs.last_hidden_state

# print(all_layer_hidden_states)      # [13 layer, Time steps, 768 feature_dim]
# print(all_layer_hidden_states[-1])
# print(last_hidden_state)            # [1 layer, Time steps, 768 feature_dim]

print(outputs)
