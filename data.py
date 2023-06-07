import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random

# path setup
whole_data_dir = 'minibabyslakh/'
# print(os.listdir(data_dir))

class TrackDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.track_list = self._load_track_list()
        self.data_list = self._load_data_list()

    def _load_track_list(self):
        track_list = []
        # Load track information from the directory
        # [{track: track path, bass_audio: [bass audio path], residuals: [redidual audio path]}, ...]
        # Assuming track, bass_audio, and residuals are the required keys in the data dictionary
        for trackname in os.listdir(self.data_dir):
            track_path = os.path.join(self.data_dir, trackname)
            if os.path.isdir(track_path):
                bass_files = []
                residuals_files = []

                audio_path = os.path.join(track_path, "bass")
                for filename in os.listdir(audio_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(audio_path, filename)
                        if 'bass' in filename:
                            bass_files.append(file_path)
                        elif 'residuals' in filename:
                            residuals_files.append(file_path)
                if len(bass_files) != len(residuals_files):
                    raise ValueError("The number of bass and residuals files are not equal in track {}".format(track_path))
                if len(bass_files) != 1 or len(residuals_files) != 1:
                    raise ValueError("The number of bass and residuals files are not 1 in track {}, check the data structure".format(track_path))
                track_info = {
                    'track': track_path,
                    'bass': bass_files[0],
                    'residuals': residuals_files[0]
                }
                track_list.append(track_info)
        return track_list
    
    def _load_data_list(self):
        data_list = []
        # Load data into torch tensors form
        # [{track: track path, bass_data: tuple(tensor, sample_rate), residuals_data: tuple(tensor, sample_rate)}, ...]
        for track_info in self.track_list:
            # Load and process the audio files into tensors
            bass_file = track_info['bass']
            residuals_file = track_info['residuals']
            
            bass_wav_sr = torchaudio.load(bass_file)
            residuals_wav_sr = torchaudio.load(residuals_file)

            
            data_item = {
                'track': track_info['track'],
                'bass_data': bass_wav_sr,
                'residuals_data': residuals_wav_sr
            }
            data_list.append(data_item)
            
        return data_list
    
    def __len__(self):
        return len(self.track_list)

    def __getitem__(self, index):
        # a easier API to get the torth_data, AKA in default: self[index] is as same as self.data_list[index]
        return self.data_list[index]

    def _single_sample(self, index, window_size):
        # takes in a duration and returns a random sample of the audio with that duration.
        def get_duration(wav_tensor, sample_rate):
            # the tuple_data should be in a pair of tuple(tensor, sample_rate)
            return wav_tensor.shape[1]/sample_rate
        single_sample_pair = {}
        for audio_type in "bass_data", "residuals_data":
            wave_tnesor, sample_rate = self[index][audio_type]
            audio_length = get_duration(wave_tnesor, sample_rate)
            # print(audio_length)

            if window_size > audio_length:
                raise ValueError("window_size should be smaller than the audio length, the length of {} is {}".format(self[index]["track"], audio_length))
            max_start_time = audio_length - window_size
            start_time = random.uniform(0, max_start_time)

            start_sample = int(start_time * sample_rate)
            end_sampoe = start_sample + int(window_size * sample_rate)
            # Extract the desired window from the waveform
            sampled_waveform = wave_tnesor[:, start_sample:end_sampoe]
            # print(sampled_waveform.shape)
            single_sample_pair[audio_type] = (sampled_waveform, sample_rate) # always return a tuple of (tensor, sample_rate), sample_rete for future use
        
        return single_sample_pair
    
    def get_batch(self, batch_size, window_size):
        # return a batch of these random samples from function "_single_sample".
        # batch_data = []
        # for i in range(batch_size):
        # TODO
        return

train_dataset = TrackDataset(os.path.join(whole_data_dir, 'train'))
# print(train_dataset.track_list)
# print(train_dataset.data_list)
print(train_dataset._single_sample(0, 20))
# print(train_dataset[0])
# print(len(train_dataset))