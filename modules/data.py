import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random

class TrackDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.track_list = self._load_track_list()
        self.window_size = 10 # default window size is 10 seconds
        self.sample_rate = 24000 # default sample rate is 24000

        # self.data_list = self._load_data_list() # load all the data as initialize the dataset, turn this off for now since we have get_item function to do this

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

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
    
    # not in use for now
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
        track = self.track_list[index]
        bass_file = track['bass']
        residuals_file = track['residuals']

        bass_wav_sr = torchaudio.load(bass_file)
        residuals_wav_sr = torchaudio.load(residuals_file)
        bass_length = get_duration(bass_wav_sr[0], bass_wav_sr[1])
        residuals_length = get_duration(residuals_wav_sr[0], residuals_wav_sr[1])
        if bass_length != residuals_length:
            raise ValueError("The length of bass and residuals are not equal in track {}".format(track['track']))
        else:
            audio_length = bass_length
        if self.window_size > audio_length:
            raise ValueError("window_size should be smaller than the audio length, the length of {} is {}".format(track['track'], audio_length))
        max_start_time = audio_length - self.window_size
        start_time = random.uniform(0, max_start_time)

        items = []
        for wav_sr in bass_wav_sr, residuals_wav_sr:
            wav, sr = wav_sr
            
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                wav = resampler(wav)
                sr = self.sample_rate

            start_sample = int(start_time * sr)
            end_sample = start_sample + int(self.window_size * sr)
            # Extract the desired window from the waveform
            sampled_waveform = wav[:, start_sample:end_sample]
            items.append(sampled_waveform)
        
        residual_audio = items[1]
        target_audio = items[0]

        residual_audio = residual_audio.squeeze() if residual_audio.shape[0] == 1 else residual_audio.mean(0)
        target_audio = target_audio.squeeze() if target_audio.shape[0] == 1 else target_audio.mean(0)

        return residual_audio, target_audio

# a help function to get the audio duration of a wav tensor
def get_duration(wav_tensor, sample_rate):
    if wav_tensor.dim() == 1:
        return wav_tensor.shape[0]/sample_rate
    return wav_tensor.shape[1]/sample_rate


def main():
    # methods validation
    # initialize the dataset
    whole_data_dir = 'data/mini/'
    train_dataset = TrackDataset(os.path.join(whole_data_dir, 'train'))
    # track list (directory)
    print("track list: \n", train_dataset.track_list)
    # data size:
    print("data size: ", len(train_dataset))
    # How to set the window size and sample rate (These two parameters are already set by default, so no need to set them again if that's what you want)
    train_dataset.set_window_size(10)
    train_dataset.set_sample_rate(24000)
    # get a single sample
    print("single sample (residual tensor + bass tensor): \n", train_dataset[0])
    print("the shape of residuals is: ", train_dataset[0][0].shape)
    print("the shape of bass is: ", train_dataset[0][1].shape)
    # check if the length matches the window size
    print("if the length matches the window size:")
    print("window size: ", train_dataset.window_size)
    print("length of the sample: ", get_duration(train_dataset[0][0], train_dataset.sample_rate))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, 5, shuffle=True)
    for batch in train_loader:
        print(batch)
        print(batch[0].shape)
        print(batch[1].shape)
        
if __name__=="__main__":
    main()