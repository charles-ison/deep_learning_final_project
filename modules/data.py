import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

# TODO: @shawn finish TrackDataset class.
class TrackDataset(Dataset):
    """
    @Shawn This class should do a couple things:

    Create a list of track files.
    Create a sample_track(window_size, idx) function which returns a random slice of a track
        specifically sample_track should return residuals_files, bass_audio as lists.
    Create a get_batch(batch_size, window_size) that gets a batch of samples.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.track_list = self._load_track_list()
        # load all the waves
        # self.data = self._load_audio()
        # self.sample_rate = 

    def _load_track_list(self):
        track_list = []
        # Load track information from the directory
        # Assuming trackname, bass_audio, and residuals are the required keys in the data dictionary
        for trackname in os.listdir(self.data_dir):
            track_path = os.path.join(self.data_dir, trackname)
            if os.path.isdir(track_path):
                bass_audio_files = []
                residuals_files = []
                
                for filename in os.listdir(track_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(track_path, filename)
                        if 'bass' in filename:
                            bass_audio_files.append(file_path)
                        elif 'residuals' in filename:
                            residuals_files.append(file_path)
                track_info = {
                    'trackname': trackname,
                    'bass_audio': bass_audio_files,
                    'residuals': residuals_files
                }
                track_list.append(track_info)
        return track_list
    
    def __len__(self):
        return len(self.track_list)

    def __getitem__(self, index):
        track_info = self.track_list[index]
        bass_audio_tensors = []
        residuals_tensors = []
        # Load and process the audio files into tensors
        for bass_file, residuals_file in zip(track_info['bass_audio'], track_info['residuals']):
            bass_audio, _ = torchaudio.load(bass_file)
            residuals, _ = torchaudio.load(residuals_file)
            bass_audio_tensors.append(bass_audio)
            residuals_tensors.append(residuals)
        
        return {
            'trackname': track_info['trackname'],
            'bass_audio': bass_audio_tensors,
            'residuals': residuals_tensors
        }