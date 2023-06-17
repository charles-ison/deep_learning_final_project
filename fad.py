# from frechet_audio_distance import FrechetAudioDistance

# # # to use `vggish`
# # frechet = FrechetAudioDistance(
# #     model_name="vggish",
# #     use_pca=False, 
# #     use_activation=False,
# #     verbose=False
# # )
# # to use `PANN`
# frechet = FrechetAudioDistance(
#     model_name="pann",
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )
# fad_score = frechet.score("test/out", "test/tgt")
# print(fad_score)
import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist
import math

def extract_features(wav_file, sample_rate):
    signal, _ = librosa.load(wav_file, sr=sample_rate)
    return librosa.feature.mfcc(y=signal, sr=sample_rate)

def compute_fad_score(features1, features2):
    distances = cdist(features1, features2, metric='euclidean')
    fad = np.max(np.maximum(np.min(distances, axis=1), np.min(distances, axis=0)))
    return fad

directory1 = 'test/out'
directory2 = 'test/tgt'
sample_rate = 22050  # Set the desired sample rate

# Get the list of .wav files in directory1
files1 = [file for file in os.listdir(directory1) if file.endswith('.wav')]

# Get the list of .wav files in directory2
files2 = [file for file in os.listdir(directory2) if file.endswith('.wav')]

fad_scores = []
for file1 in files1:
    for file2 in files2:
        # Construct the full file paths
        file_path1 = os.path.join(directory1, file1)
        file_path2 = os.path.join(directory2, file2)

        # Extract features from the .wav files
        features1 = extract_features(file_path1, sample_rate)
        features2 = extract_features(file_path2, sample_rate)[:,:-1]

        # Compute the FAD score
        fad_score = compute_fad_score(features1, features2)

        fad_scores.append(fad_score)

print("min:", min(fad_scores))
print("min:", max(fad_scores))
print("avg:", sum(fad_scores) / len(fad_scores))
