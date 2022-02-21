import os
import librosa
import numpy as np
import json

clipfolders = "C:\\Users\\halit.erdogan\\Documents\\ANN Proje\\Audiowav\\FullClips"

allaudio = []

for f_r_folder in os.listdir(clipfolders):
    fr_path = clipfolders + "\\" + f_r_folder
    for folder in os.listdir(fr_path):
        folder_path = fr_path + "\\" + folder
        for file in os.listdir(folder_path):
            file_path = folder_path + "\\" + file

            samples, sample_rate = librosa.load(file_path, sr=None)

            noise = max(samples) * 0.005 * np.random.randn(len(samples))
            samples = samples + noise

            spec = librosa.feature.melspectrogram(y=samples,
                                                  sr=sample_rate,
                                                  n_fft=2048,
                                                  hop_length=512,
                                                  win_length=None,
                                                  window='hann',
                                                  center=True,
                                                  pad_mode='reflect',
                                                  power=2.0,
                                                  n_mels=128)

            #spec = librosa.power_to_db(spec, ref=np.max)

            spec = librosa.feature.mfcc(S = librosa.power_to_db(spec, ref=np.max))

            if spec.shape[1] == 938:
                specList = spec.tolist()
                sampleTemp = {"person": folder.split()[0] , "file":file, "mel":specList , "label" : f_r_folder}
                allaudio.append(sampleTemp)

with open('FULL_spects_and_labels_db_mfcc.json', 'w') as fout:
    json.dump(allaudio , fout)