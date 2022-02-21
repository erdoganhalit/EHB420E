import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

fakefolders = "C:\\Users\\halit.erdogan\\Documents\\ANN Proje\\Audiowav\\RealClips"
for folder in os.listdir(fakefolders):
    folderpath = fakefolders + "\\" + folder
    for file in os.listdir(folderpath):
        filepath = folderpath + "\\" + file
        samples, sample_rate = librosa.load(filepath)
        print("Sample rate of " + folder + " is: " + str(sample_rate))
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

        spec_db = librosa.power_to_db(spec, ref=np.max)

        xx = len(spec)
        yy = len(spec[0])
        print("Spec'in uzunluğu: " + str(xx))
        print("Spec'in elementinin uzunluğu: " + str(yy))

        #fig = plt.figure()
        img = librosa.display.specshow(spec_db, y_axis='mel', x_axis='time')
        plt.title(folder + "-" + file[0:3])
        plt.colorbar(img, format="%+2.f dB")
        plt.savefig( "C:\\Users\\halit.erdogan\\Documents\\ANN Proje\\Audiowav\\Real Mels\\" + folder + "\\" + folder + "-" + file[0:4] + '.png')
        plt.close()



