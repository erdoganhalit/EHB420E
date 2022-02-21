

#print("stop")
import os

from pydub import AudioSegment
import math


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - min_per_split:
                print('All splited successfully')

fakefolders = "C:\\Users\\halit.erdogan\\Documents\\ANN Proje\\Audiowav\\Real"
for folder in os.listdir(fakefolders):
    #print(folder)
    folder = "C:\\Users\\halit.erdogan\\Documents\\ANN Proje\\Audiowav\\Real" + "\\" + folder
    file = os.listdir(folder)[0]
    #print(file)
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=10)
    print("Next folder")