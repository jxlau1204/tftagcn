import pickle
import numpy as np  
import pandas as pd
import  matplotlib.pyplot as plt 
import librosa
from collections import defaultdict
import decimal
import math
import librosa.display
import os

SEGMENT_LEN = 0.265 
SEGMENT_STEP = 0.025 

class Extract_Spectrogram():
    def __init__(self):
        self.m_bands = 128

        self.nfft = 512  
        self.winlen = 0.016  
        self.winstep = 0.008 
        self.segment_len = SEGMENT_LEN  # 2 
        self.segment_step = SEGMENT_STEP  # 1 
        self.segment_nums = defaultdict(list)  
        self.spectrum = defaultdict(list)  
        self.digit_label_summary = defaultdict(list)  
        self.files_name = []
        self.speakers = []
        self.dialog_utterances = []
        self.utterances_label = []

    def _round_half_up(self, number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

    def segmentsig(self, sig, segment_len, segment_step, winfunc=lambda x: np.ones((x,))):
        slen = len(sig)
        segment_len = int(self._round_half_up(segment_len))
        segment_step = int(self._round_half_up(segment_step))
        if slen <= segment_len:
            numsegments = 1
        else:
            numsegments = 1 + int(math.ceil((1.0 * slen - segment_len) / segment_step))

        padlen = int((numsegments - 1) * segment_step + segment_len)
        zeros = np.zeros((padlen - slen,))
        padsignal = np.concatenate((sig, zeros))

        indices = np.tile(np.arange(0, segment_len), (numsegments, 1)) + np.tile(
            np.arange(0, numsegments * segment_step, segment_step), (segment_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(segment_len), (numsegments, 1))
        return frames * win

    def extract_spectrogram(self, single_file_path, result_file = None):
        sample_rate, ori_sig = self.audio_read(single_file_path)
        sig = self.segmentsig(sig=ori_sig, segment_len=self.segment_len * sample_rate,
                                segment_step=self.segment_step * sample_rate)
        segment_num = sig.shape[0]  
        spectrums = []
        for j in range(segment_num):
            S = librosa.feature.melspectrogram(y=sig[j], sr=sample_rate, n_fft=self.nfft,
                                                win_length=int(self.winlen * sample_rate),
                                                hop_length=int(self.winstep * sample_rate),
                                                window="hann", n_mels=self.m_bands)  
            logmelspec = librosa.power_to_db(S, ref=np.max).T
            spectrums.append(logmelspec[:32,:128])
        return segment_num, spectrums
    def audio_read(self, single_file_path, sr=16000):
        y, sr = librosa.load(single_file_path, sr=sr)
        return sr, y
    def plot_spectrogram(self, logmelspec, sample_rate, result_file):
        librosa.display.specshow(logmelspec, sr=sample_rate, hop_length=int(self.winstep * sample_rate), x_axis="s", y_axis="mel")
        plt.xlabel("Time/ms")
        plt.ylabel("Frequency/Hz")
        plt.title("Log-Mel Spectrogram")
        if result_file != None:
            plt.savefig(result_file)

if __name__ == "__main__":
    data_csv = "data_csv/IEMOCAP.csv"
    audio_data_path = "data/iemocap_vad/full/{session}/{fileName}.wav"
    spectrogram_data_path = "data/iemocap_vad/spectrograms"
    extract_spectrogram = Extract_Spectrogram()
    data_pd = pd.read_csv(data_csv)
    sessions = {}
    labels_name = ["N", "A", "H", "S"]
    
    
    for session_i in range(1, 6):
        last_dialog = "blank"
        segment_nums = []
        spectrograms = []
        speakers = []
        dialog_utterances = []
        labels = []
        utterances_label = []
        data_pd_i = data_pd[data_pd["session"]==f"Session{session_i}"]
        for i, row_i in data_pd_i.iterrows():
            print(f"processing file {i}")
            segment, label, text, video, segment2, session, dialog, frames_num, wav_num = row_i
            audio_file_name = audio_data_path.format(**{"session":session.lower(),"fileName":segment2})
            segment_num, spectrogram = extract_spectrogram.extract_spectrogram(audio_file_name)
            spectrograms.append(np.stack(spectrogram))
            segment_nums.append(segment_num)
            labels.extend([labels_name.index(label)] * segment_num)
            utterances_label.append(labels_name.index(label))
            speakers.append(['F', 'M'].index(segment[-4]))
            if dialog != last_dialog:
                last_dialog = dialog
                dialog_utterances.append(1)
            else:
                dialog_utterances[-1] += 1
                
        sessions[session_i] = {
            "spectrograms":np.concatenate(spectrograms),
            "segment_nums":np.array(segment_nums),
            "speakers":np.array(speakers),
            "dialog_utterances":np.array(dialog_utterances),
            "labels":np.array(labels),
            "utterances_label":np.array(utterances_label)
        } 
    
    for flod_i in range(1, 6):
        
        valid = sessions[flod_i]
        train_session_index = list(range(1,6))
        train_session_index.remove(flod_i)
        train = {
            "spectrograms":np.concatenate([sessions[i]["spectrograms"] for i in train_session_index]),
            "segment_nums":np.concatenate([sessions[i]["segment_nums"] for i in train_session_index]),
            "speakers":np.concatenate([sessions[i]["speakers"] for i in train_session_index]),
            "labels":np.concatenate([sessions[i]["labels"] for i in train_session_index]),
            "utterances_label":np.concatenate([sessions[i]["utterances_label"] for i in train_session_index]),
            "dialog_utterances":np.concatenate([sessions[i]["dialog_utterances"] for i in train_session_index])
        }
        spectrogram_data_path_i = f"{spectrogram_data_path}/cross_flod{flod_i}"
        if not os.path.isdir(spectrogram_data_path_i):
            os.makedirs(spectrogram_data_path_i)
        train_path = open(f"{spectrogram_data_path_i}/train.pkl", 'wb')
        pickle.dump(train, train_path)
        valid_path = open(f"{spectrogram_data_path_i}/valid.pkl", 'wb')
        pickle.dump(valid, valid_path)
        
    print("finished")   