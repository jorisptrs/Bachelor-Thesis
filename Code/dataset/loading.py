import gc
import os
import pickle as pkl

import librosa
import librosa.display
import numpy as np  # linear algebra
import pandas as pd  # dataset processing, CSV file I/O (e.g. pd.read_csv)
from scipy.interpolate import CubicSpline

# Inspired by https://www.kaggle.com/code/reganmaharjan/phoneme-recognition-using-hmm-on-timit

class DataLoader():

    def __init__(self, path, cache_dir=None, stepsize=0.001, winlen=0.025):

        self._winstep = stepsize
        self._winlen = winlen
        self._nfft = 512
        self.__main_directory = path + 'TIMIT/'
        self.__data_directory = self.__main_directory + "data/"
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            cache_dir = current_dir + '/../cache/'
        self.cache_path = cache_dir

        # TimitBet 61 phoneme mapping to 39 phonemes
        # by Lee, K.-F., & Hon, H.-W. (1989). Speaker-independent phone recognition using hidden Markov models. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(11), 1641–1648. doi:10.1109/29.46546 
        self.phone_map = {
            'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ix': 'ih', 'ax': 'ah', 'ah': 'ah', 'uw': 'uw',
            'ux': 'uw', 'uh': 'uh', 'ao': 'aa', 'aa': 'aa', 'ey': 'ey', 'ay': 'ay', 'oy': 'oy', 'aw': 'aw',
            'ow': 'ow', 'l': 'l', 'el': 'l', 'r': 'r', 'y': 'y', 'w': 'w', 'er': 'er', 'axr': 'er',
            'm': 'm', 'em': 'm', 'n': 'n', 'nx': 'n', 'en': 'n', 'ng': 'ng', 'eng': 'ng', 'ch': 'ch',
            'jh': 'jh', 'dh': 'dh', 'b': 'b', 'd': 'd', 'dx': 'dx', 'g': 'g', 'p': 'p', 't': 't',
            'k': 'k', 'z': 'z', 'zh': 'sh', 'v': 'v', 'f': 'f', 'th': 'th', 's': 's', 'sh': 'sh',
            'hh': 'hh', 'hv': 'hh', 'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'bcl': 'h#', 'dcl': 'h#',
            'gcl': 'h#', 'h#': 'h#', 'pau': 'h#', 'epi': 'h#', 'ax-h': 'ah', 'q': 'h#'
        }
        self.phon61 = list(self.phone_map.keys())
        self.phon39 = list(set(self.phone_map.values()))
        self.phon61.sort()
        self.phon39.sort()

        self.label_p39 = {}
        self.p39_label = {}
        for i, p in enumerate(self.phon39):
            self.label_p39[p] = i
            self.p39_label[i] = p

        self.label_p61 = {}
        self.p61_label = {}
        for i, p in enumerate(self.phon61):
            self.label_p61[p] = i
            self.p61_label[i] = p

        self.phon39_map61 = {}
        for p61, p39 in self.phone_map.items():
            if not p39 in self.phon39_map61:
                self.phon39_map61[p39] = []
            self.phon39_map61[p39].append(p61)

        pkl.dump(self.label_p39, open(self.cache_path + 'phon_label_index.pkl', 'wb'))
        pkl.dump(self.phone_map, open(self.cache_path + 'phon_map_61To39.pkl', 'wb'))

    # ------------------------------------------------------------------------

    def get_39_from_61(self, p):
        return self.phone_map[self.remove_stress_markers(p)]

    @staticmethod
    def remove_stress_markers(phone):
        phone = phone.replace('1', '')
        phone = phone.replace('2', '')
        return phone

    def get_window(self, sr):
        """
        Compute converions for the MFFC 
        """
        return int(self._winlen * sr), int(self._winstep * sr)

    def read_descriptions(self, train_or_test="Train", speakers=[], dr=[], sentence=""):
        """
        Read the relevant part of the testing or training CSV
        """
        file_path = self.__main_directory
        file_path += 'test_data.csv' if train_or_test == "Test" else 'train_data.csv'
        descriptions = pd.read_csv(file_path)

        if not dr:
            dr = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
        descriptions = descriptions[descriptions['dialect_region'].isin(dr)]
        if speakers != []:
            descriptions = descriptions[descriptions['speaker_id'].isin(speakers)]
        if sentence:
            descriptions = descriptions[descriptions['filename'].str.contains(sentence)]

        return descriptions

    def get_audio_file_names(self, of='Train', speakers=[], dr=[], sentence=""):
        """
        Returns the wav files from the CSV
        """
        descriptions = self.read_descriptions(of, speakers, dr, sentence=sentence)
        return descriptions[descriptions['is_converted_audio'] == True]

    def get_transcription_file_names(self, of='Train'):
        """
        Returns the phoneme files from the CSV
        """
        descriptions = self.read_descriptions(of)
        return descriptions[descriptions['is_phonetic_file'] == True]

    def read_audio(self, fpath=None, pre_emp=False):
        """
        Read an audio file
        """
        if fpath is None:
            return np.zeros(1), 0
        fpath = self.__data_directory + fpath
        if os.path.exists(fpath):
            S, sr = librosa.load(fpath, sr=None)
            if pre_emp:
                S = librosa.effects.preemphasis(S)
            return S, sr
        else:
            return np.zeros(1), 0

    # -----------------------end readAudio()

    def get_transcription_path_from_audio_path(self, audio_path):
        return audio_path.split(".WAV")[0] + ".PHN"

    def read_transcription(self, fpath=None):
        """
        Read a transcriptions file
        """
        if fpath is None:
            raise Exception('phon file data_dir not provided')

        fpath = self.__data_directory + fpath
        ph_ = pd.read_csv(fpath, sep=" ")
        first = ph_.columns
        ph_.columns = ['start', 'end', 'phoneme']
        ph_.loc[-1] = [int(first[0]), int(first[1]), first[2]]
        ph_ = ph_.sort_index()
        ph_.index = range(ph_.index.shape[0])
        return ph_

    # ---------------end readPhon()

    def getFeatureAndLabel(self, ftype='mfcc', audio_path=None, phon_path=None, n_mels=128, delta=False,
                           delta_delta=False, long_version=False, subsamples=10):
        """
        Returns
        - feature_vectors (a list of all mfcc time steps found in the audio sample)
        - labels (corresponding phonemes)
        """
        if audio_path == None:
            raise Exception("Path to audio (Wav) file must be provided")
        wav, sr = self.read_audio(fpath=audio_path, pre_emp=True)
        winlen, winstep = self.get_window(sr)
        if (ftype == 'amplitudes'):
            db_melspec = librosa.feature.melspectrogram(wav, sr=sr, hop_length=winstep, win_length=winlen,
                                                        n_fft=self._nfft,
                                                        n_mels=n_mels)

        if (ftype == 'mfcc'):
            db_melspec = librosa.feature.mfcc(wav, sr=sr, hop_length=winstep, win_length=winlen, n_mfcc=n_mels)

        if (delta):
            mD = librosa.feature.delta(db_melspec)
            db_melspec = np.concatenate([db_melspec, mD])
            if (delta_delta):
                mDD = librosa.feature.delta(mD)
                db_melspec = np.concatenate([db_melspec, mDD])

        audio_phon_transcription = None
        if phon_path is None:
            phon_path = self.get_transcription_path_from_audio_path(audio_path)

        audio_phon_transcription = self.read_transcription(phon_path)

        feature_vectors = []
        db_melspec = db_melspec.T
        time = db_melspec.shape[0]

        labels = []
        for i in range(time):
            # ---collecting p---
            feature_vectors.append(db_melspec[i])

            # ---collecting phoneme label ---
            start = winstep * i
            end = start + winlen
            index = start + int(winlen / 2)
            phoneme = list(
                audio_phon_transcription[
                    (audio_phon_transcription['start'] <= index) &
                    ((audio_phon_transcription['end'] > index))
                    ].to_dict()['phoneme'].values()
            )

            try:
                phoneme = phoneme[0]
                if not long_version:
                    phoneme = self.get_39_from_61(phoneme)
                labels.append(phoneme)
            except:
                labels.append('h#')

        return feature_vectors, labels, sr

    def getFeatureAndLabelInSegments(self, audio_path=None, phon_path=None, n_mels=15, delta=True,
                                     delta_delta=False, long_version=False, subsamples=10):
        """
        Returns
        - labels: a list of phonemes
        - feature_vectors: all corresponding sections of the signal
        """
        oversamplings = 0
        if audio_path is None:
            raise Exception("Path to audio (Wav) file must be provided")
        wav, sr = self.read_audio(fpath=audio_path, pre_emp=True)

        if phon_path is None:
            phon_path = self.get_transcription_path_from_audio_path(audio_path)
        audio_phon_transcription = self.read_transcription(phon_path)
        split_wav = []
        labels = []

        for _, row in audio_phon_transcription.iterrows():
            split_wav.append(wav[row['start']:row['end']])
            try:
                phoneme = row['phoneme']
                if not long_version:
                    phoneme = self.get_39_from_61(phoneme)
            except:
                phoneme = 'h#'
            labels.append(phoneme)

        feature_vectors = []

        for segment in split_wav:
            winlen, winstep = self.get_window(sr)
            db_melspec = librosa.feature.mfcc(segment, sr=sr, hop_length=winstep, win_length=winlen, n_mfcc=n_mels)

            if (delta):
                width = db_melspec.shape[1] + db_melspec.shape[1] % 2 - 1 if db_melspec.shape[1] < 9 else 9
                mD = librosa.feature.delta(db_melspec, width=width)
                db_melspec = np.concatenate([db_melspec, mD])
                if (delta_delta):
                    mDD = librosa.feature.delta(mD, width=width)
                    db_melspec = np.concatenate([db_melspec, mDD])
            db_melspec = db_melspec.T

            if subsamples:
                if subsamples > db_melspec.shape[0]:
                    oversamplings += 1
                cs = CubicSpline(np.arange(db_melspec.shape[0]), db_melspec)
                db_melspec = cs(np.linspace(0, db_melspec.shape[0], subsamples))
            feature_vectors.append(db_melspec)

        return feature_vectors, labels, oversamplings

    # --------------------end getMelSpectrogramFeatureAndLabel()

    def collectFeatures(self, ft='Train', ftype='mfcc', n_mels=128, delta=False, delta_delta=False,
                        normalize=True, long_version=False, speakers=[], dr=[], sentence=[], n=0, path_option=""):
        """
        Returns
        - feature_vectors (a list of all mfcc time steps found in the audio sample)
        - labels (corresponding phonemes)        
        """

        if path_option and os.path.exists(self.cache_path + path_option + '_features.pkl') and os.path.exists(
                self.cache_path + path_option + '_labels.pkl'):
            print("-from output")
            ffp = open(self.cache_path + path_option + '_features.pkl', 'rb')
            flp = open(self.cache_path + path_option + '_labels.pkl', 'rb')
            features = pkl.load(ffp)
            labels = pkl.load(flp)
            ffp.close()
            flp.close()
            print('---- success')
            return features, labels, 0
            # --------
        else:
            print('--- Failed')
            print('Collecting Features from Audio Files')
            # -------------
            tddA = self.get_audio_file_names(ft, speakers=speakers, dr=dr, sentence=sentence)
            tddA.index = range(tddA.shape[0])
            feature_vectors = []
            labels = []
            sr = 0
            for i in range(tddA.shape[0]):
                print(tddA.loc[i]['path_from_data_dir'])
                if not i % 100:
                    print(i)
                fv, lv, sr = self.getFeatureAndLabel(ftype=ftype, audio_path=tddA.loc[i]['path_from_data_dir'],
                                                     n_mels=n_mels, delta=delta, delta_delta=delta_delta,
                                                     long_version=long_version)
                feature_vectors += fv
                labels += lv
                if n != 0 and len(feature_vectors) > n:
                    break

            print(f"length of feature_vectors is {n} and length of labels is {n}")
            if n:
                labels = np.asarray(np.array(labels[:n]))
                feature_vectors = np.asarray(np.array(feature_vectors[:n], dtype=object)).astype(np.float32)
            else:
                labels = np.asarray(np.array(labels))
                feature_vectors = np.asarray(np.array(feature_vectors, dtype=object)).astype(np.float32)
            if normalize:
                mini = np.expand_dims(feature_vectors.min(axis=1), 1)
                maxi = np.expand_dims(feature_vectors.max(axis=1), 1)
                feature_vectors = (feature_vectors - mini) / (maxi - mini) - .5
            if path_option:
                ffp = open(self.cache_path + path_option + "_features.pkl", 'wb')
                pkl.dump(feature_vectors, ffp)
                flp = open(self.cache_path + path_option + "_labels.pkl", 'wb')
                pkl.dump(labels, flp)
                ffp.close()
                flp.close()
            print('--- Completed')
            # -------

            return feature_vectors, labels, sr

    def collectFeaturesInSegments(self, ft='Train', n_mels=15, delta=False, delta_delta=False,
                                  normalize=True, long_version=False, speakers=[], dr=[], sentence="", subsamples=10,
                                  path_option=""):
        """
        Returns
        - labels: a list of phonemes
        - feature_vectors: all corresponding sections of the signal
        """

        if path_option != "" and os.path.exists(self.cache_path + path_option + '_features.pkl') and os.path.exists(
                self.cache_path + path_option + '_labels.pkl'):
            print("-from output")
            ffp = open(self.cache_path + path_option + '_features.pkl', 'rb')
            flp = open(self.cache_path + path_option + '_labels.pkl', 'rb')
            features = pkl.load(ffp)
            labels = pkl.load(flp)
            ffp.close()
            flp.close()
            print('---- success')
            return features, labels, 0
            # --------
        else:
            print('--- Failed')
            print('Collecting Features from Audio Files')
            # -------------
            tddA = self.get_audio_file_names(ft, speakers, dr, sentence)
            tddA.index = range(tddA.shape[0])
            feature_vectors = []
            labels = []

            print(tddA.shape[0])
            for i in range(tddA.shape[0]):
                if not i % 100:
                    print(i)
                fv, lv, oversamplings = self.getFeatureAndLabelInSegments(audio_path=tddA.loc[i]['path_from_data_dir'],
                                                                          n_mels=n_mels,
                                                                          delta=delta, delta_delta=delta_delta,
                                                                          long_version=long_version,
                                                                          subsamples=subsamples)
                for feature in fv:
                    feature_vectors.append(np.asarray(np.array(feature, dtype=object)).astype(np.float32))
                labels += lv

            if normalize:
                unrolled = np.asarray(feature_vectors).transpose(1, 0, 2).reshape(feature_vectors[0].shape[0], -1)
                mini = np.expand_dims(unrolled.min(axis=1), 1)
                maxi = np.expand_dims(unrolled.max(axis=1), 1)
                feature_vectors = [(fv - mini) / (maxi - mini) - .5 for fv in feature_vectors]
            if path_option != "":
                ffp = open(self.cache_path + path_option + "_features.pkl", 'wb')
                pkl.dump(feature_vectors, ffp)
                flp = open(self.cache_path + path_option + "_labels.pkl", 'wb')
                pkl.dump(labels, flp)
                ffp.close()
                flp.close()
            print('--- Completed')
            # -------
        gc.collect()

        print(f"Loaded to {len(feature_vectors)} samples of shape {feature_vectors[0].shape}")
        return feature_vectors, labels, oversamplings
