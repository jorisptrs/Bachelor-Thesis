import gc
import os
import pickle as pkl

import librosa
import librosa.display
import numpy as np  # linear algebra
import pandas as pd  # dataset processing, CSV file I/O (e.g. pd.read_csv)
from scipy.interpolate import CubicSpline


class Feature_Collector():
    __data_directory = './dataset'
    __main_directory = './TIMIT'
    _winlen = 0.025
    _winstep = 0.001

    def __init__(self, path=None, stepsize=0.001):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_path = current_dir + '/../cache/'

        self._winstep = stepsize
        self.__main_directory = path + 'TIMIT/'
        self.__data_directory = path + "TIMIT/dataset/"

        # TimitBet 61 phoneme mapping to 39 phonemes
        # by Lee, K.-F., & Hon, H.-W. (1989). Speaker-independent phone recognition using hidden Markov models. IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(11), 1641–1648. doi:10.1109/29.46546 
        self.phon61_map39 = {
            'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ix': 'ih', 'ax': 'ah', 'ah': 'ah', 'uw': 'uw',
            'ux': 'uw', 'uh': 'uh', 'ao': 'aa', 'aa': 'aa', 'ey': 'ey', 'ay': 'ay', 'oy': 'oy', 'aw': 'aw',
            'ow': 'ow', 'l': 'l', 'el': 'l', 'r': 'r', 'y': 'y', 'w': 'w', 'er': 'er', 'axr': 'er',
            'm': 'm', 'em': 'm', 'n': 'n', 'nx': 'n', 'en': 'n', 'ng': 'ng', 'eng': 'ng', 'ch': 'ch',
            'jh': 'jh', 'dh': 'dh', 'b': 'b', 'd': 'd', 'dx': 'dx', 'g': 'g', 'p': 'p', 't': 't',
            'k': 'k', 'z': 'z', 'zh': 'sh', 'v': 'v', 'f': 'f', 'th': 'th', 's': 's', 'sh': 'sh',
            'hh': 'hh', 'hv': 'hh', 'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'qcl': 'h#', 'bcl': 'h#', 'dcl': 'h#',
            'gcl': 'h#', 'h#': 'h#', '#h': 'h#', 'pau': 'h#', 'epi': 'h#', 'nx': 'n', 'ax-h': 'ah', 'q': 'h#'
        }

        self.phon61 = list(self.phon61_map39.keys())

        self.phon39 = list(set(self.phon61_map39.values()))
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
        for p61, p39 in self.phon61_map39.items():
            if not p39 in self.phon39_map61:
                self.phon39_map61[p39] = []
            self.phon39_map61[p39].append(p61)

        pkl.dump(self.label_p39, open(self.cache_path + 'phon_label_index.pkl', 'wb'))
        pkl.dump(self.phon61_map39, open(self.cache_path + 'phon_map_61To39.pkl', 'wb'))

    # ------------------------------------------------------------------------
    def get39EquiOf61(self, p):
        return self.phon61_map39[self.removePhonStressMarker(p)]

    def get39Index(self, phon):
        return self.label_p39[phon]

    def get39Phon(self, index):
        return self.p39_label[index]

    def get61Index(self, phon):
        return self.label_p61[phon]

    def get61Phon(self, index):
        return self.p61_label[index]

    def removePhonStressMarker(self, phon):
        phon = phon.replace('1', '')
        phon = phon.replace('2', '')
        return phon

    def getWindow(self, sr):
        """
        Compute converions for the MFFC 
        """
        nfft = 512
        winlen = self._winlen * sr
        winstep = self._winstep * sr
        return nfft, int(winlen), int(winstep)

    def readTrainingDataDescriptionCSV(self, speakers=[], dr=[], sentence=""):
        """
        Read the relevant part of the training CSV
        """
        file_path = self.__main_directory + 'train_data.csv'  # check if train_data.csv is in correct path
        self._Tdd = pd.read_csv(file_path)
        # removing NaN entries in the train_data.csv file
        if not dr:
            dr = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
        self._Tdd = self._Tdd[self._Tdd['dialect_region'].isin(dr)]
        if speakers != []:
            self._Tdd = self._Tdd[self._Tdd['speaker_id'].isin(speakers)]
        if sentence:
            self._Tdd = self._Tdd[self._Tdd['filename'].str.contains(sentence)]
        return self._Tdd

    def readTestingDataDescriptionCSV(self, speakers=[], dr=[], sentence=""):
        """
        Read the relevant part of the testing CSV
        """
        file_path = self.__main_directory + 'test_data.csv'  # check if train_data.csv is in correct path
        self._tdd = pd.read_csv(file_path)
        if not dr:
            dr = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
        self._tdd = self._tdd[self._tdd['dialect_region'].isin(dr)]
        if speakers != []:
            self._tdd = self._tdd[self._tdd['speaker_id'].isin(speakers)]
        if sentence:
            self._tdd = self._Tdd[self._Tdd['filename'].str.contains(sentence)]
        return self._tdd

    def getListAudioFiles(self, of='Train', speakers=[], dr=[], sentence=""):
        """
        Returns the wav files from the CSV
        """
        if of == 'Train':
            self.readTrainingDataDescriptionCSV(speakers, dr, sentence=sentence)
            return self._Tdd[self._Tdd['is_converted_audio'] == True]
        if of == 'Test':
            self.readTestingDataDescriptionCSV(speakers, dr, sentence=sentence)
            return self._tdd[self._tdd['is_converted_audio'] == True]

    def getListPhonemeFiles(self, of='Train'):
        """
        Returns the phoneme files from the CSV
        """
        if of == 'Train':
            self.readTrainingDataDescriptionCSV()
            return self._Tdd[self._Tdd['is_phonetic_file'] == True]
        if of == 'Test':
            self.readTestingDataDescriptionCSV()
            return self._tdd[self._tdd['is_phonetic_file'] == True]

    def readAudio(self, fpath=None, pre_emp=False):
        """
        Read an audio file
        """
        if (fpath == None):
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

    def getPhonPathFromAudioPath(self, audio_path):
        return audio_path.split(".WAV")[0] + ".PHN"

    def readPhon(self, fpath=None):
        """
        Read a phoneme annotations file
        """
        if (fpath == None):
            raise Exception('phon file path not provided')

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
        wav, sr = self.readAudio(fpath=audio_path, pre_emp=True)
        nfft, winlen, winstep = self.getWindow(sr)
        if (ftype == 'amplitudes'):
            db_melspec = librosa.feature.melspectrogram(wav, sr=sr, hop_length=winstep, win_length=winlen, n_fft=nfft,
                                                        n_mels=n_mels)

        if (ftype == 'mfcc'):
            db_melspec = librosa.feature.mfcc(wav, sr=sr, hop_length=winstep, win_length=winlen, n_mfcc=n_mels)

        mD = None
        mDD = None
        if (delta):
            mD = librosa.feature.delta(db_melspec)
            db_melspec = np.concatenate([db_melspec, mD])
            if (delta_delta):
                mDD = librosa.feature.delta(mD)
                db_melspec = np.concatenate([db_melspec, mDD])

        audio_phon_transcription = None
        if phon_path == None:
            phon_path = self.getPhonPathFromAudioPath(audio_path)

        audio_phon_transcription = self.readPhon(phon_path)

        feature_vectors = []
        db_melspec = db_melspec.T
        time = db_melspec.shape[0]

        labels = []
        for i in range(time):
            # ---collecting feature---
            feature_vectors.append(db_melspec[i])

            # ---collecting phoneme label ---
            start = winstep * i
            end = start + winlen
            index = start + int(winlen / 2)
            # phoneme = list(
            #             audio_phon_transcription[
            #                 ((audio_phon_transcription['start']<=start) & 
            #                 ((audio_phon_transcription['end']-start)>=int(winlen/2)))
            #                 |
            #                 ((audio_phon_transcription['start']<=end) & 
            #                     (audio_phon_transcription['end']>end))  
            #             ].to_dict()['phoneme'].values()
            # )
            phoneme = list(
                audio_phon_transcription[
                    (audio_phon_transcription['start'] <= index) &
                    ((audio_phon_transcription['end'] > index))
                    ].to_dict()['phoneme'].values()
            )

            try:
                if long_version:
                    phoneme = phoneme[0]
                else:
                    phoneme = self.get39EquiOf61(phoneme[0])
                labels.append(phoneme)
            except:
                labels.append('h#')

        return feature_vectors, labels, sr

    def getFeatureAndLabelInSegments(self, audio_path=None, phon_path=None, n_mels=15, delta=True,
                                     delta_delta=True, long_version=False, subsamples=10):
        """
        Returns
        - labels: a list of phonemes
        - feature_vectors: all corresponding sections of the signal
        """
        oversamplings = 0
        if audio_path == None:
            raise Exception("Path to audio (Wav) file must be provided")
        wav, sr = self.readAudio(fpath=audio_path, pre_emp=True)

        audio_phon_transcription = None
        if phon_path == None:
            phon_path = self.getPhonPathFromAudioPath(audio_path)
        audio_phon_transcription = self.readPhon(phon_path)
        split_wav = []
        labels = []

        for _, row in audio_phon_transcription.iterrows():
            split_wav.append(wav[row['start']:row['end']])
            try:
                if long_version:
                    phoneme = row['phoneme']
                else:
                    phoneme = self.get39EquiOf61(row['phoneme'])
            except:
                phoneme = 'h#'
            labels.append(phoneme)

        feature_vectors = []

        for segment in split_wav:
            _, winlen, winstep = self.getWindow(sr)
            db_melspec = librosa.feature.mfcc(segment, sr=sr, hop_length=winstep, win_length=winlen, n_mfcc=n_mels)

            mD = None
            mDD = None
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
    def label_to_1hot(self, l, long_version=False):
        """
        Converts a label to an 1-hot encoded array 
        """
        label = [0 for i in range(63 if long_version else 39)]
        if long_version:
            label[self.label_p61[l] - 1] = 1
        else:
            label[self.label_p39[l] - 1] = 1
        return np.array(label)

    def labels_to_1hot(self, labels, long_version=False):
        """
        Converts a list of labels to 1-hot encodings
        """
        print('Preparing Labels')
        label_vector = []
        for l in labels:
            label_vector.append(self.label_to_1hot(l, long_version))

        return label_vector

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
            tddA = self.getListAudioFiles(ft, speakers=speakers, dr=dr, sentence=sentence)
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
            tddA = self.getListAudioFiles(ft, speakers, dr, sentence)
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
