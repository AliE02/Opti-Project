import math
import hydra
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
from scipy.io import wavfile as wav
import scipy.signal
from scipy.signal import get_window
from gammatone.filters import make_erb_filters, centre_freqs, erb_filterbank



def exponential_lr_decay(step: int, k: float):
    return math.e ** (-step * k)

def load_datasets(dataset: str, seed: int = 42):
    if dataset == "mnist":
        import torchvision
        transform = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]

        train_dataset = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        test_dataset = torchvision.datasets.MNIST(
            "/tmp/data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )

        val_dataset = test_dataset

        return train_dataset, test_dataset, val_dataset

    elif dataset == "esc10":
        dataset_dir = '../../../data/ESC-50-master/dataset_1.pt'
        if os.path.exists(dataset_dir):
            print('Loading dataset from file')
            dataset = torch.load(dataset_dir)

        # shuffle the dataset list using the seed
        torch.manual_seed(seed)
        # dataset = dataset[torch.randperm(len(dataset))]
        

        # Split dataset into training and testing while keeping the torch tensor format
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        # take 10% of the training dataset as validation
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

        return train_dataset, test_dataset, val_dataset
    ###################################################################
    #                TODO: Implement other datasets                   #
    ###################################################################





    ###################################################################
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
import numpy as np
import librosa
from gammatone.filters import make_erb_filters, centre_freqs, erb_filterbank

def gammatone_filter_bank(signal, sr, num_bands=128, fmin=100, fmax=None):
    """Apply gammatone filter bank to a signal."""
    if fmax is None:
        fmax = sr / 2
    freqs = centre_freqs(sr, num_bands, fmin)
    filters = make_erb_filters(sr, freqs)
    filtered_signal = erb_filterbank(signal, filters)
    return filtered_signal

def calculate_delta(spectrogram):
    """Calculate the delta (first derivative) of a spectrogram."""
    return librosa.feature.delta(spectrogram)

def custom_transform(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', target_size=(128, 128)):    
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        center=center, 
        pad_mode=pad_mode, 
        n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Compute delta (first-order derivative) of the mel spectrogram
    delta_spectrogram = librosa.feature.delta(mel_spectrogram_db)
    
    # Resize the spectrogram and its delta to the target size
    resized_spectrogram = np.resize(mel_spectrogram_db, target_size)
    resized_delta = np.resize(delta_spectrogram, target_size)
    
    # Create a 2-channel representation
    spectrogram_2d = np.stack((resized_spectrogram, resized_delta), axis=-1)

    # convert to tensor
    spectrogram_2d = torch.tensor(spectrogram_2d).permute(2, 0, 1).float()
    
    return spectrogram_2d


class ESC10Dataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['esc10'] == True]
        self.audio_dir = audio_dir
        self.transform = transform

        self.transforms = []

        # load the audio files and perform the transform
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            file_path = os.path.join(self.audio_dir, row['filename'])
            waveform, sample_rate = librosa.load(file_path, sr=None)
            if self.transform:
                self.transforms.append(self.transform(waveform, sample_rate))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.transforms[idx], self.data.iloc[idx]['target']
    

if __name__ == '__main__':
    # print the working directory
    print(os.getcwd())
    train, test, _ = load_datasets("esc10")
    print("Train dataset length:", len(train))
    # print shape of the first sample
    print(train[0][0])