import torch
from torch.utils.data import Dataset
import torchaudio


class EmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, tokenizer, max_length=160000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_sampling_rate = 16000

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        input_values, sampling_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(
            sampling_rate, self.target_sampling_rate
        )
        input_values = resampler(input_values)

        if input_values.shape[1] < self.max_length:
            input_values = torch.nn.functional.pad(
                input_values, (0, self.max_length - input_values.shape[1]),
            )
        else:
            input_values = input_values[:, : self.max_length]
        input_values = self.tokenizer(
            input_signal=input_values, length=torch.tensor([input_values.shape[1]])
        )
        return input_values[0].squeeze(0), input_values[1][0], label
