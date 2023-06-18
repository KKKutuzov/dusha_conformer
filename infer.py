import torch
import torchaudio
import json

from nemo.collections.asr.modules import (
    ConformerEncoder,
    AudioToMelSpectrogramPreprocessor,
)
from model.dusha_conformer import ClassificationHead, EmotionClassifier


id2label = {0: "angry", 1: "sad", 2: "neutral", 3: "positive"}


def predict(audio_path, model, tokenizer, device):
    input_values, sampling_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    input_values = resampler(input_values)
    if input_values.shape[1] < 160000:
        input_values = torch.nn.functional.pad(
            input_values,
            (0, 160000 - input_values.shape[1]),
        )
    else:
        input_values = input_values[:, :160000]
    input_values = tokenizer(
        input_signal=input_values, length=torch.tensor([input_values.shape[1]])
    )
    return id2label[
        model(input_values[0].to(device), input_values[1].to(device))[0]
        .argmax()
        .cpu()
        .item()
    ]


with open("model/cfg_encoder.json") as handle:
    encoder_cfg = json.loads(handle.read())
with open("model/cfg_processor.json") as handle:
    processor_cfg = json.loads(handle.read())

CHECKPOINT_PATH = ""

model = ConformerEncoder(**encoder_cfg)
tokenizer = AudioToMelSpectrogramPreprocessor(**processor_cfg)
clf = ClassificationHead()

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))
else:
    checkpoint = torch.load(CHECKPOINT_PATH)

emotion_classifier = EmotionClassifier(model, clf, device=device)
emotion_classifier.load_state_dict(checkpoint["state_dict"])
emotion_classifier.eval()

emotion_classifier = emotion_classifier.to(device)

print(
    predict(
        "positive.wav",
        emotion_classifier,
        tokenizer,
        device,
    )
)
