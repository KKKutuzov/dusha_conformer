import torch
import torchaudio

import nemo.collections.asr as nemo_asr
from model.dusha_conformer import ClassificationHead, EmotionClassifier


id2label = {0: "angry", 1: "sad", 2: "neutral", 3: "positive"}


def predict(audio_path, model, tokenizer):
    input_values, _ = torchaudio.load(audio_path)
    if input_values.shape[1] < 160000:
        input_values = torch.nn.functional.pad(
            input_values, (0, 160000 - input_values.shape[1]),
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


CHECKPOINT_PATH = ""
ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(
    model_name="ssl_en_conformer_large"
)
model = ssl_model.encoder
tokenizer = ssl_model.preprocessor.cpu()
clf = ClassificationHead()

checkpoint = torch.load(CHECKPOINT_PATH)

emotion_classifier = EmotionClassifier(model, clf)

checkpoint = torch.load(CHECKPOINT_PATH)

emotion_classifier = EmotionClassifier(model, clf)
emotion_classifier.load_state_dict(checkpoint["state_dict"])
emotion_classifier.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_classifier = emotion_classifier.to(device)

print(predict("example.wav"))
