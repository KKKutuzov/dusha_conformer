import nemo.collections.asr as nemo_asr

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import click
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, TanhDistortion


from model.dusha_conformer import ClassificationHead, EmotionClassifier
from data_utils.dataset import EmotionDataset


@click.command()
@click.option(
    "-train", "--train-manifest-path", required=True, type=click.Path(exists=True),
)
@click.option(
    "-test", "--test-manifest-path", required=True, type=click.Path(exists=True),
)
@click.option(
    "-c", "--checkpoint-dir", required=True, type=click.Path(exists=True),
)
def train(train_manifest_path: Path, test_manifest_path: Path, checkpoint_dir: Path):

    ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(
        model_name="ssl_en_conformer_large"
    )
    model = ssl_model.encoder
    tokenizer = ssl_model.preprocessor.cpu()
    checkpoint_callback_podcast = ModelCheckpoint(
        monitor="macro_f1_podcast",
        filename="dusha_conformer-{epoch:02d}-{macro_f1_podcast:.2f}",
        mode="max",
    )
    checkpoint_callback_crowd = ModelCheckpoint(
        monitor="macro_f1_crowd",
        filename="dusha_conformer-{epoch:02d}-{macro_f1_crowd:.2f}",
        mode="max",
    )
    train_crowd = pd.read_table(train_manifest_path / "crowd_train.tsv").sample(5)
    train_podcast = pd.read_table(train_manifest_path / "podcast_train.tsv").sample(5)
    test_crowd = pd.read_table(test_manifest_path / "crowd_test.tsv").sample(5)
    test_podcast = pd.read_table(test_manifest_path / "podcast_test.tsv").sample(5)

    train = pd.concat([train_crowd, train_podcast])
    class_sample_count = np.array(
        [
            len(np.where(train.label.values == t)[0])
            for t in np.unique(train.label.values)
        ]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train.label.values])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight)
    )
    augm_transform = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            TimeStretch(min_rate=0.5, max_rate=1.4, p=0.3),
            TanhDistortion(min_distortion=0.01, max_distortion=0.5, p=0.3),
        ]
    )

    train_dataset = EmotionDataset(
        train.path.values, train.label.values, tokenizer, augm_transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_crowd_dataset = EmotionDataset(
        test_crowd.path.values, test_crowd.label.values, tokenizer
    )

    val_crowd_dataloader = DataLoader(val_crowd_dataset, batch_size=8)

    val_podcast_dataset = EmotionDataset(
        test_podcast.path.values, test_podcast.label.values, tokenizer
    )

    val_podcast_dataloader = DataLoader(val_podcast_dataset, batch_size=8)

    emotion_classifier = EmotionClassifier(model, clf)

    clf = ClassificationHead()

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=25,
        callbacks=[checkpoint_callback_podcast, checkpoint_callback_crowd],
        default_root_dir=checkpoint_dir,
    )

    trainer.fit(
        emotion_classifier,
        train_dataloader,
        [val_crowd_dataloader, val_podcast_dataloader],
    )
    return 0


if __name__ == "__main__":
    train()
