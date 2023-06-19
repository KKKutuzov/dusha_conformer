### Dusha conformer 

Распознавание эмоций в речи (Speech Emotion Recognition, SER) — сложная задача,
имеющая множество практических применений в таких областях, как взаимодействие
человека с диалоговой системой, психология и обслуживание клиентов.

В коде представлен пайплайн обучения и обработки данных DUSHA. А также предобученная модель с примером использования.

Данные можно скачать из репозитория [DUSHA](https://github.com/salute-developers/golos/tree/master/dusha)

Для корректной работы необходимо установить библиотеку [NeMo](https://github.com/NVIDIA/NeMo)
и дополнительные зависимости:

    pip install -r requirements.txt

## Обрабокта данных

Чтобы обработать данные, загрузите сырой набор данных Dusha (crowd.tar, podcast.tar), разархивируйте в папку DATASET_PATH и запустите:

    python processing.py -dataset_path  DATASET_PATH 

Если вы хотите изменить пороговое значение для агрегации, запустите обработку с флагом -threshold:

    python data_utils/processing.py  -dataset-path  DATASET_PATH -threshold THRESHOLD  

Для воспроизведения результатов наилучшей модели необходимо провести агрегацию с порогом 0.8 и 0.9.


## Запуск обучения 

Для запуска обучения необходимо выполнить скрипт:

    python train.py --train-manifest-path PROCESSED_DATASET_TRAIN_PATH --test-manifest-path PROCESSED_DATASET_TRAIN_PATH --checkpoint-dir CHECKPOINT_DIR

PROCESSED_DATASET_TRAIN_PATH, PROCESSED_DATASET_TRAIN_PATH - папка с манифестом для трейна и теста. У наилучшей модели train часть с порогом 0.8, тест часть с порогом 0.9.

CHECKPOINT_DIR - Путь по которому будет сохраняться чекпоинт модели.

## Применение модели 

Для применения модели можно воспользоваться кодом  infer.py

Лучший чекпоинт доступен по [ссылке](https://drive.google.com/file/d/1HMWFDReiZF0rEV4B3KILKgRG8OPUE-Ef/view?usp=sharing)
