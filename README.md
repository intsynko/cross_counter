# Cross counter
Скрипт подсчета машин на перекрестке на записанном видео

![plot](./images/work_example.png)

## Зависимости
Python 3.7 (точно будет работать 3.7.6)
Установить зависимости:
```bash
pip install -r requirements.txt
```
Если использовать нейросеть, то скачать граф yolo.h5 для распонавания [отсюда](https://imageai.readthedocs.io/en/latest/video/index.html#note-imageai-will-switch-to-pytorch-backend-starting-from-june-2021)

И установить зависимости для нейросети:
```bash
pip install -r requirements_web.txt
```

## Запуск
```bash
python example.py
```

Результатом работы является видеофайл с размеченными 
машинами - `output.avi`
