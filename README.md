# Cross counter
Скрипт подсчета машин на перекрестке на записанном видео

![plot](./images/work_example.png)

## Запуск
Python 3.7+
Установить зависимости:
```bash
pip install -r requirements.txt
```
Если использовать сетку, то скачать граф для распонавания [отсюда](https://imageai.readthedocs.io/en/latest/video/index.html#note-imageai-will-switch-to-pytorch-backend-starting-from-june-2021)
Запуск:
```bash
python main.py
```

Результатом работы является видеофайл с размеченными 
машинами - `output.avi`
