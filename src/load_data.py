import pandas as pd
import os

def load_data(file_name):
    data_folder = '../data'  # Папка с данными

    # Полный путь к файлу
    file_path = os.path.join(data_folder, file_name)

    # Получаем расширение файла
    file_extension = os.path.splitext(file_path)[1]

    # Загружаем файл в зависимости от его типа
    if file_extension == '.json':
        df = pd.read_json(file_path, lines=True)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла. Пожалуйста, используйте .json или .csv.")

    # Проверяем загруженные данные
    print(df.head())

    return df  # Возвращаем загруженный DataFrame

file_name = 'Sarcasm_Headlines_Dataset_v2.json'  # как в папке data

# Загрузка и чтение данных
df = load_data(file_name)
df.head()