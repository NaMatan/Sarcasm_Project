import re
from time import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import torch

import nltk

from load_data import load_data
from data_preprocessing import clean_data, prepare_data, data_split, vector
from model import train_and_evaluate_models, best_model

file_name = 'Sarcasm_Headlines_Dataset_v2.json'  # как в папке data

# Запуск таймера
start_time = time()

# Загрузка и чтение данных
df = load_data(file_name)

# Чистка и предобработка данных
df = clean_data(df)
print(df)

df = prepare_data(df)

X_train, X_test, y_train, y_test = data_split (df)

# Векторизация текстовых данных
X_train_vec, X_test_vec = vector(X_train, X_test)

# Завершение таймера
end_time = time()
prep_time = end_time - start_time

# Обучение и предсказание
results_df = train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, prep_time)


# Выбор лучшей модели
model1 = best_model(results_df)
print(model1)

