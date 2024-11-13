import pandas as pd
import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Инициализация лемматизатора и получение списка стоп-слов
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_data(df):
    # Изучение данных
    df.info()
    df.head()

    # Удаление дубликатов
    df.drop_duplicates(ignore_index=True, inplace=True)
    # Удаление столбца с линком, т.к. он не имеет вес
    df = df.drop(columns='article_link')

    # Просмотр нескольких заголовков для примера
    list(df.head(10)['headline'])

    # Проверка дисбаланса классов
    sns.barplot(df.groupby('is_sarcastic').agg('count')['headline']);

    # Длины заголовков
    df.headline.str.len().describe()

    # Вычисление длины строк
    df['length'] = df['headline'].str.len()

    # Вычисление квартилей и IQR
    Q1 = df['length'].quantile(0.25)
    Q3 = df['length'].quantile(0.75)
    IQR = Q3 - Q1

    # Определение границ для выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Удаление выбросов
    df_cleaned = df[(df['length'] >= lower_bound) & (df['length'] <= upper_bound)]

    # Длины заголовков
    df_cleaned.headline.str.len().describe()

    # Удаление столбца с длиной строки
    df_cleaned = df_cleaned.drop(columns='length')

    return df_cleaned

def preprocess_text(text):
    # Удаление специальных символов
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Оставляем только буквы и пробелы
    # Токенизация
    tokens = word_tokenize(text.lower())  # Приведение к нижнему регистру
    # Удаление стоп-слов и лемматизация
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

def prepare_data(df):
    df['headline'] = df['headline'].apply(preprocess_text)

    return df


def data_split (df):
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def vector(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec
