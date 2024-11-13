import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, prep_time):
    # Список моделей для обучения
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    # Словари для хранения результатов
    results = {}

    # Обучение моделей и оценка их производительности
    for model_name, model in models.items():
        start_time = time()  # Запуск таймера
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        end_time = time()  # Остановка таймера

        # Сохранение результатов с учетом времени подготовки
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'time': (end_time - start_time) + prep_time  # Добавление времени подготовки
        }

    # Преобразование результатов в DataFrame для удобства
    results_df = pd.DataFrame(results).T

    # Вывод результатов
    print(results_df)

    return results_df  # Вернуть DataFrame с результатами








def plots(results_df):
    # Построение графиков
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # График F1 Score
    axs[0, 0].bar(results_df.index, results_df['f1_score'], color='skyblue')
    axs[0, 0].set_title('F1 Score')
    axs[0, 0].set_ylabel('F1 Score')
    axs[0, 0].axhline(y=0.7, color='r', linestyle='--')  # Линия для 70%

    # График Precision
    axs[0, 1].bar(results_df.index, results_df['precision'], color='lightgreen')
    axs[0, 1].set_title('Precision')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].axhline(y=0.7, color='r', linestyle='--')

    # График Recall
    axs[1, 0].bar(results_df.index, results_df['recall'], color='salmon')
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].axhline(y=0.7, color='r', linestyle='--')

    # График времени выполнения
    axs[1, 1].bar(results_df.index, results_df['time'], color='gold')
    axs[1, 1].set_title('Time to Train (seconds)')
    axs[1, 1].set_ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

    return


def best_model(results_df):
    # Нормализация времени
    results_df['normalized_time'] = (results_df['time'] - results_df['time'].min()) / (
                results_df['time'].max() - results_df['time'].min())

    # Расчет KPI
    results_df['KPI'] = (results_df['accuracy'] +
                         results_df['f1_score'] +
                         results_df['precision'] +
                         results_df['recall'] -
                         results_df['normalized_time'])
    print(results_df)

    # Поиск модели с максимальным KPI
    best_model_index = results_df['KPI'].idxmax()
    best_model = results_df.loc[best_model_index]
    print("")
    print("*" * 100)
    print("Лучшая модель:")
    print(best_model.name)  # Выводим всю информацию о лучшей модели

    return best_model
