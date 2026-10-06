"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    # 1. Загружаем CSV в DataFrame
    df = pd.read_csv(data.csv_path)

    # 2. Считаем пропуски в каждом столбце.
    # isnull() находит пустые значения, sum() складывает их по столбцам.
    missing_by_column = df.isnull().sum().to_dict()

    # 3. Считаем пассажиров старше 30 лет
    # Фильтруем строки, где Age > 30, и берем длину (len) этого списка
    adults_over_30_count = len(df[df['Age'] > 30])

    # 4. Средний возраст для каждого Pclass
    # groupby('Pclass') группирует строки по классу, а ['Age'].mean() считает среднее.
    # Pandas автоматически игнорирует пропуски (NaN), так что среднее не сломается.
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()

    # 5. Доля выживших для каждого Pclass
    # Survived — это 1 или 0. Среднее значение (mean) как раз даст долю выживших.
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()

    # 6. Пять наибольших значений Fare по убыванию
    # nlargest(5) вернет топ-5, а tolist() превратит их в обычный список
    highest_fares = df['Fare'].nlargest(5).tolist()

    # Возвращаем объект-контракт
    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )