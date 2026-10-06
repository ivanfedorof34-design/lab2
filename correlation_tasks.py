"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    # 1. Загружаем данные. sep='\t' указывает, что разделитель — табуляция.
    # na_values=['NA'] говорит Pandas, что строки "NA" — это пропуски.
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])

    # 2. Разделяем на мужчин и женщин
    df_men = df[df['Gender'] == 'Male']
    df_women = df[df['Gender'] == 'Female']

    # Признаки, которые будем коррелировать с MRI_Count
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    # 3. Считаем корреляции для мужчин
    men_corr = {}
    for feature in features:
        # .corr() по умолчанию считает корреляцию Пирсона
        men_corr[feature] = df_men[feature].corr(df_men['MRI_Count'])

    # Считаем корреляции для женщин
    women_corr = {}
    for feature in features:
        women_corr[feature] = df_women[feature].corr(df_women['MRI_Count'])

    # 4. Ищем признак с максимальным модулем корреляции
    strongest_feature = None
    max_abs_corr = -1.0

    # Проверяем мужчин
    for feature, corr_value in men_corr.items():
        if abs(corr_value) > max_abs_corr:
            max_abs_corr = abs(corr_value)
            strongest_feature = feature

    # Проверяем женщин
    for feature, corr_value in women_corr.items():
        if abs(corr_value) > max_abs_corr:
            max_abs_corr = abs(corr_value)
            strongest_feature = feature

    # 5. Возвращаем результат
    return BrainCorrelationSummary(
        men_count=len(df_men),
        women_count=len(df_women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature
    )