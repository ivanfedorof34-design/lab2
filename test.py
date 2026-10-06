# test.py
import os
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations
from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITANIC_PATH = os.path.join(BASE_DIR, "titanic.csv")
BRAIN_PATH = os.path.join(BASE_DIR, "brainsize.txt")


def test_titanic():
    data = TitanicInput(csv_path=TITANIC_PATH)
    result = analyze_titanic(data)

    assert result.row_count == 891, "Неверное количество строк в Титанике"
    assert "Age" in result.missing_by_column, "Нет данных о пропусках в Age"
    assert result.adults_over_30_count > 0, "Не нашлись люди старше 30"
    assert len(result.highest_fares) == 5, "Должно быть ровно 5 самых дорогих билетов"
    print("\n[OK] Часть 1 (Титаник) пройдена!")
    print(f"Пропуски в Age: {result.missing_by_column['Age']}")
    print(f"Людей старше 30: {result.adults_over_30_count}")
    print(f"Топ-5 тарифов: {result.highest_fares}")


def test_brain():
    data = BrainDataInput(csv_path=BRAIN_PATH)
    result = analyze_brain_correlations(data)

    assert result.men_count == 20, "Неверное количество мужчин"
    assert result.women_count == 20, "Неверное количество женщин"
    assert len(result.men_mri_correlation) == 5, "Должно быть 5 признаков для мужчин"
    assert result.strongest_mri_feature is not None, "Не найден самый сильный признак"
    print("\n[OK] Часть 2 (Корреляции) пройдена!")
    print(f"Мужчин: {result.men_count}, Женщин: {result.women_count}")
    print(f"Самый сильный признак: {result.strongest_mri_feature}")
    print(f"Корреляции у мужчин: {result.men_mri_correlation}")


if __name__ == "__main__":
    test_titanic()
    test_brain()