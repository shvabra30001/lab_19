import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

print("Задание 2")
np.random.seed(42)
hours = np.random.uniform(0, 12, 30)
grade = 40 + 5 * hours + np.random.normal(0, 5, 30)

X = hours.reshape(-1, 1)
y = grade

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)

print("Задание 3")
model = LinearRegression()
model.fit(X_train, y_train)

k_val = float(model.coef_.item())
b_val = float(model.intercept_)
print("Наклон (k):", round(k_val, 2))
print("Свободный член (b):", round(b_val, 2))

print("Задание 4")
y_pred = model.predict(X_test)
print("Предсказания:", np.round(y_pred[:3], 1))
print("Реальные:     ", np.round(y_test[:3], 1))

new_hours = np.array(list((list((6.0,)),)))
pred_6h = float(model.predict(new_hours).item())
print("За 6 часов модель обещает:", round(pred_6h, 1))

print("Задание 5")
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", round(mae, 2))
print("Модель ошибается в среднем на", round(mae, 1), "балла")
# Вывод по MAE: ошибка в ~4 балла при шкале от 0 до 100 — это хороший результат (погрешность 4%).
# При шкале от 0 до 5 такая ошибка была бы катастрофической (почти вся шкала).

print("Задание 6")
np.random.seed(42)
n = 30
age = np.random.randint(14, 60, n)
minutes = np.random.randint(10, 400, n)
buy = ((age < 35) & (minutes > 120)).astype(int)
flip = np.random.choice(n, 3, replace=False)
buy[flip] = 1 - buy[flip]

df = pd.DataFrame({"age": age, "minutes": minutes, "bought": buy})
print("Баланс классов:", df["bought"].value_counts().to_dict())

Xk = df[list(("age", "minutes"))]
yk = df["bought"]
Xk_train, Xk_test, yk_train, yk_test = train_test_split(
    Xk, yk, test_size=0.3, random_state=42, stratify=yk
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xk_train, yk_train)
pred = knn.predict(Xk_test)
print("Accuracy kNN:", round(accuracy_score(yk_test, pred), 2))
# Наблюдение при изменении k: при n_neighbors=1 модель сильнее реагирует на шум (переобучается),
# при n_neighbors=3 или 5 предсказания становятся более сглаженными и устойчивыми.

new_user = pd.DataFrame({"age": list((25,)), "minutes": list((300,))})
pred_user = int(knn.predict(new_user).item())
print("Новый пользователь 25 лет, 300 минут -> купит?", pred_user)

print("Задание 7")
os.makedirs("images", exist_ok=True)
plt.figure(figsize=(7, 4))
plt.scatter(X_train.ravel(), y_train, label="Обучающие данные")
plt.scatter(X_test.ravel(), y_test, color="red", label="Тестовые данные")
xs = np.linspace(0, 12, 100)
plt.plot(xs, b_val + k_val * xs, "g-", label="Линия регрессии")
plt.xlabel("Часы подготовки")
plt.ylabel("Оценка")
plt.title("Линейная регрессия: часы -> оценка")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("images/regression.png", dpi=150, bbox_inches="tight")
plt.close()
print("График сохранён в images/regression.png")
