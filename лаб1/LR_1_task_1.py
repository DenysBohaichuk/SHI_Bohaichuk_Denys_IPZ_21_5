import numpy as np
import matplotlib.pyplot as plt

# Функція активації
def step_function(x):
    return 1 if x >= 0 else 0

# Персептрон для функції OR
def or_perceptron(x1, x2):
    weights = np.array([1, 1])  # Ваги
    threshold = -0.5  # Поріг
    inputs = np.array([x1, x2])  # Вхідні дані
    linear_combination = np.dot(weights, inputs) + threshold  # Лінійна комбінація
    return step_function(linear_combination)

# Персептрон для функції AND
def and_perceptron(x1, x2):
    weights = np.array([1, 1])  # Ваги
    threshold = -1.5  # Поріг
    inputs = np.array([x1, x2])  # Вхідні дані
    linear_combination = np.dot(weights, inputs) + threshold  # Лінійна комбінація
    return step_function(linear_combination)

# Генерація випадкових точок
np.random.seed(42)  # Для відтворюваності результатів
num_points = 200  # Кількість точок
x_random = np.random.rand(num_points) * 2 - 0.5  # Генерація значень від -0.5 до 1.5
y_random = np.random.rand(num_points) * 2 - 0.5  # Генерація значень від -0.5 до 1.5

# Класифікація точок за допомогою персептронів OR та AND
or_results = np.array([or_perceptron(x, y) for x, y in zip(x_random, y_random)])
and_results = np.array([and_perceptron(x, y) for x, y in zip(x_random, y_random)])

# Встановлення кольорів: синій для 1, оранжевий для 0
colors_or = ['blue' if result == 1 else 'orange' for result in or_results]
colors_and = ['blue' if result == 1 else 'orange' for result in and_results]

# Створення графіків
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Графік для OR
ax[0].scatter(x_random, y_random, c=colors_or, alpha=0.7)
ax[0].set_title('Функція OR')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')

# Графік для AND
ax[1].scatter(x_random, y_random, c=colors_and, alpha=0.7)
ax[1].set_title('Функція AND')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')

plt.show()
