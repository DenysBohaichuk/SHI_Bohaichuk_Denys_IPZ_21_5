import tensorflow as tf
import numpy as np

# Параметри
n_samples = 1000  # Кількість зразків
batch_size = 100  # Розмір міні-батчу
num_steps = 10000  # Кількість кроків

# Генерація синтетичних даних
X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

# Змінні моделі
k = tf.Variable(tf.random.normal([1, 1], dtype=tf.float32), name="k")
b = tf.Variable(tf.zeros([1], dtype=tf.float32), name="b")


# Функція для обчислення втрат
def compute_loss(X, y):
    y_pred = tf.matmul(X, k) + b  # Лінійна модель
    return tf.reduce_mean((y - y_pred) ** 2)


# Оптимізатор
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Тренування моделі
for step in range(1, num_steps + 1):
    # Вибір випадкового міні-батчу
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    # Виконання одного кроку оптимізації
    with tf.GradientTape() as tape:
        loss = compute_loss(X_batch, y_batch)
    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    # Виведення прогресу кожні 1000 кроків
    if step % 1000 == 0 or step == 1:
        print(f"Крок {step}: Втрати={loss.numpy():.4f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")

# Остаточні параметри моделі
print(f"\nОстаточні параметри: k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}")
