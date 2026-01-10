import tensorflow as tf

X = tf.constant([[0.,0.],
                 [0.,1.],
                 [1.,0.],
                 [1.,1.]])

y = tf.constant([[0.],
                 [1.],
                 [1.],
                 [0.]])

W1 = tf.Variable(tf.random.normal([2, 4]))
b1 = tf.Variable(tf.zeros([4]))

W2 = tf.Variable(tf.random.normal([4, 1]))
b2 = tf.Variable(tf.zeros([1]))

def mlp(x):
    hidden = tf.tanh(tf.matmul(x, W1) + b1)
    output = tf.sigmoid(tf.matmul(hidden, W2) + b2)
    return output

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

optimizer = tf.optimizers.Adam(learning_rate=0.1)

for epoch in range(1000):
    with tf.GradientTape() as tape:
        preds = mlp(X)
        loss = loss_fn(y, preds)
    grads = tape.gradient(loss, [W1, b1, W2, b2])
    optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))

print("TensorFlow Low-Level XOR Predictions:")
print(tf.round(mlp(X)))
