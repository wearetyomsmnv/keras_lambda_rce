import tensorflow as tf

input_shape = (128, 128, 3)
# change input shape

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
    tf.keras.layers.Flatten(name="flatten_1"),
    tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
    tf.keras.layers.Dense(10, activation='softmax', name="output_layer")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.save('test_model.h5')

print("Model save as 'test_model.h5'")

