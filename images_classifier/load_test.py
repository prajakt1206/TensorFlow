import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
new_model = tf.keras.models.load_model('../model.h5')

# Check its architecture
i = 600
new_model.summary()
predictions = new_model.predict(test_images)
print(predictions[i])
print(np.argmax(predictions[i]))
print(test_labels[i])
