import tensorflow as tf
import numpy as np
from tensorflow import tensorflow_addons
from tensorflow import Keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int64)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature_cols = tf.keras.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.keras.learn.DNNClassifier(hidden_units=[300,100], n_classes=10,
 feature_columns=feature_cols)
dnn_clf = tf.keras.learn.SKCompat(dnn_clf) # if TensorFlow >= 1.1
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)
