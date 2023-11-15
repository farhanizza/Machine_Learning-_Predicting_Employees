import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

employee = pd.read_csv('./Employee.csv')

employee.rename(columns={'JoiningYear': 'Join', 'PaymentTier': 'Position', 'EverBenched': 'UnemployedAtWork', 'ExperienceInCurrentDomain': 'Experience'}, inplace=True)

dataset = employee[['Education', 'Join', 'City', 'Position', 'Age', 'Gender', 'UnemployedAtWork', 'Experience', 'LeaveOrNot']]

from sklearn.preprocessing import LabelEncoder

Lb = LabelEncoder()

dataset['labelEducation'] = Lb.fit_transform(dataset['Education'])

dataset['labelUnemployedAtWork'] = Lb.fit_transform(dataset['UnemployedAtWork'])

dataset['labelGender'] = Lb.fit_transform(dataset['Gender'])

dataset['labelCity'] = Lb.fit_transform(dataset['City'])

X = dataset[['labelEducation', 'Join', 'labelCity', 'Position', 'Age', 'labelGender', 'labelUnemployedAtWork', 'Experience']]

y = dataset['LeaveOrNot']

algorithm = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(4, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')
])

algorithm.compile(optimizer=tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

history = algorithm.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

y_pred = algorithm.predict(X_test)

y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)

print("Classification Report:")

print(classification_report(y_test, y_pred_classes))

print("\n")

print("Accuracy Score Model Deep Learning: {:.2f}%".format(accuracy * 100))
