import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load data
employee = pd.read_csv('./Employee.csv')
employee.rename(columns={'JoiningYear': 'Join', 'PaymentTier': 'Position', 'EverBenched': 'UnemployedAtWork', 'ExperienceInCurrentDomain': 'Experience'}, inplace=True)
dataset = employee[['Education', 'Join', 'City', 'Position', 'Age', 'Gender', 'UnemployedAtWork', 'Experience', 'LeaveOrNot']]

# Label encoding
from sklearn.preprocessing import LabelEncoder
Lb = LabelEncoder()
dataset['labelEducation'] = Lb.fit_transform(dataset['Education'])
dataset['labelUnemployedAtWork'] = Lb.fit_transform(dataset['UnemployedAtWork'])
dataset['labelGender'] = Lb.fit_transform(dataset['Gender'])
dataset['labelCity'] = Lb.fit_transform(dataset['City'])

# Features and target
X = dataset[['labelEducation', 'Join', 'labelCity', 'Position', 'Age', 'labelGender', 'labelUnemployedAtWork', 'Experience']]
y = dataset['LeaveOrNot']

# Model definition
algorithm = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

algorithm.compile(optimizer=tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lists to store results
epoch_range = range(1, 101)
train_accuracy_list, val_accuracy_list, loss_list = [], [], []

# Training with epoch logging
for epoch in epoch_range:
    sys.stdout.write("\rEpoch {}/{} - Training Accuracy: {:.4f} - Validation Accuracy: {:.4f} - Loss {:.4f}".format(epoch, len(epoch_range), train_accuracy_list[-1] if train_accuracy_list else 0, val_accuracy_list[-1] if val_accuracy_list else 0, loss_list[-1] if loss_list else 0))
    sys.stdout.flush()
    
    history = algorithm.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    train_accuracy_list.append(history.history['accuracy'][0])
    val_accuracy_list.append(history.history['val_accuracy'][0])
    loss_list.append(history.history['loss'][0])

# Create DataFrame from results
results_df = pd.DataFrame({
    'Epoch': epoch_range, 
    'Train Accuracy': train_accuracy_list, 
    'Validation Accuracy': val_accuracy_list,
    'Loss': loss_list
})

last_train_accuracy = results_df['Train Accuracy'].iloc[-1]
last_val_accuracy = results_df['Validation Accuracy'].iloc[-1]
last_loss = results_df['Loss'].iloc[-1]

# Clear loading message
sys.stdout.write('\r' + ' ' * 100 + '\r')
sys.stdout.flush()

# Menampilkan informasi
print("Train Accuracy: {:.4f}%".format(last_train_accuracy * 100))
print("Validation Accuracy: {:.4f}%".format(last_val_accuracy * 100))
print("Loss {:.4f}%".format(last_loss * 100))

# Plotting
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.lineplot(x='Epoch', y='Train Accuracy', data=results_df, label='Train Accuracy')
sns.lineplot(x='Epoch', y='Validation Accuracy', data=results_df, label='Validation Accuracy')
sns.lineplot(x='Epoch', y='Loss', data=results_df, label='Loss')
plt.title('Training, Validation and Loss Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy and Loss')
plt.legend()
plt.show()