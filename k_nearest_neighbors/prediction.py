import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

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

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def searchBestModel():
    param_grid = {
        'n_neighbors': np.arange(1, 11, 1),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    print('\n')
    print(f"Best Params: {grid_search.best_params_} \n")

    print(f"Best Estimator: {grid_search.best_estimator_} \n")

    print(f"Best Score: {grid_search.best_score_} \n")
    print('\n')

def executeModel():
    knn = KNeighborsClassifier(algorithm='brute', n_neighbors=9, weights='uniform')

    knn.fit(X_train, y_train)

    y_predict = knn.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_test, y_predict)

    conf_matrix = confusion_matrix(y_test, y_predict)

    class_report = classification_report(y_test, y_predict)

    class_report_0_1 = classification_report(y_test, y_predict, target_names=['0', '1'], output_dict=True)

    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    print("\n")

    print("True Positive (TP) Jumlah karyawan yang benar-benar mengundurkan diri dan diprediksi mengundurkan diri: ", TP)
    print("True Negative (TN) Jumlah karyawan yang benar-benar tidak mengundurkan diri dan diprediksi tidak mengundurkan diri: ", TN)
    print("False Positive (FP) Jumlah karyawan yang tidak mengundurkan diri tetapi diprediksi mengundurkan diri: ", FP)
    print("False Negative (FN) Jumlah karyawan yang mengundurkan diri tetapi diprediksi tidak mengundurkan diri: ", FN)

    print("\n")

    print(class_report)

    print("\n")

    support_0 = class_report_0_1['0']['support']
    support_1 = class_report_0_1['1']['support']

    print("Class 0 (Will Leave):", support_0)
    print("Class 1 (Not):", support_1)

    print("\n")

    accuracy_percent = accuracy * 100

    print("Accuracy Score Model KNN: {:.2f}%".format(accuracy_percent))


# searchBestModel()

executeModel()