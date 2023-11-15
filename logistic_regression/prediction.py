import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

employee = pd.read_csv('./Employee.csv')

employee.dropna()

employee.drop_duplicates(keep='first', inplace=True)

employee.rename(columns={'JoiningYear': 'Join', 'PaymentTier': 'Position', 'EverBenched': 'UnemployedAtWork', 'ExperienceInCurrentDomain': 'Experience'}, inplace=True)

dataset = employee[['Education', 'Join', 'City', 'Position', 'Age', 'Gender', 'UnemployedAtWork', 'Experience', 'LeaveOrNot']]

Lb = LabelEncoder()

dataset['labelEducation'] = Lb.fit_transform(dataset['Education'])

dataset['labelUnemployedAtWork'] = Lb.fit_transform(dataset['UnemployedAtWork'])

X = dataset[['labelEducation', 'Age', 'Join', 'Position', 'labelUnemployedAtWork', 'Experience']]

y = dataset['LeaveOrNot']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

def searchBestModel():
    global result_C, result_Penalty, result_Solver, result_Random_State

    param_grid = {
    'C': np.arange(0.1, 1.0),

    'penalty': ['l1', 'l2', 'elasticnet', 'none'],

    'solver': ['liblinear', 'saga'],

    'random_state': np.arange(2, 42, 2)
    }

    logisticRegression = LogisticRegression()

    grid_search = GridSearchCV(logisticRegression, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    print('\n')

    print(f"Best Params: {grid_search.best_params_} \n")

    print(f"Best Estimator: {grid_search.best_estimator_} \n")

    print(f"Best Score: {grid_search.best_score_} \n")

    print('\n')

    result_C = grid_search.best_params_['C']

    result_Penalty = grid_search.best_params_['penalty']
    
    result_Solver = grid_search.best_params_['solver']

    result_Random_State = grid_search.best_params_['random_state']

def ExecuteModel():
    logisticRegression = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=2)

    logisticRegression.fit(X_train, y_train)

    y_pred = logisticRegression.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    class_report = classification_report(y_test, y_pred)

    class_report_0_1 = classification_report(y_test, y_pred, target_names=['0', '1'], output_dict=True)

    accuracy_percent = accuracy * 100

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

    print("Accuracy Score Model Decision Tree: {:.2f}%".format(accuracy_percent))


# searchBestModel()

ExecuteModel()