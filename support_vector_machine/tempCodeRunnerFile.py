employee = pd.read_csv('./Employee.csv')

employee.rename(columns={'JoiningYear': 'Join', 'PaymentTier': 'Position', 'EverBenched': 'UnemployedAtWork', 'ExperienceInCurrentDomain': 'Experience'}, inplace=True)

dataset = employee[['Education', 'Join', 'City', 'Position', 'Age', 'Gender', 'UnemployedAtWork', 'Experience', 'LeaveOrNot']]

from sklearn.preprocessing import LabelEncoder

Lb = LabelEncoder()

dataset['labelEducation'] = Lb.fit_transform(dataset['Education'])

dataset['labelUnemployedAtWork'] = Lb.fit_transform(dataset['UnemployedAtWork'])

X = dataset[['labelEducation', 'Age', 'Join', 'Position', 'labelUnemployedAtWork', 'Experience']]

y = dataset['LeaveOrNot']