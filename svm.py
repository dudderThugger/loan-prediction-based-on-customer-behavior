from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import metrics

data_dir = '/mnt/c/Users/Szelestey/projects/loan-prediction-based-on-customer-behavior/data'

df_train = pd.read_csv(data_dir + '/train_encoded.csv')
df_test = pd.read_csv(data_dir + '/test.csv')
df_labels = pd.read_csv(data_dir + '/labels.csv')
## 'Id', 'Income', 'Age', 'Experience', 'Married_labeled', 'House_labeled', 'Car_labeled', 'Profession', 'City_labeled', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'Risk_Flag'
attribs_to_use = ['Id', 'Risk_Flag','Income', 'Age', 'Experience', 'Married_labeled', 'House_labeled', 'Car_labeled', 'Profession', 'City_labeled', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

data_count = 100000

df_train = df_train.sample(data_count)[attribs_to_use]

X = df_train.drop(['Id', 'Risk_Flag'], axis=1)
y = df_train['Risk_Flag']

# Transforming textual attributes to numerical with LabelEncoder
for i, dtype in enumerate(X.dtypes):
    if dtype == 'object':
        le = LabelEncoder()
        X.iloc[:, i] = le.fit_transform(X.iloc[:, i])
        X.iloc[:, i] = le.fit_transform(X.iloc[:, i])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)


clf = svm.LinearSVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred))
# print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
