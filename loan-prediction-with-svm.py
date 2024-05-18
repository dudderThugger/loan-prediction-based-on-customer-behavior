from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

data_dir = 'C:\\Users\\Szelestey\\projects\\loan-prediction-based-on-customer-behavior\\data'

df_train = pd.read_csv(data_dir + '\\train.csv')
df_test = pd.read_csv(data_dir + '\\test.csv')
df_labels = pd.read_csv(data_dir + '\\labels.csv')

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Risk_Flag', axis=1), df_train['Risk_Flag'], test_size=0.2, random_state=109)

# Transforming textual attributes to numerical with LabelEncoder
for i, dtype in enumerate(x_train.dtypes):
    if dtype == 'object':
        le = LabelEncoder()
        x_train.iloc[:, i] = le.fit_transform(x_train.iloc[:,i])
        x_test.iloc[:, i] = le.fit_transform(x_test.iloc[:,i])

clf = svm.SVC(kernel='linear', verbose=False)

clf.fit(x_train.iloc[0:100, 0:6], y_train[0:100])

y_pred = clf.predict(x_test.iloc[0:100, 0:6])

print("Accuracy:", metrics.accuracy_score(y_test[0:100], y_pred))
