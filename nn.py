import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics

cutoff = 0.8
batch_size = 64
lr = 0.04
epochs = 20
train_data_to_use = 250000

data_dir = '/mnt/c/Users/Szelestey/projects/loan-prediction-based-on-customer-behavior/data'
version_no = 'v1.0'

df_train = pd.read_csv(data_dir + '/train.csv').loc[0:train_data_to_use,:]
df_test = pd.read_csv(data_dir + '/test.csv')
df_labels = pd.read_csv(data_dir + '/labels.csv')

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

print(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=109)

cb1 = ModelCheckpoint(data_dir + '/results/cnn' + version_no + '.h5', monitor="val_specificity", verbose=2,
                      save_best_only=True, mode="max")

opt = keras.optimizers.Adam(learning_rate=lr)

metr = [metrics.SpecificityAtSensitivity(cutoff, name='specificity'),
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.AUC(name='auc'),
        metrics.Recall(name='recall')]

model = keras.Sequential()
model.add(keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=opt,
              loss='BinaryCrossentropy',
              metrics=metr)

model.summary()

model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[cb1]
)
