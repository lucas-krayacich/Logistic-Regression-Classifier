
#Logistic Regression Classifier full implementation
#Created by Lucas Krayacich, Sam Chan, Tanner Ozcan

#---------------------------------------------------------------------------------------------------------------

#The following program is used to discern walking from jumping in the input CSV files of accelerometer data

import pandas as pd
import numpy as np
import matplotlib as mpl

import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import h5py
import math
import pickle
import joblib

mpl.use('TkAgg')  # !IMPORTANT


# Denoising function
def noise_removal(df, window_size):
    denoised = df.copy()

    denoised = denoised.iloc[:, 0:-1].rolling(window=window_size).mean()

    denoised["label"] = df.iloc[:, -1]

    denoised = denoised.dropna()

    return denoised


# feature extraction function
def extract_features(df, window_size):
    features = pd.DataFrame(
        columns=['maximum', 'minimum', 'range', 'mean', 'median', 'variance', 'skewness', 'std', 'kurtosis', 'quantile',
                 'label'])
    # the label is on the last column
    features['label'] = df.iloc[:, -1]
    # extracting features from absolute acceleration which is the second last column
    features['maximum'] = df.iloc[:, -2].rolling(window=window_size).max()
    features['minimum'] = df.iloc[:, -2].rolling(window=window_size).min()
    features['range'] = features['maximum'] - features['minimum']
    features['mean'] = df.iloc[:, -2].rolling(window=window_size).mean()
    features['median'] = df.iloc[:, -2].rolling(window=window_size).median()
    features['variance'] = df.iloc[:, -2].rolling(window=window_size).var()
    features['skewness'] = df.iloc[:, -2].rolling(window=window_size).skew()
    features['std'] = df.iloc[:, -2].rolling(window=window_size).std()
    features['kurtosis'] = df.iloc[:, -2].rolling(window=window_size).kurt()
    features['quantile'] = df.iloc[:, -2].rolling(window=window_size).quantile(0.8)

    features = features.dropna()
    return features


# load the data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# initialize all member data into their own dataframes --- from here they will be stored in hdf5 files and combined for preprocessing
LucasJumpRight = pd.read_csv('luc jump right.csv')
LucasJumpLeft = pd.read_csv('luc jump left.csv')
LucasWalkLeft = pd.read_csv('luc walk left.csv')
LucasWalkRight = pd.read_csv('luc walk right.csv')

TannerJumpRight = pd.read_csv('tan jump right.csv')
TannerJumpRight = TannerJumpRight.drop(TannerJumpRight.index[::2])
TannerJumpRight.reset_index(drop=True)

TannerJumpLeft = pd.read_csv('tan jump left.csv')
TannerJumpLeft = TannerJumpLeft.drop(TannerJumpLeft.index[::2])
TannerJumpLeft.reset_index(drop=True)

TannerWalkLeft = pd.read_csv('tan walk left.csv')
TannerWalkLeft = TannerWalkLeft.drop(TannerWalkLeft.index[::2])
TannerWalkLeft.reset_index(drop=True)

TannerWalkRight = pd.read_csv('tan walk right.csv')
TannerWalkRight = TannerWalkRight.drop(TannerWalkRight.index[::2])
TannerWalkRight.reset_index(drop=True)

SamJumpLeft = pd.read_csv('sam jump left.csv')
SamJumpLeft = SamJumpLeft.drop(SamJumpLeft.index[::2])
SamJumpLeft.reset_index(drop=True)

SamJumpRight = pd.read_csv('sam jump right.csv')
SamJumpRight = SamJumpRight.drop(SamJumpRight.index[::2])
SamJumpRight.reset_index(drop=True)

SamWalkLeft = pd.read_csv('sam walk left.csv')
SamWalkLeft = SamWalkLeft.drop(SamWalkLeft.index[::2])
SamWalkLeft.reset_index(drop=True)

SamWalkRight = pd.read_csv('sam walk right.csv')
SamWalkRight = SamWalkRight.drop(SamWalkRight.index[::2])
SamWalkRight.reset_index(drop=True)

# Concatenate all datasets together (name of variable is arbitrary)
LucasJumpLeft = pd.concat([LucasJumpLeft, SamJumpLeft, TannerJumpLeft])
LucasJumpRight = pd.concat([LucasJumpRight, SamJumpRight, TannerJumpRight])
LucasWalkLeft = pd.concat([LucasWalkLeft, SamWalkLeft, TannerWalkLeft])
LucasWalkRight = pd.concat([LucasWalkRight, SamWalkRight, TannerWalkRight])

print(LucasJumpLeft.info())


# combine the jump and walk data
combinedJump = pd.concat([LucasJumpLeft, LucasJumpRight])
combinedWalk = pd.concat([LucasWalkLeft, LucasWalkRight])

# noise removal
combinedJumpSma200 = noise_removal(combinedJump, 100)
combinedWalkSma200 = noise_removal(combinedWalk, 100)

# drop the time column
combinedJumpSma200 = combinedJumpSma200.drop(combinedJumpSma200.columns[0], axis=1)
combinedWalkSma200 = combinedWalkSma200.drop(combinedWalkSma200.columns[0], axis=1)

# extracting features
combinedJumpSma200Features = extract_features(combinedJumpSma200, 500)
combinedWalkSma200Features = extract_features(combinedWalkSma200, 500)

# combining the 2 feature dataframes
combinedfeatures = pd.concat([combinedJumpSma200Features, combinedWalkSma200Features])

# separating the lable and data
labels = combinedfeatures.iloc[:, -1]
data = combinedfeatures.iloc[:, 0:-1]

##########################################################################
# train test split with 10% for testing
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)
# normalization function
scaler = StandardScaler()
# logistic regression
l_reg = LogisticRegression(max_iter=10000)
# making the pipeline
clf = make_pipeline(StandardScaler(), l_reg)
# training
clf.fit(X_train, Y_train)
# obtaining the predictions and the probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
print('y_pred is:', y_pred)
print('y_clf_prob is:', y_clf_prob)

# obtaining the classification accuracy
acc = accuracy_score(Y_test, y_pred)
print('accuracy is: ', acc)

# obtaining the classification recall
recall = recall_score(Y_test, y_pred)
print('recall is: ', recall)

# plotting the confusion matrix
cm = confusion_matrix(Y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# plotting the roc curve
fpr, tpr, _ = roc_curve(Y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# calculating the auc
auc = roc_auc_score(Y_test, y_clf_prob[:, 1])
print('the auc is: ', auc)

# f1 score
f1 = f1_score(Y_test, y_pred)
print('the f1 score is: ', f1)
#################################

# https://scikit-learn.org/stable/model_persistence.html
# classified = pickle.dumps(clf)
joblib.dump(clf, 'classifier.joblib')

# store them in hdf file
with h5py.File("./data.h5", "w") as hdf:
    LucasData = hdf.create_group("Lucas")
    SamData = hdf.create_group("Sam")
    TannerData = hdf.create_group("Tanner")
    GroupData = hdf.create_group("dataset")

    LucasData.create_dataset('Lucas jump left', data=LucasJumpLeft, compression='gzip', compression_opts=7)
    LucasData.create_dataset('Lucas jump right', data=LucasJumpRight, compression='gzip', compression_opts=7)
    LucasData.create_dataset('Lucas walk left', data=LucasWalkLeft, compression='gzip', compression_opts=7)
    LucasData.create_dataset('Lucas walk right', data=LucasWalkRight, compression='gzip', compression_opts=7)

    SamData.create_dataset('Sam jump left pocket', data=SamJumpLeft, compression='gzip', compression_opts=7)
    SamData.create_dataset('Sam jump right pocket', data=SamJumpRight, compression='gzip', compression_opts=7)
    SamData.create_dataset('Sam walk left pocket', data=SamWalkLeft, compression='gzip', compression_opts=7)
    SamData.create_dataset('Sam walk right pocket', data=SamWalkRight, compression='gzip', compression_opts=7)

    TannerData.create_dataset('Tanner jump left pocket', data=TannerJumpLeft, compression='gzip', compression_opts=7)
    TannerData.create_dataset('Tanner jump right pocket', data=TannerJumpRight, compression='gzip', compression_opts=7)
    TannerData.create_dataset('Tanner walk left pocket', data=TannerWalkLeft, compression='gzip', compression_opts=7)
    TannerData.create_dataset('Tanner walk right pocket', data=TannerWalkRight, compression='gzip', compression_opts=7)

    GroupData.create_dataset('x_train', data=X_train, compression='gzip', compression_opts=7)
    GroupData.create_dataset('x_test', data=X_test, compression='gzip', compression_opts=7)
    GroupData.create_dataset('y_train', data=Y_train, compression='gzip', compression_opts=7)
    GroupData.create_dataset('y_test', data=Y_test, compression='gzip', compression_opts=7)