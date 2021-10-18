import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics

df = pd.read_csv('../../drug200.csv')
classes = ['drugY','drugX','drugA','drugB','drugC']


df.BP = pd.Categorical(df.BP,['LOW','NORMAL','HIGH'],ordered=True)
df.BP = df.BP.cat.codes

df.Cholesterol = pd.Categorical(df.Cholesterol,['LOW','NORMAL','HIGH'],ordered=True)
df.Cholesterol = df.Cholesterol.cat.codes

df.Drug = pd.Categorical(df.Drug,classes)
df.Drug = df.Drug.cat.codes


# Converting the nominal values(Sex field) into numerical values
df = pd.get_dummies(df)

X = df[['Age','BP','Cholesterol','Na_to_K','Sex_F','Sex_M']]
y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Perceptron
clf= Perceptron()
clf.fit(X_train,y_train)
predicted = clf.predict(X_test)

clf_matrix = metrics.confusion_matrix(y_test, predicted)
clf_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(clf_matrix)
print(clf_report)

# per_acc= np.array((0.42,0.50,0.50,0.40,0.38,0.55,0.57,0.40,0.53,0.45))
# per_mac_F1 =np.array((0.27,0.21,0.29,0.33,0.23,0.14,0.26,0.11,0.19,0.36))
# per_w_F1= np.array((0.44,0.35,0.42,0.37,0.37,0.39,0.52,0.23,0.41,0.41))

# per_acc_ave = per_acc.mean()
# per_acc_std = per_acc.std()


# per_mac_F1_ave = per_mac_F1.mean()
# per_mac_F1_std = per_mac_F1.std()


# per_w_F1_ave = per_w_F1.mean()
# per_w_F1_std = per_w_F1.std()

# print('acc ave: ',per_acc_ave)
# print('mac F1 ave: ',per_mac_F1_ave)
# print('w F1 ave: ',per_w_F1_ave)
# print('acc std: ',per_acc_std)
# print('mac F1 std: ',per_mac_F1_std)
# print('w F1 std: ',per_w_F1_std)