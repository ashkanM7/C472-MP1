import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


df = pd.read_csv('../../drug200.csv')
classes = ['drugY','drugX','drugA','drugB','drugC']
# print(df.head())


df.BP = pd.Categorical(df.BP,['LOW','NORMAL','HIGH'],ordered=True)
df.BP = df.BP.cat.codes

df.Cholesterol = pd.Categorical(df.Cholesterol,['LOW','NORMAL','HIGH'],ordered=True)
df.Cholesterol = df.Cholesterol.cat.codes

df.Drug = pd.Categorical(df.Drug,classes)
df.Drug = df.Drug.cat.codes
print(df.head())
print(df.Drug)

# Converting the Sex field to numerical values
df = pd.get_dummies(df)

X = df[['Age','BP','Cholesterol','Na_to_K','Sex_F','Sex_M']]
y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# GuassianNB
GN= GaussianNB()
GN.fit(X_train,y_train)
predicted = GN.predict(X_test)


clf_matrix = metrics.confusion_matrix(y_test, predicted)
clf_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(clf_matrix)
print(clf_report)

# GN_acc= np.array((0.88,0.78,0.85,0.88,0.78,0.88,0.88,0.90,0.90,0.80))
# GN_mac_F1 =np.array((0.83,0.71,0.83,0.86,0.84,0.85,0.89,0.90,0.73))
# GN_w_F1= np.array((0.88,0.79,0.85,0.88,0.76,0.86,0.90,0.90,0.81))

# GN_acc_ave = GN_acc.mean()
# GN_acc_std = GN_acc.std()
# print('acc ave: ',GN_acc_ave)
# print('acc std: ',GN_acc_std)

# GN_mac_F1_ave = GN_mac_F1.mean()
# GN_mac_F1_std = GN_mac_F1.std()
# print('mac F1 ave: ',GN_mac_F1_ave)
# print('mac F1 std: ',GN_mac_F1_std)


# GN_w_F1_ave = GN_w_F1.mean()
# GN_w_F1_std = GN_w_F1.std()
# print('w F1 ave: ',GN_w_F1_ave)
# print('w F1 std: ',GN_w_F1_std)

