import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


df = pd.read_csv('../../drug200.csv')
classes = ['drugY','drugX','drugA','drugB','drugC']


df.BP = pd.Categorical(df.BP,['LOW','NORMAL','HIGH'],ordered=True)
df.BP = df.BP.cat.codes

df.Cholesterol = pd.Categorical(df.Cholesterol,['LOW','NORMAL','HIGH'],ordered=True)
df.Cholesterol = df.Cholesterol.cat.codes

df.Drug = pd.Categorical(df.Drug,classes)
df.Drug = df.Drug.cat.codes

# Converting the Sex field to numerical values
df = pd.get_dummies(df)

X = df[['Age','BP','Cholesterol','Na_to_K','Sex_F','Sex_M']]
y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

clf = DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=4)
clf.fit(X_train,y_train)
predicted = clf.predict(X_test)


clf_matrix = metrics.confusion_matrix(y_test, predicted)
clf_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(clf_matrix)
print('*****************************************')
print(clf_report)


# base_dt_acc= np.array((1.00,1.00,0.95,0.97,0.97,1.00,1.00,0.97,0.97,0.97))
# base_dt_mac_F1 =np.array((1.00,1.00,0.92,0.99,0.96,1.00,1.00,0.94,0.95,0.98))
# base_dt_w_F1= np.array((1.00,1.00,0.95,0.97,0.97,1.00,1.00,0.97,0.97,0.97))

# base_dt_acc_ave = base_dt_acc.mean()
# base_dt_acc_std = base_dt_acc.std()


# base_dt_mac_F1_ave = base_dt_mac_F1.mean()
# base_dt_mac_F1_std = base_dt_mac_F1.std()


# base_dt_w_F1_ave = base_dt_w_F1.mean()
# base_dt_w_F1_std = base_dt_w_F1.std()

# print('acc ave: ',base_dt_acc_ave)
# print('mac F1 ave: ',base_dt_mac_F1_ave)
# print('w F1 ave: ',base_dt_w_F1_ave)
# print('acc std: ',base_dt_acc_std)
# print('mac F1 std: ',base_dt_mac_F1_std)
# print('w F1 std: ',base_dt_w_F1_std)
