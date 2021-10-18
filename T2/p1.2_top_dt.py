import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics,svm

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

clf = GridSearchCV(svm.SVC(gamma='auto'),{
  'C':[1,10],
  'kernel':['rbf','linear'],
}, cv=10,return_train_score=False)
clf.fit(X_train,y_train)

predicted = clf.predict(X_test)

clf_matrix = metrics.confusion_matrix(y_test, predicted)
clf_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(clf_matrix)
print('*****************************************')
print(clf_report)


# top_dt_acc= np.array((0.97,0.97,1.00,1.00,1.00,1.00,0.97,0.97,1.00,1.00))
# top_dt_mac_F1 =np.array((0.99,0.98,1.00,1.00,1.00,1.00,0.96,0.98,1.00,1.00))
# top_dt_w_F1= np.array((0.97,0.97,1.00,1.00,1.00,1.00,0.97,0.97,1.00,1.00))

# top_dt_acc_ave = top_dt_acc.mean()
# top_dt_acc_std = top_dt_acc.std()


# top_dt_mac_F1_ave = top_dt_mac_F1.mean()
# top_dt_mac_F1_std = top_dt_mac_F1.std()


# top_dt_w_F1_ave = top_dt_w_F1.mean()
# top_dt_w_F1_std = top_dt_w_F1.std()

# print('acc ave: ',top_dt_acc_ave)
# print('mac F1 ave: ',top_dt_mac_F1_ave)
# print('w F1 ave: ',top_dt_w_F1_ave)
# print('acc std: ',top_dt_acc_std)
# print('mac F1 std: ',top_dt_mac_F1_std)
# print('w F1 std: ',top_dt_w_F1_std)