import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier , MLPRegressor
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

# MLP
mlp= MLPClassifier(activation='logistic',hidden_layer_sizes=100, max_iter=500)
mlp.fit(X_train,y_train)
predicted = mlp.predict(X_test)

mlp_matrix = metrics.confusion_matrix(y_test, predicted)
mlp_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(mlp_matrix)
print(mlp_report)

# mlp_acc= np.array((0.53,0.84,0.97,0.85,0.88,0.90,0.93,0.88,0.95,0.88))
# mlp_mac_F1 =np.array((0.32,0.66,0.99,0.85,0.69,0.88,0.85,0.80,0.92,0.76))
# mlp_w_F1= np.array((0.42,0.78,0.97,0.91,0.85,0.90,0.72,0.87,0.95,0.85))

# mlp_acc_ave = mlp_acc.mean()
# mlp_acc_std = mlp_acc.std()


# mlp_mac_F1_ave = mlp_mac_F1.mean()
# mlp_mac_F1_std = mlp_mac_F1.std()


# mlp_w_F1_ave = mlp_w_F1.mean()
# mlp_w_F1_std = mlp_w_F1.std()

# print('acc ave: ',mlp_acc_ave)
# print('mac F1 ave: ',mlp_mac_F1_ave)
# print('w F1 ave: ',mlp_w_F1_ave)
# print('acc std: ',mlp_acc_std)
# print('mac F1 std: ',mlp_mac_F1_std)
# print('w F1 std: ',mlp_w_F1_std)