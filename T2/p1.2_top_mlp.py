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

# Top-MLP
top_mlp= MLPClassifier(activation='tanh', hidden_layer_sizes=(30,50), solver='adam')
top_mlp.fit(X_train,y_train)
predicted = top_mlp.predict(X_test)

top_mlp_matrix = metrics.confusion_matrix(y_test, predicted)
top_mlp_report = metrics.classification_report(y_test, predicted,target_names=classes)
print(top_mlp_matrix)
print(top_mlp_report)

# top_mlp_acc= np.array((0.88,0.88,0.88,0.80,0.80,0.90,0.80,0.78,0.78,0.93))
# top_mlp_mac_F1 =np.array((0.68,0.76,0.61,0.62,0.64,0.84,0.64,0.71,0.68,0.70))
# top_mlp_w_F1= np.array((0.86,0.83,0.85,0.76,0.77,0.90,0.77,0.73,0.74,0.91))

# top_mlp_acc_ave = top_mlp_acc.mean()
# top_mlp_acc_std = top_mlp_acc.std()


# top_mlp_mac_F1_ave = top_mlp_mac_F1.mean()
# top_mlp_mac_F1_std = top_mlp_mac_F1.std()


# top_mlp_w_F1_ave = top_mlp_w_F1.mean()
# top_mlp_w_F1_std = top_mlp_w_F1.std()

# print('acc ave: ',top_mlp_acc_ave)
# print('mac F1 ave: ',top_mlp_mac_F1_ave)
# print('w F1 ave: ',top_mlp_w_F1_ave)
# print('acc std: ',top_mlp_acc_std)
# print('mac F1 std: ',top_mlp_mac_F1_std)
# print('w F1 std: ',top_mlp_w_F1_std)