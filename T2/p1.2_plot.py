import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv('../../drug200.csv')
classes = ['drugY','drugX','drugA','drugB','drugC']
# print(df.head())

# Generating instances distribution plot
classes_count = df['Drug'].value_counts()
values = classes_count.values
colors=['b','r','m','c','g']
plt.bar(classes,values, color=colors)
plt.title('Drugs distribution')
plt.savefig('drug-distribution.pdf',format='pdf')
plt.show()