import os
import io
import numpy
import matplotlib.pyplot as plt 
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                lines.append(line)
            f.close()
            message = ''.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'massage': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('./BBC/business', 'business'))
data = data.append(dataFrameFromDirectory('./BBC/entertainment', 'entertainment'))
data = data.append(dataFrameFromDirectory('./BBC/politics', 'politics'))
data = data.append(dataFrameFromDirectory('./BBC/sport', 'sport'))
data = data.append(dataFrameFromDirectory('./BBC/tech', 'tech'))

instances_count = data['class'].value_counts()
values = instances_count.values
classes = ['business','entertainment','politics','sport','tech']
colors=['b','r','m','c','g']
plt.bar(classes,values, color=colors)
plt.title('BBC news distribution')
plt.savefig('bbc-distribution.pdf',format='pdf')



plt.show()