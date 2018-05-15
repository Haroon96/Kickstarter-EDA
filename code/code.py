import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from dateutil.parser import parse as parse_date
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print("Reading data set")
df = pd.read_csv('ks-projects-201801.csv')

print("Removing invalid country")
df = df[df['country'] != 'N,0"']

print("Reducing to binary states")
df = df[(df['state'] == 'failed') | (df['state'] == 'successful')]

print("Removing outliers")
q1 = df['goal'].quantile(0.25)
q3 = df['goal'].quantile(0.75)
iqr = q3 - q1
df = df.query('(@q1 - 1.5 * @iqr) <= goal <= (@q3 + 1.5 * @iqr)')

q1 = df['backers'].quantile(0.25)
q3 = df['backers'].quantile(0.75)
iqr = q3 - q1
df = df.query('(@q1 - 1.5 * @iqr) <= backers <= (@q3 + 1.5 * @iqr)')

print("Removing unnecessary fields")
df = df.drop(columns=['usd_pledged_real', 'usd_goal_real', 'name', 'category'])

print("Appending 'duration' field")
tmp = df.apply(lambda row : (parse_date(row['deadline']) - parse_date(row['launched'])).days + 1, axis=1)
df = df.join(tmp.rename('duration'))

print("Converting non-numeric fields to numeric using one-hot label encoding")
tmp = pd.get_dummies(df, columns=['main_category'])
tmp.head()
category_features = ('main_category_' + df['main_category'].unique()).tolist()
df = tmp

def getTrainingSet(df, features):
    # 0 to 80% of the data set
    train_set = df[:math.ceil(len(df) * 0.7)]
    # return features and labels
    return train_set[features],  train_set['state'].map({ 'failed': 0, 'successful': 1 })


def getTestSet(df, features):
    # 80% to 100% of the data set
    test_set = df[math.ceil(len(df) * 0.7):]
    return test_set[features], test_set['state'].map({ 'failed': 0, 'successful': 1 })

def trainModel(mdl, x, y):
    mdl.fit(x, y)

def accuracyScore(mdl, x, y):
    y_predicted = mdl.predict(x)
    return accuracy_score(y, y_predicted)
    
def runClassifier(mdl, df, features):
    trainSetX, trainSetY = getTrainingSet(df, features)
    testSetX, testSetY = getTestSet(df, features)
    trainModel(mdl, trainSetX, trainSetY)
    return accuracyScore(mdl, testSetX, testSetY)
    
def tryModel(mdl):
    print('%-50s\tAccuracy' % '[Features]')
    print('%-50s\t%.2f' % ('[goal]', runClassifier(mdl, df, ['goal'])))
    print('%-50s\t%.2f' % ('[duration]', runClassifier(mdl, df, ['duration'])))
    print('%-50s\t%.2f' % ('[main_category]', runClassifier(mdl, df, category_features)))
    print('%-50s\t%.2f' % ('[goal, duration]', runClassifier(mdl, df, ['goal', 'duration'])))
    print('%-50s\t%.2f' % ('[goal, duration, categories]', runClassifier(mdl, df, ['goal', 'duration'] + category_features)))
    print('%-50s\t%.2f' % ('[backers]', runClassifier(mdl, df, ['backers'])))
    print('%-50s\t%.2f' % ('[usd pledged]', runClassifier(mdl, df, ['usd pledged'])))
    print('%-50s\t%.2f' % ('[backers, usd pledged]', runClassifier(mdl, df, ['backers', 'usd pledged'])))
    print('%-50s\t%.2f' % ('[backers, usd pledged, goal]', runClassifier(mdl, df, ['backers', 'usd pledged', 'goal'])))
    print('%-50s\t%.2f' % ('[backers, usd pledged, goal, duration]', runClassifier(mdl, df, ['backers', 'usd pledged', 'goal', 'duration'])))
    print('%-50s\t%.2f' % ('[backers, usd pledged, goal, duration, categories]', runClassifier(mdl, df, ['backers', 'usd pledged', 'goal', 'duration'] + category_features)))


print("\nTrying logistic regression")
tryModel(LogisticRegression())


print("\nTrying decision tree classification")
tryModel(DecisionTreeClassifier())

print("\nTrying random forest classification")
tryModel(RandomForestClassifier())