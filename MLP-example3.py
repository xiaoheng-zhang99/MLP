import pandas as pd
import numpy as np
from joblib import dump
df = pd.read_csv('/Users/zhangxiaoheng/Desktop/heart_2020_cleaned.csv')
#print(df.head())
#可视化工具来观察数据缺失情况
#import missingno as msno
#msno.matrix(df)

df = df.replace('Yes', 1)
df = df.replace('No', 0)

df['Sex'] = df['Sex'].replace(['Female', 'Male'], [0, 1])

age_category_map = {
    '18-24': 1,
    '25-29': 2,
    '30-34': 3,
    '35-39': 4,
    '40-44': 5,
    '45-49': 6,
    '50-54': 7,
    '55-59': 8,
    '60-64': 9,
    '65-69': 10,
    '70-74': 11,
    '75-79': 12,
    '80 or older': 13
}
df['AgeCategory'] = df['AgeCategory'].map(age_category_map).round(0)

race_map = {
    'White': 0,
    'Hispanic': 1,
    'Black': 2,
    'Other': 3,
    'Asian': 4,
    'American Indian/Alaskan Native': 5
}
df['Race'] = df['Race'].map(race_map).round(0).astype(int)

gen_health_map = {
    'Poor': 1,
    'Fair': 2,
    'Good': 3,
    'Very good': 4,
    'Excellent': 5
}
df['GenHealth'] = df['GenHealth'].map(gen_health_map).round(0).astype(int)

df['Diabetic'] = df['Diabetic'].replace(['No, borderline diabetes', 'Yes (during pregnancy)'], [2,3])

df.describe()
print(df.head())
y = df['HeartDisease']
X = df.drop('HeartDisease', axis=1)
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(200, 100)).fit(X_train, y_train)

print(model.score(X_test, y_test))
dump(model, "train_model.m") #0.9120



