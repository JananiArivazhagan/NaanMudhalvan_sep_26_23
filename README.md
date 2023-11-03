# NaanMudhalvan_sep_26_23
Naan Mudhalvan Project
# IMDb-Score-Prediction
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
data=pd.read_csv("IMDB.csv")
X = data.drop('IMDb_Score', axis=1)
y = data['IMDb_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
 data =label_encoder.fit_transform(data["sentiment"])
 print(data)
 file_data=pandas.get_dummies(data,columns=["sentiment"])
print(file_data)
data.replace({"positive":1,"negative":0},inplace=True)
print(data.head())
import numpy as np
Q1=data['IMDB Score'].quantile(0.25)
Q3=data['IMDB Score'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers=data[(data['IMDB Score']<lower_bound)|(data['IMDB Score']>upper_bound)]
print("Outliers in IMDB Scores:",outliers)
data_cleaned=data[(data['IMDB Score']>=lower_bound)&(data['IMDB Score']<=upper_bound)]
print(data_cleaned)
from sklearn.model_selection import train_test_split
x = data_cleaned[['Runtime', 'sentiment']]
y = data_cleaned['IMDB Score']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
data = {'Movie': ['Movie1', 'Movie2', 'Movie3'],
'Runtime': [90, 125, 160]}
df = pandas.DataFrame(data)
bins = [0, 100, 150, float('inf')]
labels = ['Short', 'Medium', 'Long']
df['Runtime_Category'] = pd.cut(df['Runtime'], bins=bins, labels=labels)
category_mapping = {'Short': 1, 'Medium': 2, 'Long': 3}
df['Runtime_Category'] = df['Runtime_Category'].map(category_mapping)
print(df)
from sklearn.model_selection import train_test_split
X = data[['sentiment','Runtime']] # Features
y = data['IMDB Score'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
Model Training:
odel.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)
