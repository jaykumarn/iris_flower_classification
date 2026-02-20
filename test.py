import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('iris-with-answers.csv')
df['species'] = df.species.map({'setosa':0,'virginica':1,'versicolor':2})

X = df.iloc[:,:4]
y = df.iloc[:,[4]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DT = DecisionTreeRegressor(max_depth=4)
DT.fit(X_train, y_train)

pickle.dump(DT, open('model.pkl', 'wb'))
print("Model saved successfully!")