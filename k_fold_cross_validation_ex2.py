""" k-fold cross validation example 2:
    Multiple-factor linear regression.
    Using preprocessing.
"""

# import necessary python modules and classes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression # your model here
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import pandas as pd
import numpy as np

# import data
df = pd.read_csv('data/advertising.csv')

# engineer additional feature of interest
df['TVxradio'] = df['TV'] * df['radio']

# slice data
X = np.array(df[['TV', 'radio', 'newspaper', 'TVxradio']])
y = np.array(df['sales'])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=5)

# setup pipeline with preprocessing transformers
lr = make_pipeline(preprocessing.StandardScaler(), LinearRegression())

# k-fold cross validation
scores = cross_val_score(lr, X_train, y_train, cv=10)
print(scores)
print(
    f"95% CI Accuracy: "
    f"{round(scores.mean(), 2)} "
    f"(+/- {round(scores.std() * 2, 2)})"
)

# test model on test set
model = LinearRegression().fit(X_train, y_train)
print(f'model accuracy on test set: {model.score(X_test, y_test)}')