""" k-fold cross validation example 1:
    One-factor linear regression.
    No preprocessing needed.
"""

# import necessary python modules and classes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression # your model here
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# import data
df = pd.read_csv('data/advertising.csv')

# engineer feature of interest
df['TVxradio'] = df['TV'] * df['radio']

# slice data
x = np.array(df['TVxradio']).reshape(-1, 1)
y = np.array(df['sales'])

# train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=0)

# select model
model = LinearRegression()

# k-fold cross validation
scores = cross_val_score(model, x_train, y_train, cv=10)
print(scores)
print(
    f"95% CI Accuracy: "
    f"{round(scores.mean(), 2)} "
    f"(+/- {round(scores.std() * 2, 2)})"
)

# test model on test set
model.fit(x_train, y_train)
print(f'model accuracy on test set: {model.score(x_test, y_test)}')