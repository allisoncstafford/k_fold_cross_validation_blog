# import necessary python modules and classes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('data/advertising.csv') # import data
df['TVxradio'] = df['TV'] * df['radio'] # create feature
x = np.array(df['TVxradio']).reshape(-1, 1)
y = np.array(df['sales'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=54546)
model = LinearRegression().fit(x_train, y_train)
print(f'model accuracy on test set: {model.score(x_test, y_test)}')