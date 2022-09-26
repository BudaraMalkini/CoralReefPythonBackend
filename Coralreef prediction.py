import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import seaborn as sns;
sns.set()

#Read the coral reef data set file
df = pd.read_csv('Datafile/CoralReef.csv')
df.head()

df.dtypes

df.describe()
#drop fisheries column
df.drop(['Fisheries change(MT)'], axis=1, inplace=True)
df.head(10)
#data types read
df.dtypes
#convert float data type to integer
for x in df:
    if df[x].dtypes == "int64":
        df[x] = df[x].astype(float)
        print(df[x].dtypes)
#add comumn to x,y points
df = df.select_dtypes(exclude=['object'])
df = df.fillna(df.mean())
X = df.drop('Coral Reef change(Km2)', axis=1)
y = df['Coral Reef change(Km2)']
#import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
# fit the regressor with x and y data
regressor = RandomForestRegressor(n_estimators=1000, random_state=42)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(X_test)
print(y_pred)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Calculate the absolute errors
errors = abs(y_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 95 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import seaborn as sns

plt.figure(figsize=(5, 7))

ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values", ax=ax)

plt.title('Actual vs Fitted Values for coral reef change')

plt.show()
plt.close()



# Pull out one tree from the forest
Tree = regressor.estimators_[5]
# Import tools needed for visualization


# Pull out one tree from the forest
Tree = regressor.estimators_[5]
# Export the image to a dot file
from sklearn import tree

plt.figure(figsize=(26, 15))
tree.plot_tree(Tree, filled=True,
               rounded=True,
               fontsize=14);

# Make pickle file of our model
pickle.dump(regressor, open("CoralReef.pkl", "wb"))

# Simple testing example
test = [[2025, 8]]
print(test)
sequence = np.array(test)
regressor.predict(test)
print(regressor.predict(test)[0])
