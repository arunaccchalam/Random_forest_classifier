import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



# data
df = pd.read_csv("penguins_size.csv")

df = df.dropna()
df.head()

# Train | Test Split

X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Random Forest Classification

help(RandomForestClassifier)

# Use 10 random trees
model = RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)
model.fit(X_train,y_train)

preds = model.predict(X_test)

# Evaluation

confusion_matrix(y_test,preds)

plot_confusion_matrix(model,X_test,y_test)

# Feature Importance

model.feature_importances_

# Choosing correct number of trees

let's explore if continually adding more trees improves performance...

test_error = []
for n in range(1,40):
    # Use n random trees
    model = RandomForestClassifier(n_estimators=n,max_features='auto')
    model.fit(X_train,y_train)
    test_preds = model.predict(X_test)
    test_error.append(1-accuracy_score(test_preds,y_test))
 

plt.plot(range(1,40),test_error,label='Test Error')
plt.legend()

Clearly there are diminishing returns, on such a small dataset, we've pretty much extracted all the information we can after about 5 trees.

# Random Forest - HyperParameter 

df = pd.read_csv("data_banknote_authentication.csv")
df.head()

# potting using seaborn
sns.pairplot(df,hue='Class')

X = df.drop("Class",axis=1)
y = df["Class"]

# splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# performing grid search
n_estimators=[64,100,128,200]
max_features= [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]

param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}  # Note, oob_score only makes sense when bootstrap=True!

rfc = RandomForestClassifier()
grid = GridSearchCV(rfc,param_grid)
grid.fit(X_train,y_train)

grid.best_params_

predictions = grid.predict(X_test)

print(classification_report(y_test,predictions))

plot_confusion_matrix(grid,X_test,y_test)

grid.best_estimator_.oob_score

grid.best_estimator_.oob_score_

# plotting errors and misclassifications
errors = []
misclassifications = []f

for n in range(1,64):
    rfc = RandomForestClassifier( n_estimators=n,bootstrap=True,max_features= 2)
    rfc.fit(X_train,y_train)
    preds = rfc.predict(X_test)
    err = 1 - accuracy_score(preds,y_test)
    n_missed = np.sum(preds != y_test) # watch the video to understand this line!!
    errors.append(err)
    misclassifications.append(n_missed)

plt.plot(range(1,64),errors)

plt.plot(range(1,64),misclassifications)

