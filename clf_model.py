#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, classification_report


## Creating an evaluation function
#%%

def clf_model_summary(pipe, X_test, y_test):
    """
    Summary of a linear regression model.

    Parameters
    ----------
    pipe : Pipeline
        A pipeline containing a OneHotEncoder and a LinearRegression model.
    X_test : pd.DataFrame
        Testing data.
    y_test : pd.Series
        Target variable.

    Returns
    -------
    None

    Prints
    -------
    * Score: accuracy score of the model.
    * RMSE: Root mean squared error of the model.
    * Classification Report: Classification report of the model.
    """

    # Regression Summary
    features = pipe.named_steps['encoder'].get_feature_names_out()
    score = pipe.score(X_test, y_test)
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))

    # Print
    print(f"Score: {score:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(classification_report(y_test, pipe.predict(X_test)))

# %%

## Baseline Model
# Data
df = sns.load_dataset('tips')

# Label Encoding
df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})

# Train Test Split
X = df.drop('smoker', axis=1)
y = df['smoker']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
cat_vars = df.select_dtypes(include=['object']).columns

# Pipeline
pipe = Pipeline([
    ('encoder', OneHotEncoder(variables=['sex', 'day', 'time'],
                              drop_last=True)),
    ('model', RandomForestClassifier())
])

# Fit
pipe.fit(X_train, y_train)



# %%

# Evaluating the basic model
# print(df.describe(include='all'))
# print("\n---\n")
clf_model_summary(pipe, X_test, y_test)

#%%

## Tuning the Model using the Agent's suggestions

# Data
df = sns.load_dataset('tips')

# Label Encoding
df['smoker'] = df['smoker'].map({'Yes': 1, 'No': 0})

# Train Test Split
X = df.drop('smoker', axis=1)
y = df['smoker']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
cat_vars = df.select_dtypes(include=['object']).columns

# Pipeline
tuned_pipe = Pipeline([
    ('encoder', OneHotEncoder(variables=['sex', 'day', 'time'],
                              drop_last=True)),
    #uncomment to use Bayes Search and comment out the model

    # ('bayes_search', BayesSearchCV(RandomForestClassifier(),
    #                                      search_spaces={
    #                                          'class_weight': Categorical(['balanced', 'balanced_subsample']),
    #                                          'n_estimators': Integer(100, 500),
    #                                          'max_depth': Integer(2, 6),
    #                                          'min_samples_split': Integer(2, 6),
    #                                          'min_samples_leaf': Integer(1, 4) },
    #                                       n_iter=10,
    #                                       cv=5,
    #                                       n_jobs=-1,
    #                                       random_state=12,
    #                                       scoring='accuracy')),
    
    ('model', RandomForestClassifier(n_estimators=500, 
                                     max_depth=4,
                                     class_weight='balanced',
                                     min_samples_split=4, 
                                     min_samples_leaf=4))
])

# Fit
tuned_pipe.fit(X_train, y_train)

#%%
# Evaluating the Tuned  model
clf_model_summary(tuned_pipe, X_test, y_test)
# %%
