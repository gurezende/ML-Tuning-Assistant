#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


## Creating an evaluation function
#%%

def regression_model_summary(pipe, X_test, y_test):
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
    * Score: R^2 score of the model.
    * RMSE: Root mean squared error of the model.
    * Intercept: Intercept of the model.
    * Coefficients: A pandas DataFrame with feature names and their coefficients.
    * VIF: Variance inflation factor (VIF) of the model.
    """

    # Regression Summary
    features = pipe.named_steps['encoder'].get_feature_names_out()
    score = pipe.score(X_test, y_test)
    rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
    intercept = pipe.named_steps['model'].intercept_
    coefs = pipe.named_steps['model'].coef_

    # Dataframe
    summary = pd.DataFrame({
        'feature': features,
        'coefficient': coefs
    })

    # Print
    print(f"Score: {score:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print('\n')
    print("Coefficients:")
    print(summary)

    print('\n')
    # VIF
    print("VIF:")
    corr_mat = np.array( df.select_dtypes(exclude=['category']).corr() )
    inv_corr_mat = np.linalg.inv(corr_mat)
    print(pd.Series(np.diag(inv_corr_mat), index=df.select_dtypes(exclude=['category']).columns))


# %%

## Baseline Model
# Data
df = sns.load_dataset('tips')

# Train Test Split
X = df.drop('tip', axis=1)
y = df['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
cat_vars = df.select_dtypes(include=['object']).columns

# Pipeline
pipe = Pipeline([
    ('encoder', OneHotEncoder(variables=['sex', 'smoker', 'day', 'time'],
                              drop_last=True)),
    ('model', LinearRegression())
])

# Fit
pipe.fit(X_train, y_train)



# %%

# Evaluating the basic model
print(df.describe(include='all'))
print("\n---\n")
regression_model_summary(pipe, X_test, y_test)

#%%

## Tuning the Model using the Agent's suggestions

# Data
df = sns.load_dataset('tips')

# Clip total_bill
df['total_bill'] = df['total_bill'].clip(upper=np.quantile(df['total_bill'], 0.95))

# Transformation
df[['total_bill','tip']] = np.log(df[['total_bill','tip']])


# Train Test Split
X = df.drop(['tip', 'time'], axis=1)
y = df['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
cat_vars = df.select_dtypes(include=['object']).columns

# Pipeline
pipe = Pipeline([
    ('encoder', OneHotEncoder(variables=['sex', 'smoker', 'day'],
                              drop_last=True)),
    ('model', LinearRegression())
])

# Fit
pipe.fit(X_train, y_train)

# Evaluating the tuned model
regression_model_summary(pipe, X_test, y_test)


## Trying Random Forest Regressor
# %%

## Tuning the Model using the Agent's suggestions

from sklearn.ensemble import RandomForestRegressor

# Data
df = sns.load_dataset('tips')

# Train Test Split
X = df.drop(['tip'], axis=1)
y = df['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical
cat_vars = df.select_dtypes(include=['object']).columns

# Pipeline
pipe = Pipeline([
    ('encoder', OneHotEncoder(variables=['sex', 'smoker', 'day', 'time'],
                              drop_last=True)),
    ('model', RandomForestRegressor(random_state=42))
])

# Fit
pipe.fit(X_train, y_train)

# Evaluating the tuned model
print(f'Score: {pipe.score(X_test, y_test)}')
print(f'RMSE: {root_mean_squared_error(y_test, pipe.predict(X_test))}')


# %%
