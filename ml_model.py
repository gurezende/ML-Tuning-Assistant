#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

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


##### Evaluation #######
#%%

# Regression Summary
score = pipe.score(X_test, y_test)
rmse = root_mean_squared_error(y_test, pipe.predict(X_test))
intercept = pipe.named_steps['model'].intercept_
coefs = pipe.named_steps['model'].coef_

# Dataframe
summary = pd.DataFrame({
    'feature': pipe.named_steps['encoder'].get_feature_names_out(),
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

# Feature engineering
# Total Bill per size
# df['total_bill_per_size'] = df['total_bill'] / df['size']

# # Train Test Split
# X = df.drop('tip', axis=1)
# y = df['tip']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Categorical
# cat_vars = df.select_dtypes(include=['object']).columns

# # Pipeline
# pipe = Pipeline([
#     ('encoder', OneHotEncoder(variables=['sex', 'smoker', 'day', 'time'],
#                               drop_last=True)),
#     ('model', LinearRegression())
# ])

# # Fit
# pipe.fit(X_train, y_train)

# %%

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

# %%
