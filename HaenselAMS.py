##### libraries #####
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
#from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn import set_config
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                   PolynomialFeatures, PowerTransformer,
                                   StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from xgboost import XGBRegressor

##### exploratoy data analysis #####
# import data
sample = pd.read_csv('datasets/sample.csv')

# head of data
sample.head()

# data types of columns
def df_datatypes(df):
    df_desc = pd.DataFrame(df.dtypes.value_counts().reset_index())
    df_desc.columns = ['Data Type', 'Count']
    return df_desc.sort_values('Count', ascending=False)

df_datatypes(sample)

# dataframe missing
def df_missing_info(df):
    pd.set_option('display.max_columns', df.shape[1])
    pd.set_option('display.max_rows', df.shape[1])
    descriptive_df = pd.DataFrame()
    descriptive_df['column'] = df.columns
    descriptive_df['# missing'] = [df[col].isnull().sum() for col in df]
    descriptive_df['% missing'] = np.round(descriptive_df['# missing'] / df.shape[0], 4)
    return descriptive_df

df_missing_info(sample)

# numerical - correlogram
def graph_correlogram(df):
    sns.set_theme(style="white") 
    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
# correlation matrix with p-values
def df_correlation_matrix(df):
    numerical = df.select_dtypes(include=['int64', 'float']).columns.tolist()
    print(f'Pearson Correlation Matrix with P-Values')
    print(f'[Coef in Btm Tri / p-Values in Up Tri]')
    print(f'*** for <0.001, ** for <0.01, * for <0.05')
    print(f'-----------------------------------------')
    return df[numerical].rcorr(method='pearson').round(3)
      
graph_correlogram(sample)
df_correlation_matrix(sample)

# categorical - describe
def df_describe_categorical(df):
    return df.select_dtypes(include=['object']).describe()
    
df_describe_categorical(sample)

# graph of all numeric data types
def graph_numeric_histograms(df):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    numeric_columns = df.select_dtypes(include=np.number)
    for col in numeric_columns:
        sns.histplot(df[col], kde=True, color='black')
        plt.title(f'Histogram of {col.title()}')
        plt.show()
    
graph_numeric_histograms(sample)
        
def df_anova_stats(num, cat, df):
    # Set the default Pandas float precision to 3 decimals
    pd.set_option("display.precision", 3)
    
    # boxplot
    sns.boxplot(x=cat, y=num, palette='colorblind', data=df, width=0.2, fliersize=1)
    plt.title('Boxplot of {} by {}'.format(num.capitalize(), cat.capitalize()))
    plt.ylabel(num.capitalize())
    plt.xlabel(cat.capitalize())
    plt.show()

    # mean, std, cont
    print(f'Mean, Standard Deviation, Count')
    display(df.groupby(cat)[num].agg(['mean', 'std', 'count']).round(2))
    
# filter warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# anova df
def get_anova_df(target, df):
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    anova_df = pd.DataFrame()
    for cat in categorical_columns:
        new_row = df.anova(dv=target, between=cat, detailed=False)
        anova_df = pd.concat([anova_df, new_row], axis='rows')
    anova_df['Target'] = target
    anova_df = anova_df[['Target', 'Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']]
    return anova_df

get_anova_df(target='price', df=sample)
    
# numeric correlation with price
pg.pairwise_corr(sample, columns=['price']).round(3)

from sklearn.feature_selection import mutual_info_regression
##### feature importance #####
from sklearn.preprocessing import OrdinalEncoder


def make_mi_scores(x_data, y_data):
    # columns to consider
    X = x_data
    y = y_data
    categorical_features_for_mi = X.select_dtypes(include=['object', 'category']).columns
    # Ordinal encoding for categoricals
    oe = OrdinalEncoder(dtype=np.int64)
    X[categorical_features_for_mi] = oe.fit_transform(X[categorical_features_for_mi])
    # select discrete features, get their indices
    discrete_features_for_mi = X.dtypes == np.int64
    # calculate mi
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features_for_mi, random_state=2022)
    # make df of mi
    mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Scores': mi_scores}).sort_values('MI Scores', ascending=False)
    return mi_scores_df

mi_scores_df = make_mi_scores(
    x_data=sample.drop(columns='price'),
    y_data=sample['price']
    )

def graph_mi_scores(df):
    # bar plot
    plt.figure(figsize=(12, 12))
    sns.barplot(y='Feature', x='MI Scores', data=mi_scores_df, color='black')
    plt.title('Mutual Information Statistic')
    plt.show()
    
graph_mi_scores(mi_scores_df)

##### Pipeline Setup #####

# train, validation split
X_train, X_test, y_train, y_test = train_test_split(
    sample.drop('price', axis='columns'),
    sample['price'],
    test_size=0.30,
    random_state=2022
)

# select numeric_features
numeric_features = ['para1', 'para3', 'para2', 'para4']
categorical_features = X_train.select_dtypes(exclude=np.number).columns.to_list()

# Numeric Feature Pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1, 2))))
numeric_pipeline_steps.append(('box-cox', PowerTransformer(method='box-cox')))
numeric_pipeline_steps.append(('poly', PolynomialFeatures(degree=2)))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Categorical Feature Pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# Preprocessing Transformer
transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, categorical_features))
transformer_steps.append(('num', numeric_pipeline, numeric_features))
preprocessing_transformer = ColumnTransformer(transformers=transformer_steps)

# target pipeline (min-max, box-cox)
target_pipeline_steps = []
target_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1,2))))
target_pipeline_steps.append(('boxcox', PowerTransformer(method=('box-cox'))))
target_pipeline = Pipeline(steps=target_pipeline_steps)

# Display Transformer for features
set_config(display='diagram')
preprocessing_transformer

# Display Transformer for target
set_config(display='diagram')
target_pipeline

##### Models #####

# Models
models = []
models.append(('AB-C', AdaBoostRegressor(random_state=2022)))
#models.append(('CBR', CatBoostRegressor(iterations=1000, loss_function='RMSE', random_state=2022)))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor(weights='distance')))
models.append(('Lasso', Lasso()))
models.append(('RF', RandomForestRegressor(random_state=2022)))
models.append(('DR', DummyRegressor(strategy="mean")))
models.append(('DTR', DecisionTreeRegressor()))

##### Fitting Models #####
# result df
results = pd.DataFrame()

# scoring metric
scoring = 'neg_root_mean_squared_error'

# cv
cv = KFold(n_splits=10, shuffle=True, random_state=2022)

for name, model in models:
    # Pipeline
    model_pipeline_steps = []
    model_pipeline_steps.append(('transformer', preprocessing_transformer))
    model_pipeline_steps.append(('model', model))
    model_pipeline = Pipeline(steps=model_pipeline_steps)
    transformed_model = TransformedTargetRegressor(
        regressor=model_pipeline, transformer=target_pipeline)
    # CV results
    cv_results = cross_val_score(
        transformed_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    abs_cv_results = pd.Series(cv_results).abs()
    temp_df = pd.DataFrame({name: abs_cv_results})
    results = pd.concat([results, temp_df], axis='columns')
    # Mean +/- std
    msg = f'{name}: {abs_cv_results.mean().round(5)} +/- {abs_cv_results.std().round(5)}'
    print(msg)

# Algorithm Comparison Boxplot
sns.pointplot(y='model', x=scoring,
            data=pd.melt(results, var_name='model', value_name=scoring), palette='colorblind')
plt.title('Spot Check Algorithm Boxplots')
plt.show()

# dataframe of folds with results
results

# dataframe of model averages and standard deviations
pd.DataFrame({
    'mean':results.mean(),
    'std':results.std()
    }).sort_values('mean', ascending=True)

##### Tune Model #####
# Model
tuning_model = KNeighborsRegressor(weights='distance')

# Scoring
scoring = 'neg_root_mean_squared_error'

# Pipelinee Steps
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', tuning_model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Tranform Target
transformed_model = TransformedTargetRegressor(
    regressor=model_pipeline,
    transformer=PowerTransformer(method='box-cox'))

# Param Grid
transformed_model.get_params()
param_grid = {
    'regressor__model__n_neighbors': np.arange(start=1,stop=30,step=2)
}

# RandomizedSearchCV
bs_model = BayesSearchCV(
    estimator=transformed_model,
    search_spaces=param_grid,
    scoring=scoring,
	n_iter=50,
    n_jobs=-1
)

# Display Model
bs_model

# Fit Model on Train
bs_model.fit(X_train, y_train)

# best params
print(f'best params: {bs_model.best_params_}')

for key, value in bs_model.best_params_.items():
    print(f'{key}: {value}')

# best score
print(f'best score: {abs(bs_model.best_score_).round(5)}')

# best model
transformed_final_model = bs_model.best_estimator_

# RMSE on Test Set
print(f'RMSE on Validation Set: {mean_squared_error(y_test, bs_model.predict(X_test), squared=False).round(5)}')

##### Finalization #####
# Save Model
model_name = 'KNN'
model_file_name = f'models/{model_name} on {datetime.now().strftime("%Y %b %d at %H.%M.%S")}.pkl'
joblib.dump(transformed_final_model, model_file_name)