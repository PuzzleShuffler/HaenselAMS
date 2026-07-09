import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OrdinalEncoder,
                                   PowerTransformer)


def df_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    # create dataframe
    df_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        '# Null': df.isna().sum()
    }).reset_index(drop=True)
    
    # get total rows
    total_rows = len(df)
    df_info['Total Count'] = total_rows
    
    # calculate null number and percentages
    if total_rows > 0:
        null_pct = (df_info['# Null'] / total_rows * 100).round(2)
        df_info['% Null'] = null_pct.astype(str) + '%'
    else:
        df_info['% Null'] = '0.0%'
        
    return df_info

def df_describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # get categorical columns
    cat_df = df.select_dtypes(exclude=np.number)

    # return empty dataframe if empty
    if cat_df.empty:
        return pd.DataFrame()

    # get number of observations
    counts = cat_df.count()

    # get missing
    n_missing = cat_df.isna().sum()

    # get number of unique values
    uniques = cat_df.nunique()

    # get mode
    modes_df = cat_df.mode().iloc[0]

    # get top categories
    top_categories = modes_df.astype(str)

    # get top category frequency
    top_freqs = (cat_df == modes_df).sum()

    # get percentage of top category
    top_pcts = ((top_freqs / counts).round(2) * 100).astype(int).astype(str) + "%"

    # build dataframe
    summary_df = pd.DataFrame(
        {
            "Column": cat_df.columns,
            "Data Type": cat_df.dtypes,
            "Count": counts,
            "Missing Values": n_missing,
            "Unique Values": uniques,
            "Top Category": top_categories,
            "Top Category Frequency": top_freqs,
            "% Top Category": top_pcts,
        }
    ).reset_index(drop=True)

    return summary_df

# Numeric Description
def get_describe_numeric(df: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    # get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # return empty dataframe if empty
    if numeric_df.empty:
        return pd.DataFrame()

    # describe columns with percentiles
    desc = numeric_df.describe(percentiles=[0.25, 0.50, 0.75]).T.reset_index()
    
    # rename columns
    desc = desc.rename(columns={
        'index': 'Column', 
        'count': 'Count', 
        'mean': 'Mean', 
        'std': 'Std Dev', 
        'min': 'Min',
        'max': 'Max'
    })

    # get data types of columns
    desc['Data Type'] = numeric_df.dtypes.values
    
    # get missing
    desc['Missing'] = numeric_df.isna().sum().values

    # calculate normality of columns
    normality = pg.normality(numeric_df, method='jarque_bera').reset_index()
    normality = normality.rename(columns={'index': 'Column'})

    # select final columns
    final_cols = [
        'Column', 'Data Type', 'Count', 'Missing', 'Mean', 
        'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'normal', 'pval'
    ]

    # merge dataframes
    summary_df = pd.merge(desc, normality[['Column', 'normal', 'pval']], on='Column')
    
    return summary_df[final_cols].round(precision)

def df_graph_correlogram(df: pd.DataFrame, precision: int = 2) -> plt.Axes:
    corr = df.corr(numeric_only=True).round(precision)
    
    if corr.empty:
        print("Warning: No numeric columns available to correlate.")
        return None

    mask = np.triu(np.ones_like(corr, dtype=bool))

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(11, 9))
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap='RdYlGn', 
            vmin=-1, 
            vmax=1, 
            center=0, 
            square=True,
            annot=True, 
            linewidths=.5,
            cbar_kws={"shrink": .5}, 
            annot_kws={"fontsize": 12}, 
            ax=ax
        )
        
        plt.title("Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
    return ax

def df_graph_numeric_histograms(df: pd.DataFrame) -> None:
    
    # get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # print error if none
    if len(numeric_columns) == 0:
        print("Warning: No numeric columns found to plot.")
        return

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    
    # create graphs
    with sns.axes_style("ticks", rc=custom_params):
        for col in numeric_columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            
            sns.histplot(data=df, x=col, kde=True, color='black', ax=ax)
            
            ax.set_title(f'Histogram of {col.title()}', fontsize=14, pad=12)
            ax.set_xlabel(col.title(), fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            
            plt.tight_layout()
            plt.show()


def df_get_anova(target: str, df: pd.DataFrame, is_target_cat: bool = True, precision: int = 4) -> pd.DataFrame:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # get features
    if is_target_cat:
        features = df.select_dtypes(include=np.number).columns
    else:
        features = df.select_dtypes(exclude=np.number).columns

    # if no features, return empty
    if len(features) == 0:
        return pd.DataFrame()

    # get results
    results = []
    for feat in features:
        dv_var = feat if is_target_cat else target
        between_var = target if is_target_cat else feat
        
        try:
            res = df.anova(dv=dv_var, between=between_var, detailed=False)
            
            res['Target'] = dv_var
            res['Source'] = between_var
            
            results.append(res)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    anova_df = pd.concat(results, axis='rows', ignore_index=True)
    
    anova_df['S.S. Diff'] = np.where(anova_df['p_unc'] <= 0.05, 'Yes', 'No')
    
    final_cols = ['Target', 'Source', 'ddof1', 'ddof2', 'F', 'p_unc', 'np2', 'S.S. Diff']
    return anova_df[final_cols].round(precision)

def pw_comp(cat: str, num: str, data: pd.DataFrame) -> None:
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=cat, x=num, data=data)
    plt.suptitle(f'Boxplot of {cat.title()} vs {num.title()}')
    plt.tight_layout()
    plt.show()
    
    print(f"\n--- Means by {cat.title()} ---")
    means_df = data.groupby(cat)[num].mean().to_frame()
    display(means_df)
    
    print(f"\n--- Pairwise Tukey Test ---")
    aov_df = pg.pairwise_tukey(dv=num, between=cat, data=data).round(4)
    
    if aov_df.empty:
        print("No pairs to compare or test failed.")
        return

    aov_df['S.S. Diff'] = np.where(aov_df['p_tukey'] <= 0.05, 'Yes', 'No')
    
    abs_hedges = aov_df['hedges'].abs()
    aov_df['Eff Size'] = pd.cut(
        abs_hedges, 
        bins=[-np.inf, 0.2, 0.5, np.inf], 
        labels=['Small', 'Medium', 'Large']
    )
    
    display(aov_df)

def get_chi_sq_df(target: str, df: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    categorical_columns = df.select_dtypes(exclude=np.number).columns.drop(target, errors='ignore')
    
    if len(categorical_columns) == 0:
        return pd.DataFrame()

    results = []
    for cat in categorical_columns:
        try:
            _, _, stats_df = pg.chi2_independence(data=df, x=target, y=cat)
            
            pearson_row = stats_df[stats_df['test'] == 'pearson'].copy()
            
            if not pearson_row.empty:
                pearson_row['Target'] = target
                pearson_row['Categorical Feature'] = cat
                results.append(pearson_row)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    chi_sq_df = pd.concat(results, axis='rows', ignore_index=True)
    
    chi_sq_df['S.S. Diff'] = np.where(chi_sq_df['pval'] <= 0.05, 'Yes', 'No')
    
    final_cols = [
        'Target', 'Categorical Feature', 'test', 'chi2', 
        'dof', 'pval', 'cramer', 'power', 'S.S. Diff'
    ]
    return chi_sq_df[final_cols].round(precision)

def chi_sq_info(x: str, y: str, data: pd.DataFrame) -> None:
    expected, observed, stats = pg.chi2_independence(data=data, x=x, y=y)
    
    stats['Conclusion'] = np.where(stats['pval'] <= 0.05, 'Dependent', 'Independent')
    
    stats['Effect Size'] = pd.cut(
        stats['cramer'],
        bins=[-np.inf, 0.2, 0.6, np.inf],
        labels=['Weak', 'Moderate', 'Strong']
    )
    
    expected_melted = expected.reset_index().melt(id_vars=x, value_name='Frequency')
    observed_melted = observed.reset_index().melt(id_vars=x, value_name='Frequency')
    
    g1 = sns.catplot(data=expected_melted, x=x, y="Frequency", col=y, kind="bar", height=4, aspect=1)
    g1.fig.suptitle('Expected Frequencies', y=1.05)
    
    g2 = sns.catplot(data=observed_melted, x=x, y="Frequency", col=y, kind="bar", height=4, aspect=1)
    g2.fig.suptitle('Observed Frequencies', y=1.05)
    plt.show()
    
    print('\n--- Expected Contingency Table ---')
    display(expected.round(4))
    
    print('\n--- Observed Contingency Table ---')
    display(observed.round(4))
    
    print('\n--- Pearson Test Summary ---')
    pearson_summary = stats[stats['test'] == 'pearson'].round(4)
    display(pearson_summary)


def mi_scores_regression(X_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
    # numeric pipeline
    numeric_pipeline_steps = []
    numeric_pipeline_steps.append(('ii_imputer', IterativeImputer()))
    numeric_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1, 2))))
    numeric_pipeline_steps.append(('box-cox', PowerTransformer(method='box-cox')))
    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

    # categorical pipeline
    categorical_pipeline_steps = []
    categorical_pipeline_steps.append(('si_impute', SimpleImputer(strategy='constant', fill_value='missing')))
    categorical_pipeline_steps.append(('oe', OrdinalEncoder(dtype=np.int64)))
    categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

    # create transformer
    cat_columns = X_data.select_dtypes(exclude=np.number).columns.tolist()
    num_columns = X_data.select_dtypes(include=np.number).columns.tolist()

    # transformer steps
    transformer_steps = []
    transformer_steps.append(('cat', categorical_pipeline, cat_columns))
    transformer_steps.append(('num', numeric_pipeline, num_columns))
    preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

    # preprocessing transformer
    preprocessing_transformer

    # preprocess X data
    pp_X = preprocessing_transformer.fit_transform(X_data)

    # target pipeline (min-max, box-cox)
    target_pipeline_steps = []
    target_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1,2))))
    target_pipeline_steps.append(('boxcox', PowerTransformer(method=('box-cox'))))
    target_pipeline = Pipeline(steps=target_pipeline_steps)

    # preprocess y data
    y = target_pipeline.fit_transform(np.reshape(y_data, (-1, 1))).ravel()

    # Get feature names out
    raw_features = preprocessing_transformer.get_feature_names_out()

    # get discrete feature indices
    discrete_features_for_mi = [col.startswith('cat__') for col in raw_features]

    # create dataframe for results
    pp_X_columns = pd.Series(raw_features).str.replace(r'^(num__|cat__)', '', regex=True)
    pp_X_df = pd.DataFrame(pp_X, columns=pp_X_columns)

    # run mutual_info_regression
    mi_scores = mutual_info_regression(pp_X_df, y, discrete_features=discrete_features_for_mi, random_state=2022)

    # make df of mi
    mi_scores_df = pd.DataFrame({'Feature': pp_X_df.columns, 'MI Scores': mi_scores}).sort_values('MI Scores', ascending=False)

    # bar plot
    plt.figure(figsize=(12, 12))
    sns.barplot(y='Feature', x='MI Scores', data=mi_scores_df, color='black')
    plt.title('Mutual Information Statistic')
    plt.show()

    # dataframe of mi scores
    display(mi_scores_df.round(5))

def mmi_scores_regression(X_data: pd.DataFrame, y_data: pd.DataFrame) -> pd.DataFrame:
    # numeric pipeline
    numeric_pipeline_steps = []
    numeric_pipeline_steps.append(('ii_imputer', IterativeImputer()))
    numeric_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1, 2))))
    numeric_pipeline_steps.append(('box-cox', PowerTransformer(method='box-cox')))
    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

    # categorical pipeline
    categorical_pipeline_steps = []
    categorical_pipeline_steps.append(('si_impute', SimpleImputer(strategy='constant', fill_value='missing')))
    categorical_pipeline_steps.append(('oe', OrdinalEncoder(dtype=np.int64)))
    categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

    # create transformer
    cat_columns = X_data.select_dtypes(exclude=[np.number]).columns.tolist()
    num_columns = X_data.select_dtypes(include=[np.number]).columns.tolist()

    transformer_steps = []
    transformer_steps.append(('cat', categorical_pipeline, cat_columns))
    transformer_steps.append(('num', numeric_pipeline, num_columns))
    preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

    # preprocessing transformer
    preprocessing_transformer

    # preprocess X data
    pp_X = preprocessing_transformer.fit_transform(X_data)

    # Get feature names out
    raw_features = preprocessing_transformer.get_feature_names_out()

    # get discrete feature indices
    discrete_features_for_mi = [col.startswith('cat__') for col in raw_features]

    # create dataframe for results
    pp_X_columns = pd.Series(raw_features).str.replace(r'^(num__|cat__)', '', regex=True)
    pp_X_df = pd.DataFrame(pp_X, columns=pp_X_columns)

    # run mutual_info_classif
    mi_scores = mutual_info_classif(pp_X_df, y_data, discrete_features=discrete_features_for_mi, random_state=2022)

    # make df of mi
    mi_scores_df = pd.DataFrame({'Feature': pp_X_df.columns, 'MI Scores': mi_scores}).sort_values('MI Scores', ascending=False)

    # bar plot
    plt.figure(figsize=(12, 12))
    sns.barplot(y='Feature', x='MI Scores', data=mi_scores_df, color='black')
    plt.title('Mutual Information Statistic')
    plt.show()


def display_model_best_params(params: Dict[str, Any], title: str = "Best Model Hyperparameters") -> None:
    # Initialize the console inside the function (or globally)
    console = Console()

    # Create the styled table using the dynamic title
    table = Table(title=f"🎯 {title}", title_style="bold cyan")
    table.add_column("Parameter", style="magenta", justify="left")
    table.add_column("Optimized Value", style="green", justify="right")

    # Loop through the dictionary passed into the function
    for key, value in params.items():
        table.add_row(key, str(value))

    # Output the table
    console.print(table)

if __name__ == '__main__':
    okay = 'okay'
    print(okay)