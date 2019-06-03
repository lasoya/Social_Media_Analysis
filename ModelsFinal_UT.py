#!/usr/bin/env python
# coding: utf-8

# # Social Media Analysis
# ## Data Cleaning

# ### Import Libraries & Function Definitions

# In[1]:


# from google.colab import files
# import libraries
#!/usr/bin/env python
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from io import StringIO
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model, metrics, preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, SelectKBest
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from statsmodels.formula.api import ols
import ast
import gc
import ijson
import itertools
import json
import matplotlib.pyplot as plt
import missingno as msno
import nltk
import numpy as np
import os
import pandas as pd
import pandas_profiling as pp
import re
import requests
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import time
import warnings
InteractiveShell.ast_node_interactivity = "all"
nltk.download('punkt')
nltk.download('vader_lexicon')
pd.set_option('display.max_columns', 300)
sns.set(style="whitegrid")


# In[2]:


# helper function to clean time column
def clean_time(time):
    time = re.sub('(:\d+)$', '', x)
    time = x.replace(':', '')
    return time


# function to bin time into AM, PM for next cell
def time_of_day(time):
    if time < 1200:
        return 'AM'
    else:
        return 'PM'


# function to convert month to season
def season(month):
    if month >= 3 and month < 6:
        return 'spring'
    elif month >= 6 and month < 9:
        return 'summer'
    elif month >= 9 and month < 12:
        return 'autumn'
    else:
        return 'winter'


# In[3]:


# Function to plot confustion matrix
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[4]:


# Function to remove highly correlated featrues
def remove_collinear_features(inp_data, corr_val):
    '''
    Returns an array or dataframe (based on type(inp_data) adjusted to drop \
        columns with high correlation to one another. Takes second arg corr_val
        that defines the cutoff

    ----------
    inp_data : np.array, pd.DataFrame
        Values to consider
    corr_val : float
        Value [0, 1] on which to base the correlation cutoff
    '''
    # Creates Correlation Matrix
    if isinstance(inp_data, np.ndarray):
        inp_data = pd.DataFrame(data=inp_data)
        array_flag = True
    else:
        array_flag = False
    corr_matrix = inp_data.corr()

    # Iterates through Correlation Matrix Table to find correlated columns
    drop_cols = []
    n_cols = len(corr_matrix.columns)

    for i in range(n_cols):
        for k in range(i + 1, n_cols):
            val = corr_matrix.iloc[k, i]
            col = corr_matrix.columns[i]
            row = corr_matrix.index[k]
            if abs(val) >= corr_val:
                # Prints the correlated feature set and the corr val
                print(col, "|", row, "|", round(val, 2))
                drop_cols.append(col)

    # Drops the correlated columns
    drop_cols = set(drop_cols)
    inp_data = inp_data.drop(columns=drop_cols)
    # Return same type as inp
    if array_flag:
        return inp_data.values
    else:
        return inp_data


# In[5]:


# Define a function that gives us a dataframe to preview the missing values
# and the % of missing values in each column:
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={
        0: 'Missing_Values',
        1: 'Pct_of_Total_Values'
    })
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            'Pct_of_Total_Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


# ## Data Import & Pre-processing

# ### Search_Depression

# In[6]:


# search_depression DataFrame
file = 'Data_Files/tweets_s_depression.json'
dfs_depression = pd.read_json(file, lines=True)
dfs_depression.shape
dfs_depression.info()
dfs_depression.head(1)


# In[7]:


# Define Unnecessary Columns to be removed in all DataFrames
remove_cols = [
    'cashtags', 'link', 'location', 'mentions', 'photos', 'place',
    'profile_image_url', 'quote_url', 'retweet', 'urls', 'video', 'created_at',
    'timezone'
]

# Drop Unnecessary Columns
dfs_depression.drop(columns=remove_cols, inplace=True)
dfs_depression.shape

# Get column lists to use in data type cleanup

num_cols = dfs_depression.columns[(
    dfs_depression.dtypes.values == np.dtype('int64'))].tolist()
num_cols

dt_cols = dfs_depression.columns[(
    dfs_depression.dtypes.values == np.dtype('datetime64[ns]'))].tolist()
dt_cols

str_cols = dfs_depression.columns[(
    dfs_depression.dtypes.values == np.dtype('object'))].tolist()
str_cols

good_cols = num_cols + dt_cols + str_cols
good_cols

# Change DataFrame Datatypes
for col in num_cols:
    dfs_depression[col] = pd.to_numeric(dfs_depression[col],
                                        downcast='integer')

for col in str_cols:
    dfs_depression[col] = dfs_depression[col].astype(str)

dfs_depression.info()


# ### Search_Anxiety

# In[8]:


file = 'Data_Files/tweets_s_anxiety.json'
dfs_anxiety = pd.read_json(file, lines=True)
dfs_anxiety.shape
dfs_anxiety.info()
dfs_anxiety.head(1)


# In[9]:


# Drop Unnecessary Columns
dfs_anxiety.drop(columns=remove_cols, inplace=True)
dfs_anxiety.shape


# In[10]:


# Change DataFrame Datatypes
for col in num_cols:
    dfs_anxiety[col] = pd.to_numeric(dfs_anxiety[col], downcast='integer')

for col in str_cols:
    dfs_anxiety[col] = dfs_anxiety[col].astype(str)

dfs_anxiety.info()


# ### Search_Happiness

# In[11]:


file = 'Data_Files/tweets_s_happiness.json'
dfs_happiness = pd.read_json(file, lines=True)
dfs_happiness.shape
dfs_happiness.info()
dfs_happiness.head(1)


# In[12]:


# Drop Unnecessary Columns
dfs_happiness.drop(columns=remove_cols, inplace=True)
dfs_happiness.shape

# Change DataFrame Datatypes
for col in num_cols:
    dfs_happiness[col] = pd.to_numeric(dfs_happiness[col], downcast='integer')

for col in str_cols:
    dfs_happiness[col] = dfs_happiness[col].astype(str)

dfs_happiness.info()


# ### Search_Happiness_2

# In[13]:


file = 'Data_Files/tweets_s_happiness_2.json'
dfs_happiness_2 = pd.read_json(file, lines=True)
dfs_happiness_2.shape
dfs_happiness_2.info()
dfs_happiness_2.head(1)


# In[14]:


# Drop Unnecessary Columns
dfs_happiness_2.drop(columns=remove_cols, inplace=True)
dfs_happiness_2.shape

# Change DataFrame Datatypes
for col in num_cols:
    dfs_happiness_2[col] = pd.to_numeric(dfs_happiness_2[col],
                                         downcast='integer')

for col in str_cols:
    dfs_happiness_2[col] = dfs_happiness_2[col].astype(str)

dfs_happiness_2.info()


# ### Search_Happy

# In[15]:


file = 'Data_Files/tweets_s_happy.json'
dfs_happy = pd.read_json(file, lines=True)
dfs_happy.shape
dfs_happy.info()
dfs_happy.head(1)


# In[16]:


# Drop Unnecessary Columns
dfs_happy.drop(columns=remove_cols, inplace=True)
dfs_happy.shape

# Change DataFrame Datatypes
for col in num_cols:
    dfs_happy[col] = pd.to_numeric(dfs_happy[col], downcast='integer')

for col in str_cols:
    dfs_happy[col] = dfs_happy[col].astype(str)

dfs_happy.info()


# ## Vader Sentiment Analysis

# In[17]:


# need to add this line in imports even after the download
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create an instance of the Sentiment Intensity Analyzer from NLTK Vader
sid = SentimentIntensityAnalyzer()


# ## ss_depression

# In[18]:


# tried it on one example first
print(sid.polarity_scores(dfs_depression.tweet[0]))

# use Vader's Sentiment Intensity Analyzer to find the sentiment for each tweet
dfs_depression['polarity'] = dfs_depression.tweet.apply(lambda x: sid.
                                                        polarity_scores(x))

# split the polarity column of dictionaries into separate columns and drop unnecessary columns
dfs_depression = pd.concat([
    dfs_depression.drop('polarity', axis=1), dfs_depression['polarity'].apply(
        pd.Series).drop(['neg', 'neu', 'pos'], axis=1)
],
                           axis=1)
dfs_depression.rename(columns={'compound': 'ss_tweet'}, inplace=True)

# check the first few examples to make sure it worked
dfs_depression.head(1)


# In[19]:


# add target column - encoding 0 = positive, 1 = negative based on hashtag
dfs_depression['target'] = 1

# add tag column - search word or hashtag word used to find tweet
dfs_depression['tag'] = 'search_depression'

# add new column with number of hashtags
dfs_depression['num_hashtags'] = dfs_depression.hashtags.apply(lambda x: len(x)
                                                               )

# remove last 2 digits of each time after last colon using regex
dfs_depression.time = dfs_depression.time.apply(lambda x: re.sub(
    '(:\d+)$', '', x))

# remove colon
dfs_depression.time = dfs_depression.time.apply(lambda x: x.replace(':', ''))

# change dtype of time column to integer
dfs_depression.time = dfs_depression.time.astype('int16')

# add new column with time split into 2 bins - AM, PM
dfs_depression['time_of_day'] = dfs_depression.time.apply(lambda x:
                                                          time_of_day(x))

# add new columns with date split to month and day of week
dfs_depression['month'] = dfs_depression['date'].dt.month
dfs_depression['day'] = dfs_depression['date'].dt.day_name()

# add new column for season
dfs_depression['season'] = dfs_depression['month'].apply(lambda x: season(x))
dfs_depression.head(1)


# ### ss_anxiety

# In[20]:


# tried it on one example first
print(sid.polarity_scores(dfs_anxiety.tweet[0]))

# use Vader's Sentiment Intensity Analyzer to find the sentiment for each tweet
dfs_anxiety['polarity'] = dfs_anxiety.tweet.apply(lambda x: sid.
                                                  polarity_scores(x))

# split the polarity column of dictionaries into separate columns and drop unnecessary columns
dfs_anxiety = pd.concat([
    dfs_anxiety.drop('polarity', axis=1), dfs_anxiety['polarity'].apply(
        pd.Series).drop(['neg', 'neu', 'pos'], axis=1)
],
                        axis=1)
dfs_anxiety.rename(columns={'compound': 'ss_tweet'}, inplace=True)

# check the first few examples to make sure it worked
dfs_anxiety.head(1)


# In[21]:


# add target column - encoding 0 = positive, 1 = negative based on hashtag
dfs_anxiety['target'] = 1

# add tag column - search word or hashtag word used to find tweet
dfs_anxiety['tag'] = 'search_anxiety'

# add new column with number of hashtags
dfs_anxiety['num_hashtags'] = dfs_anxiety.hashtags.apply(lambda x: len(x))

# remove last 2 digits of each time after last colon using regex
dfs_anxiety.time = dfs_anxiety.time.apply(lambda x: re.sub('(:\d+)$', '', x))

# remove colon
dfs_anxiety.time = dfs_anxiety.time.apply(lambda x: x.replace(':', ''))

# change dtype of time column to integer
dfs_anxiety.time = dfs_anxiety.time.astype('int16')

# add new column with time split into 2 bins - AM, PM
dfs_anxiety['time_of_day'] = dfs_anxiety.time.apply(lambda x: time_of_day(x))

# add new columns with date split to month and day of week
dfs_anxiety['month'] = dfs_anxiety['date'].dt.month
dfs_anxiety['day'] = dfs_anxiety['date'].dt.day_name()

# add new column for season
dfs_anxiety['season'] = dfs_anxiety['month'].apply(lambda x: season(x))
dfs_anxiety.head(1)


# ### ss_happy

# In[22]:


# tried it on one example first
print(sid.polarity_scores(dfs_happy.tweet[0]))

# use Vader's Sentiment Intensity Analyzer to find the sentiment for each tweet
dfs_happy['polarity'] = dfs_happy.tweet.apply(lambda x: sid.polarity_scores(x))

# split the polarity column of dictionaries into separate columns and drop unnecessary columns
dfs_happy = pd.concat([
    dfs_happy.drop('polarity', axis=1), dfs_happy['polarity'].apply(
        pd.Series).drop(['neg', 'neu', 'pos'], axis=1)
],
                      axis=1)
dfs_happy.rename(columns={'compound': 'ss_tweet'}, inplace=True)

# check the first few examples to make sure it worked
dfs_happy.head(1)


# In[23]:


# add target column - encoding 0 = positive, 1 = negative based on hashtag
dfs_happy['target'] = 0

# add tag column - search word or hashtag word used to find tweet
dfs_happy['tag'] = 'search_happy'

# add new column with number of hashtags
dfs_happy['num_hashtags'] = dfs_happy.hashtags.apply(lambda x: len(x))

# remove last 2 digits of each time after last colon using regex
dfs_happy.time = dfs_happy.time.apply(lambda x: re.sub('(:\d+)$', '', x))

# remove colon
dfs_happy.time = dfs_happy.time.apply(lambda x: x.replace(':', ''))

# change dtype of time column to integer
dfs_happy.time = dfs_happy.time.astype('int16')

# add new column with time split into 2 bins - AM, PM
dfs_happy['time_of_day'] = dfs_happy.time.apply(lambda x: time_of_day(x))

# add new columns with date split to month and day of week
dfs_happy['month'] = dfs_happy['date'].dt.month
dfs_happy['day'] = dfs_happy['date'].dt.day_name()

# add new column for season
dfs_happy['season'] = dfs_happy['month'].apply(lambda x: season(x))
dfs_happy.head(1)


# ### ss_happiness

# In[24]:


# tried it on one example first
print(sid.polarity_scores(dfs_happiness.tweet[0]))

# use Vader's Sentiment Intensity Analyzer to find the sentiment for each tweet
dfs_happiness['polarity'] = dfs_happiness.tweet.apply(lambda x: sid.
                                                      polarity_scores(x))

# split the polarity column of dictionaries into separate columns and drop unnecessary columns
dfs_happiness = pd.concat([
    dfs_happiness.drop('polarity', axis=1), dfs_happiness['polarity'].apply(
        pd.Series).drop(['neg', 'neu', 'pos'], axis=1)
],
                          axis=1)
dfs_happiness.rename(columns={'compound': 'ss_tweet'}, inplace=True)

# check the first few examples to make sure it worked
dfs_happiness.head(1)


# In[25]:


# add target column - encoding 0 = positive, 1 = negative based on hashtag
dfs_happiness['target'] = 0

# add tag column - search word or hashtag word used to find tweet
dfs_happiness['tag'] = 'search_happiness'

# add new column with number of hashtags
dfs_happiness['num_hashtags'] = dfs_happiness.hashtags.apply(lambda x: len(x))

# remove last 2 digits of each time after last colon using regex
dfs_happiness.time = dfs_happiness.time.apply(lambda x: re.sub(
    '(:\d+)$', '', x))

# remove colon
dfs_happiness.time = dfs_happiness.time.apply(lambda x: x.replace(':', ''))

# change dtype of time column to integer
dfs_happiness.time = dfs_happiness.time.astype('int16')

# add new column with time split into 2 bins - AM, PM
dfs_happiness['time_of_day'] = dfs_happiness.time.apply(lambda x: time_of_day(
    x))

# add new columns with date split to month and day of week
dfs_happiness['month'] = dfs_happiness['date'].dt.month
dfs_happiness['day'] = dfs_happiness['date'].dt.day_name()

# add new column for season
dfs_happiness['season'] = dfs_happiness['month'].apply(lambda x: season(x))
dfs_happiness.head(1)


# ### ss_happiness_2

# In[26]:


# tried it on one example first
print(sid.polarity_scores(dfs_happiness_2.tweet[0]))

# use Vader's Sentiment Intensity Analyzer to find the sentiment for each tweet
dfs_happiness_2['polarity'] = dfs_happiness_2.tweet.apply(lambda x: sid.
                                                          polarity_scores(x))

# split the polarity column of dictionaries into separate columns and drop unnecessary columns
dfs_happiness_2 = pd.concat([
    dfs_happiness_2.drop('polarity', axis=1),
    dfs_happiness_2['polarity'].apply(pd.Series).drop(['neg', 'neu', 'pos'],
                                                      axis=1)
],
                            axis=1)
dfs_happiness_2.rename(columns={'compound': 'ss_tweet'}, inplace=True)

# check the first few examples to make sure it worked
dfs_happiness_2.head(1)


# In[27]:


# add target column - encoding 0 = positive, 1 = negative based on hashtag
dfs_happiness_2['target'] = 0

# add tag column - search word or hashtag word used to find tweet
dfs_happiness_2['tag'] = 'search_happiness'

# add new column with number of hashtags
dfs_happiness_2['num_hashtags'] = dfs_happiness_2.hashtags.apply(lambda x: len(
    x))

# remove last 2 digits of each time after last colon using regex
dfs_happiness_2.time = dfs_happiness_2.time.apply(lambda x: re.sub(
    '(:\d+)$', '', x))

# remove colon
dfs_happiness_2.time = dfs_happiness_2.time.apply(lambda x: x.replace(':', ''))

# change dtype of time column to integer
dfs_happiness_2.time = dfs_happiness_2.time.astype('int16')

# add new column with time split into 2 bins - AM, PM
dfs_happiness_2['time_of_day'] = dfs_happiness_2.time.apply(lambda x:
                                                            time_of_day(x))

# add new columns with date split to month and day of week
dfs_happiness_2['month'] = dfs_happiness_2['date'].dt.month
dfs_happiness_2['day'] = dfs_happiness_2['date'].dt.day_name()

# add new column for season
dfs_happiness_2['season'] = dfs_happiness_2['month'].apply(lambda x: season(x))
dfs_happiness_2.head(1)


# ### Combine Dataframes

# In[28]:


dfs_depression = pd.concat([dfs_depression, dfs_anxiety],
                           ignore_index=True).reset_index(drop=True)
dfs_happiness = pd.concat([dfs_happiness, dfs_happiness_2],
                          ignore_index=True).reset_index(drop=True)
dfs_happiness = pd.concat([dfs_happiness, dfs_happy],
                          ignore_index=True).reset_index(drop=True)
dfs_tweets = pd.concat([dfs_depression, dfs_happiness],
                       ignore_index=True).reset_index(drop=True)
dfs_tweets.shape
dfs_tweets.info()
dfs_tweets.head(1)
dfs_tweets.to_pickle('Data_Files/dfs_tweets_combineoutput.pkl')


# In[29]:


# Drop Duplicates based on tweet id
dfs_tweets.shape
dfs_tweets.drop_duplicates('id', inplace=True)
dfs_tweets.shape

# # Archive dfs_tweets DataFrame
# dfs_tweets.to_csv('dfs_tweets.csv')
# dfs_tweets.to_pickle('Data_Files/dfs_tweets.pkl')


# In[30]:


### Memory Management: delete unused dfs
get_ipython().run_line_magic('whos', 'DataFrame')

#delete when no longer needed
del dfs_anxiety
del dfs_depression
del dfs_happiness
del dfs_happiness_2
del dfs_happy

#collect residual garbage
gc.collect()

get_ipython().run_line_magic('whos', 'DataFrame')


# In[31]:


dfs_tweets = pd.read_pickle('Data_Files/dfs_tweets_combineoutput.pkl')
dfs_tweets.head(1)

# Combine Hashtag Data
dfs_hashtag = pd.read_pickle('Data_Files/hashtag_data.pkl')
dfs_hashtag.rename({'compound': 'ss_tweet'}, axis=1, inplace=True)
dfs_hashtag.shape
dfs_hashtag.head(1)
dfs_tweets = pd.concat([dfs_tweets, dfs_hashtag], ignore_index=True,
                       sort=True).reset_index(drop=True)

# Drop Duplicates based on tweet id
dfs_tweets.shape
dfs_tweets.drop_duplicates('id', inplace=True)
dfs_tweets.shape
dfs_tweets.head(1)

# Memory Management
get_ipython().run_line_magic('whos', 'DataFrame')

#delete when no longer needed
del dfs_hashtag

#collect residual garbage
gc.collect()

get_ipython().run_line_magic('whos', 'DataFrame')


# ## Feature Engineering

# In[32]:


# Engineered Metric #1
# avg_ss_tweet
dfs_tweets['avg_ss_tweet'] = dfs_tweets['ss_tweet'].mean()
dfs_tweets.head(1)


# In[33]:


# Engineered Metric #2
# avg_ss_convo
dfgb_assc = dfs_tweets.groupby(
    'conversation_id')['ss_tweet'].mean().reset_index()
dfgb_assc.rename({'ss_tweet': 'avg_ss_convo'}, axis='columns', inplace=True)
dfs_tweets = dfs_tweets.merge(dfgb_assc, on='conversation_id', how='left')

### Memory Management
#delete when no longer needed
del dfgb_assc

#collect residual garbage
gc.collect()

dfs_tweets.tail(1)


# In[34]:


# Engineered Metric #3
# avg_ss_tag
dfgb_asst = dfs_tweets.groupby('tag')['ss_tweet'].mean().reset_index()
dfgb_asst.rename({'ss_tweet': 'avg_ss_tag'}, axis='columns', inplace=True)
dfs_tweets = dfs_tweets.merge(dfgb_asst, on='tag', how='left')

### Memory Management
#delete when no longer needed
del dfgb_asst

#collect residual garbage
gc.collect()

dfs_tweets.tail(1)


# In[35]:


# Engineered Metric #4
# avg_ss_dow
dfgb_assd = dfs_tweets.groupby('day')['ss_tweet'].mean().reset_index()
dfgb_assd.rename({'ss_tweet': 'avg_ss_dow'}, axis='columns', inplace=True)
dfs_tweets = dfs_tweets.merge(dfgb_assd, on='day', how='left')

### Memory Management
#delete when no longer needed
del dfgb_assd

#collect residual garbage
gc.collect()

dfs_tweets.tail(1)


# In[36]:


# Engineered Metric #5
# avg_ss_season
dfgb_asss = dfs_tweets.groupby('season')['ss_tweet'].mean().reset_index()
dfgb_asss.rename({'ss_tweet': 'avg_ss_season'}, axis='columns', inplace=True)
dfs_tweets = dfs_tweets.merge(dfgb_asss, on='season', how='left')

### Memory Management
#delete when no longer needed
del dfgb_asss

#collect residual garbage
gc.collect()

dfs_tweets.tail(1)


# In[37]:


# Engineered Metric #6
# avg_ss_tod
dfgb_tod = dfs_tweets.groupby('time_of_day')['ss_tweet'].mean().reset_index()
dfgb_tod.rename({'ss_tweet': 'avg_ss_tod'}, axis='columns', inplace=True)
dfs_tweets = dfs_tweets.merge(dfgb_tod, on='time_of_day', how='left')

### Memory Management
#delete when no longer needed
del dfgb_tod

#collect residual garbage
gc.collect()

dfs_tweets.tail(1)


# ### Create Dummies

# In[38]:


cat_cols = ['month', 'season', 'tag', 'time_of_day', 'day']

for col in cat_cols:
    dfs_tweets[col] = dfs_tweets[col].astype('category')

dfs_tweets.info()


# In[39]:


# Adding step to remove 'tag' column to address multicollinearity
dfs_tweets.drop(columns=['tag', 'avg_ss_tag'], inplace=True)
# dfs_tweets.rename({'ss_tweet' : 'ss_tweet'}, inplace=True)

dummy_cols = dfs_tweets.columns[dfs_tweets.dtypes == 'category']
dummy_cols

dfs_tweets.shape
dfs_dummies = pd.get_dummies(dfs_tweets, columns=dummy_cols, drop_first=True)
dfs_dummies.shape

drop_cols1 = [
    'conversation_id', 'date', 'hashtags', 'id', 'name', 'time', 'tweet',
    'username', 'user_id'
]
keys = dfs_dummies.drop(columns=drop_cols1).columns.tolist()
values = ['mean' for x in range(len(dfs_dummies.columns.tolist()))]
dfs_dict = dict(zip(keys, values))
dfs_dict.update({'target': 'max'})
print(dfs_dict)

# create user dataframe where each row is a user using user_id as unique identifier
user_df = dfs_dummies.groupby(['user_id'
                               ]).agg(dfs_dict).reset_index(drop=False)
user_df.shape


# ### EDA

# In[40]:


# target variable 'target' counts
print('Target Variable')
print(user_df.groupby(['target']).target.count())

# Target Variable Countplot
sns.set_style('darkgrid')
plt.figure(figsize=(11.7, 8.27))
sns.countplot(user_df['target'], alpha=.80, palette=['blue', 'orange'])
plt.title('Class Imbalence')
plt.ylabel('Total Count')
plt.show()


# In[41]:


# Create Profile Report & Export to HTML

profile = pp.ProfileReport(user_df)
# profile.to_file('SocialMediaAnalysis_ProfileReport.html', check_correlation=False)
profile.to_file('SocialMediaAnalysis_ProfileReport.html')
profile


# In[42]:


# Preprocessing Correlation Matrix
correlations_data = user_df.corr()['target'].sort_values()
correlations_data

sns.set(style="white")

# Compute the correlation matrix
corr = user_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(corr,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .7})


# In[43]:


# Correlation with output variable
# Output = no correlations

cor_target = abs(corr['target'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print('List of Features w/correlation > 0.5 to target variable: ')
print(relevant_features)

# Create box plots for each numeric column
for i in user_df.select_dtypes(['float64', 'int64']).columns.tolist():
    user_df.boxplot(column=i)
    plt.show()


# In[44]:


# preserve copy of dataframe before transformation
# user_df.to_pickle('Data_Files/user_df_pretran.pkl')


# ### Feature Importance

# In[45]:


# # Remove any columns with all na values
# features  = features.dropna(axis=1, how = 'all')
# features.shape

# Remove outliers from DataFrame
user_df.shape
user_df = user_df[(np.abs(stats.zscore(user_df)) < 3).all(axis=1)]
user_df.shape

# Create Target and Features Variables
target = user_df.target
target.shape
# features = user_df.drop(columns='target')
target[:5]

# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(
    user_df.drop(columns=['user_id', 'target']), 0.7)
features.shape
features.columns.tolist()
column_names = features.columns.tolist()

# Create test train splits
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    random_state=34,
                                                    test_size=0.3)

# Scale Features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train), columns=column_names)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=column_names)

# Sanity Check
X_train.columns.tolist()
y_train[:3]
X_test.columns.tolist()
y_test[:3]


# ### Dummy Classifier

# In[46]:


# DummyClassifier to predict target 'sad' = 1
with io.capture_output() as captured:
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    dummy_pred_y_train = dummy.predict(X_train)
    dummy_pred_y_test = dummy.predict(X_test)

# checking accuracy
print('Dummy Classifier Scores')
print('Train Accuracy score: ', accuracy_score(y_train, dummy_pred_y_train))
print('Train F1 score: ', f1_score(y_train, dummy_pred_y_train))
print('Test Accuracy score: ', accuracy_score(y_test, dummy_pred_y_test))
print('Test F1 score: ', f1_score(y_test, dummy_pred_y_test))

with io.capture_output() as captured:
    ### Print and Plot Confustion Matrix  ###

    # save confusion matrix and slice into four pieces
    cm = metrics.confusion_matrix(y_test, dummy_pred_y_test)
    classes = ['Happy', 'Sad']
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Plot confustion matrix
    plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)


# ### Downsampling

# In[47]:


# Refresh Data
# Create Target and Features Variables
target = user_df.target
target.shape
# features = user_df.drop(columns='target')
target[:5]

# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(
    user_df.drop(columns=['user_id', 'target']), 0.7)
features.shape
features.columns.tolist()
column_names = features.columns.tolist()

# Create test train splits
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    random_state=34,
                                                    test_size=0.3)

# Scale Features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train), columns=column_names)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=column_names)

# Sanity Check
X_train.columns.tolist()
y_train[:3]
X_test.columns.tolist()
y_test[:3]


# In[48]:


# concatenate our training data back together
with io.capture_output() as captured:
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    training = pd.concat([X_train, y_train.to_frame()], axis=1)
    training = training[~training.duplicated()].reset_index(drop=True)
#     training.drop_duplicates(inplace=True)
# training.columns.duplicated()
# training.dropna().reset_index(inplace=True)
# training.head(1)

    # separate minority and majority classes
    happy = training[training['target'] == 0]  # Majority Class
    sad = training[training['target'] == 1]  # Minority Class

    # downsample majority class i.e. happy
    happy_downsampled = resample(happy,
                                 replace=False,  # sample without replacement
                                 n_samples=len(sad),  # match minority n
                                 random_state=34)  # reproducible results
    # combine minority and downsampled majority
    # #  Drop indexes to prevent nans post-concat
    # sad_downsampled.reset_index(drop=True, inplace=True)
    # sad.reset_index(drop=True, inplace=True)

    downsampled = pd.concat([happy_downsampled, sad])

    # Drop duplicate columns should the exist
    downsampled = downsampled.loc[:, ~downsampled.columns.duplicated()]

# checking Results
print('Minority and Majority Class Counts: ')
print('Sad  Count: ' + str(len(sad)))
print('Happy  Count: ' + str(len(happy)))

print('Downsampling results: ')
downsampled['target'].value_counts()


# ### Decision Tree Classifier

# In[49]:


# Plot Feature importance with Tree Based Classifiers inbuilt class
# using Random Forest Classifier to fit model, predict, and extract the top 10 features
with io.capture_output() as captured:
    X_train = downsampled.drop('target', axis=1)  #independent columns
    y_train = downsampled.target   #target column i.e price range
    rf_clf = ExtraTreesClassifier(max_depth=5, warm_start=True)
    rf_clf.fit(X_train,y_train)
    rfc_pred_y_train = rf_clf.predict(X_train)
    rfc_pred_y_test = rf_clf.predict(X_test)

print('Optimal number of features :', rf_clf.n_features_)
# print('Best features :', X_train.columns[rf_clf.get_support()])

# # checking accuracy
print('Feature Importance via Extra Trees Classifier')
print('Train Accuracy score: ', accuracy_score(y_train, rfc_pred_y_train))
print('Train F1 score: ', f1_score(y_train, rfc_pred_y_train))
print('Test Accuracy score: ', accuracy_score(y_test, rfc_pred_y_test))
print('Test F1 score: ', f1_score(y_test, rfc_pred_y_test))
print('Feature Counts - X_train Shape: {xt}, Y_train Shape: {yt}'.format(xt=X_train.shape, yt=y_train.shape))

#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show();


# ### Recursive Feature Elimination

# In[50]:


# # Reset Data
# X_train, X_test, y_train, y_test =  reset_data()

# Initalize Variable and Parameters
# rfe = linear_model.LogisticRegression(random_state=34, solver='saga', n_jobs=-1)
rfe = linear_model.LogisticRegression(random_state=34, solver='saga', n_jobs=-1)

# Create recursive feature eliminator that scores features by f1
rfe = RFECV(estimator=rfe, step=1, cv=5, scoring='f1')

# Select variables and calulate test accuracy
# example model = http://dkopczyk.quantee.co.uk/feature-selection/
rfe = rfe.fit(X_train, y_train)
selected_columns = X_train.columns[rfe.support_]
removed_columns = X_train.columns[~rfe.support_]
X_train = X_train[selected_columns]
X_test = X_test[selected_columns]
y_train_pred = rfe.estimator_.predict(X_train)
y_pred = rfe.estimator_.predict(X_test)
X_train.shape
len(selected_columns)

# Feature selection
print('Optimal number of features :', rfe.n_features_)
# print('Best features :', X_train.columns[rfe.support_])

# checking accuracy
print('Feature Importance via Recursive Feature Elimination')
print('Train Accuracy score: ', accuracy_score(y_train, y_train_pred))
print('Train F1 score: ', f1_score(y_train, y_train_pred))
print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
print('Test F1 score: ', f1_score(y_test, y_pred))
print('Feature Counts - X_train Shape: {xt}, Y_train Shape: {yt}'.format(xt=X_train.shape, yt=y_train.shape))

# Plot number of features vs CV scores
plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[51]:


coef_dict = {}
for coef, feat in zip(rfe.estimator_.coef_.flatten().tolist(), selected_columns.tolist()):
    coef_dict[feat] = coef
coef_dict

selected_columns
#use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.DataFrame.from_dict(coef_dict, orient='index',columns=['coef_val'])
feat_importances.sort_values(by='coef_val', ascending=True).plot(kind='barh')
plt.show();


# ### RandomizedSearchCV

# In[52]:


default_SVC = svm.SVC()
print("Default SVC parameters are: \n{}".format(default_SVC.get_params))

# Create, fit, and test default SVM
rbfSVM = SVC(kernel='rbf')
rbfSVM.fit(X_train, y_train)
svm_predictions = rbfSVM.predict(X_test)

print(metrics.classification_report(y_test, svm_predictions))

# Designate distributions to sample hyperparameters from
np.random.seed(123)
g_range = np.random.uniform(0.0, 0.3, 5).astype(float)
C_range = np.random.normal(1, 0.1, 5).astype(float)

# Check that gamma>0 and C>0
C_range[C_range < 0] = 0.0001

hyperparameters = {'gamma': list(g_range),
                   'C': list(C_range),
                   'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                   'probability': [True, False]}

print(hyperparameters)


# In[53]:


# Run randomized search
randomCV = RandomizedSearchCV(
    SVC(random_state=34), param_distributions=hyperparameters, n_iter=20)
randomCV.fit(X_train, y_train)

# Identify optimal hyperparameter values
best_gamma = randomCV.best_params_['gamma']
best_C = randomCV.best_params_['C']

print("The best performing gamma value is: {:5.2f}".format(best_gamma))


# In[54]:


# Train SVM and output predictions
rbfSVM = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
rbfSVM.fit(X_train, y_train)
svm_predictions = rbfSVM.predict(X_test)

print(metrics.classification_report(y_test, svm_predictions))
print("Overall Accuracy:", round(
    metrics.accuracy_score(y_test, svm_predictions), 4))

with io.capture_output() as captured:
    ### Print and Plot Confustion Matrix  ###

    # save confusion matrix and slice into four pieces
    cm = metrics.confusion_matrix(y_test, svm_predictions)
    classes = ['Happy', 'Sad']
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Plot confustion matrix
    plt.figure(figsize = (11.7,8.27))
    plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)


# In[55]:


def GridSearch_table_plot(randomCV, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):
    '''Display grid search results

    Arguments
    ---------

    randomCV           the estimator resulting from a grid search
                       for example: randomCV = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(randomCV, "min_samples_leaf")

                          '''
    from matplotlib import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = randomCV.best_estimator_
    clf_params = randomCV.best_params_
    if negative:
        clf_score = -randomCV.best_score_
    else:
        clf_score = randomCV.best_score_
    clf_stdev = randomCV.cv_results_['std_test_score'][randomCV.best_index_]
    cv_results = randomCV.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results)
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(
            param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()


GridSearch_table_plot(randomCV, "C", negative=False)


# In[56]:


# /Users/utaveras/FlatironSchool/DS-031119/Week12/Mod4-Project4/Mod4-Project4.ipynb
get_ipython().system('jupyter nbconvert Mod4-Project4.ipynb --to pdf')
get_ipython().system('jupyter nbconvert Mod4-Project4.ipynb --to html')
get_ipython().system('jupyter nbconvert Mod4-Project4.ipynb --to script')
get_ipython().system('jupyter nbconvert Mod4-Project4.ipynb --to slides')


# In[ ]:




