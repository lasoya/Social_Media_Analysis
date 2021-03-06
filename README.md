# Inferring Overall User Mood With Twitter Sentiment Analysis


## Project Overview

According to the <a src='https://www.nimh.nih.gov/health/topics/depression/index.shtml'>National Institute of Mental Health</a>, depression is one of the most common mental disorders in the U.S., caused by a mix of genetic, environmental, and psychological factors. This disorder can occur at any age and symptoms, which can be disruptive, vary per individual. Fortunately, it can be treated regardless of severity, but treatment is more effective when it is diagnosed at an earlier stage.

For this project, we were interested in answering the question: is it possible to infer if a user is at risk for depression from social media activity. With the emergence of the internet where individuals spend most of their time online, there is a wealth of data that can be tapped for analysis to better understand the user's overall mood. However, we were unable to obtain medical data, which would be important to tackling this particular question. Therefore, we decided to approach it from another perspective.

The problem evaluated in this project was: can we predict the user's overall mood based on certain features from tweets? Our thinking was that if a user is on the happier side of the spectrum, s/he is less at risk for depression than if a user is on the sadder side of the spectrum. We approached the question as a binary classification problem with our target variable as the user's overall mood (0 = happy, 1 = sad). The data used were 1-year of tweets from May 28th 2018 to May 28th 2019, scraped from Twitter using TWINT, a Python advanced scraping tool for Twitter. We scraped tweets based on specific keywords and hashtags, based on their relationships to the words happiness and depression as showed by the word network map on <a src='http://www.ritetag.com'>Ritetag.com</a>. 


<p align='center'>
    <img src='./images/happiness_tags.png' title="Words Relational Map Around Happiness" width="425"/> <img hspace="10"/> <img src='./images/depression_tags.png' title="Words Relational Map Around Happiness" width="425"/>
</p>


About a half million tweets were able to be obtained, and after processing, the final dataset consisted of about 103,000 tweets. Each tweet was given a compound sentiment score using the Valence Aware Dictionary for Sentiment Reasoning (VADER), which is a model sensitive to the polarity (positive/negative) and intensity (strength) of emotion used in text sentiment analysis especially tailored for social media evaluation. Using the sentiment score for each tweet, we engineered additional variables which looked at the interactions between different features. After data processing and feature engineering, including creating dummy variables for all categorical variables, all tweets were aggregated to the user-level, with each value representing an average. After our base models, all highly correlated features (|correlation| > 0.7) were removed. The feature elimination results from using the Logistic Regression estimator in Recursive Feature Elimination showed that the top 20 features were: average sentiment score of conversation, average sentiment score of each tweet user posted, and the average number of tweets user posted depending of time of day, week, month, and season. According to feature elimination, the optimal number of features for the model was 20. Our final model after Randomized Search was a Support Vector Machine model with 20 features and parameters: C = 1.1979, kernel = 'rbf', and gamma = 0.0858. The accuracy score of the model on testing data was 0.84. The confusion matrix showed that the model was pretty accurate in predicting both True Postives (predict user is sad when user is actually sad) and True Negatives (predict user is happy when user is happy).


## Target Variable Labelling Methodology

Some assumptions were made during this project:
- A tweet was labeled as ‘sad’ if the search word or hashtag used to obtain the tweet was a negative term and vice versa.
- If a user posted a ‘depressed’ tweet before, this overrode prior tweets that were ‘happy’.


## Disclaimer

Because this is sensitive topic and assumptions made, we would like to note that we are not making amy medical claims or diagnoses about a user with these results. In addition, because we are not medical professionals and since we are unable to obtain real medical data for comparison with the results of this project, we do not recommend using this for actual medical practice or medical advice. This project is only for educational purposes.


## Data Preprocessing

- All columns with null values were dropped.
- Only columns that remained as possible features for modelling were kept.
- Some data types were downcasted to reduce memory footprint.
- Two major dataframes were created during this project:
    - Original dataframe: Tweet-level data
    - Main dataframe (after processing): User-level data (for modelling and predictions)


## Natural Language Processing

VADER is 'a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.' As a module specifically created to evaluate the sentiments in social media, it was the prefect NLP tool to use in this project. More information on VADER can be found on their website <a src='https://github.com/cjhutto/vaderSentiment'>here</a>.

During the data preprocessing step in the previous section, we did not remove any punctuation, remove any emojis, lowercase any words, or tokenize words, which usually occurs for NLP, because they would adversely affect the sentiment scores given by VADER. In order to better evaluate the intensity of emotion and polarity of the language used in the tweets, punctuation, case of the words, and emojis are important. Using the Sentiment Intensity Analyzer in VADER, each tweet was given a sentiment score between -1 (most negative) and 1 (most positive).


## Feature Engineering

New features were created at the tweet-level and later aggregated as averages at the user-level. All categorical features were changed to dummy variables at the tweet-level before aggregation. The features engineered were:

- Target Variable (0 = happy, 1 = sad)
- Number of Hashtags: number of hashtags used in each tweet
- Time of Day: when hashtag was posted (AM, PM)
- Day of Week: the day of the week when tweet was posted
- Season: based on month tweet was posted
- Sentiment score: sentiment score of each tweet using Vader


## Feature Importance & Selection

We reviewed the correlation between pairs of features for feature selection. The highly correlated features removed (|correlation| > 0.7) were:

- Average Count of Replies Per Tweet User Posts
- Average Count of Retweets Per Tweet User Posts
- Average Sentiment Score Per Conversation
- Average Sentime Score Based on Tag (Search Keyword or Hashtag)
- Search Keyword = Depression
- Search Keyword = Happy
- Average Number of Tweets User Posts on a Sunday
- Average Number of Tweets User Posts in Spring
- Average Number of Tweets User Posts in the Afternoon/Night


<p align='center'>
    <img src='./images/correlation_heatmap.png' title='Correlation Heatmap'>
</p>


In addition, we used multiple feature importance tools to assist with feature elimination for modelling. These are the two major results:


<b>Feature Importance Using Decision Tree Classifier</b>

<u>Top 5 Important Features</u>

- Average User Tweet Sentiment Score
- Average Sentiment Score Per Season
- Average Count of Likes Per Tweet Posted
- Average Number of Hashtags Used Per Tweet
- Average Sentiment Score of Tweet Posted Based on Day of Week
- Average Sentiment Score of Tweet Posted Based on Time of Day


<p align='center'>
    <img src='./images/feat_import_dtree2.png' title='Feature Importance Using Decision Tree Classifier'>
</p>


<b>Feature Selection Using Logistic Regression Classifer through Recursive Feature Elimination</b>

These features defined through Recursive Feature Elimination were ulimately used for the final model as they were noted to be the optimal 20 features.

<u>Top 20 Important Features</u>

- Average Number of Tweets Posted in April
- Average Number of Tweets Posted on Tuesday
- Average Number of Tweets Posted on Wednesday
- Average Number of Tweets Posted in March
- Average Number of Tweets Posted on Monday
- Average Number of Tweets Posted on Thursday
- Average Number of Tweets Posted in June
- Average Number of Tweets Posted in Winter
- Average Number of Tweets Posted in November
- Average Number of Tweets Posted in October
- Average Sentiment Score of Each Tweet
- Average Number of Tweets Posted in Summer
- Average Number of Tweets Posted on Saturday
- Average Number of Tweets Posted in September
- Average Number of Tweets Posted in the Afternoon/Night (PM)
- Average Number of Tweets Posted in December
- Average Number of Hashtags Used in Each Tweet
- Average Number of Tweets Posted on Sunday
- Average Number of Tweets Posted in Spring
- Average Sentiment Score of Conversation


<p align='center'>
<img src='./images/coefficients_final_model.png' title='Chart of Features and Coefficients for Final Model'>
</p>/p>


## Class Evaluation

We used downsampling to minimize the influence of the majority class in the model.

<p align='center'>
<img src='./images/class_evaluation.png' title='Class Evaluation of Target Variable'>
</p>


## Model Evaluation

<i>Iteration 1</i>:

During this iteration, we evaluated two models, Naive Bayes and Support Vector Machine. The accuracy and F1 scores of these models came back as 1, which seemed highly unlikely to us. 

<p align='center'>
<img src='./images/model_metrics1.png' title='Chart of Model Metrics of Iteration 1'>
</p>

<i>Iteration 2</i>:

Based on iteration 1 results, we removed features that were highly correlated (|correlation| > 0.7) and reran the models. The accuracy and F1 scores of these models moved away from the perfect score but still seemed there was room for improvement. In addition, we realized that one of the features in the model was used for defining the target variable classes.

<p align='center'>
<img src='./images/model_metrics2.png' title='Chart of Model Metrics of Iteration 2'>
</p>

<i>Iteration 3</i>:

Based on iteration 2 findings, we removed the feature which was used to define the target variable labels ('tag' column or dummy variables created from the 'tag' column) and reran the models including a Logistic Regression model. The accuracy and F1 scores of these models decreased but we felt that these results were more plausible. However, further exploration was needed.

<p align='center'>
<img src='./images/model_metrics3.png' title='Chart of Model Metrics of Iteration 3'>
</p>

<i>Iteration 4</i>:

Based on previous findings, we used feature elimination to narrow down the optimal number of features as the prior iteration seems to indicate overfitting. In addition, we conducted Randomized Search to find the best parameters for the model. After these steps, our final model yielded 20 features.

<p align='center'>
<img src='./images/model_metrics4.png' title='Chart of Model Metrics of Iteration 4'>
</p>

<br>
The confusion matrix for the final model showed that it was fairly accurate in predicting True Positives (predict user is sad if actually sad) and True Negatives (predict user is happy if actually happy).

<p align='center'>
<img src='./images/final_model_confus_matrix.png' title='Confusion Matrix for Final Model'>
</p>


## Conclusion & Possible Applications

This topic is worth further exploration. If we are able to use Twitter or other social media platforms to identify those who are at-risk for depression earlier, we can provide necessary resources at an earlier time for more effective treatment. However, there are some caveats. Some thoughts to keep in mind are that there may be ethical implications with employers pre-screening prospective candidates, privacy concerns regarding using social media data for diagnosis, and possible misidentification due to the error rate.
<br>

## Recommended Next Steps

- Run more models to determine whether any improvement to our predictions can be made
- Additional data cleaning can be done in consideration of data integrity and to possibly use other analysis methods to find other relationships in the data
- Obtain medical data (if possible) to check the accuracy of the model and validate our assumptions