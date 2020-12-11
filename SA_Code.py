# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import scipy.stats as sp
import re
import matplotlib.pyplot as plt
# %matplotlib inline
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import string

from google.colab import drive 
drive.mount('/content/gdrive')

# Cahnge the link and write the name of the cs file you want to import here.
df=pd.read_csv('gdrive/My Drive/Data/ssr.csv', usecols=[1, 6])

df

def cleanTxt(df):
    df = re.sub(r'@[A-Za-z0-9]+', '', df) #Removing mentions
    df = re.sub(r'#[A-Za-z0-9]+', '', df)  #Removing hastags
    df = re.sub(r'RT[\s]+', '', df)   #Removing RTs
    df = re.sub(r'https?:\/\/\S+', '', df) #Removing URLs (generally promotional posts)
    return df
df['text'] = df['text'].apply(cleanTxt)
df

from textblob import TextBlob
#Function for getting subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Function for getting polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Creating new columns
df['Subjectivity'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)
df

from wordcloud import WordCloud 
# Plot the Word Cloud
allWords = ' '.join([twts for twts in df['text']])
wordCloud = WordCloud(width =1000, height =800, random_state = 21, max_font_size = 119).generate(allWords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

# Sentiment analysis
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
df['Class'] = df['Polarity'].apply(getAnalysis)

df

# Scatter-Plot the polarity and subjectivity in graph
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Red')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Get the percentage of positive tweets
ptweets = df[df.Class == 'Positive']
ptweets = ptweets['text']
round((ptweets.shape[0]/df.shape[0]*100), 1)

# Get the percentage of negative tweets
ntweets = df[df.Class == 'Negative']
ntweets = ntweets['text']
round((ntweets.shape[0]/df.shape[0]*100), 1)

# Get the percentage of neutral tweets
nttweets = df[df.Class == 'Neutral']
nttweets = nttweets['text']
round((nttweets.shape[0]/df.shape[0]*100), 1)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Displaying the Sentiment Coutns
df['Class'].value_counts()

#plot and visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Class'].value_counts().plot(kind='bar')
plt.show()
