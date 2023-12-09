#!/usr/bin/env python
# coding: utf-8

# https://medium.com/@AmyGrabNGoInfo/sentiment-analysis-hugging-face-zero-shot-model-vs-flair-pre-trained-model-57047452225d

# ## Load the data

# In[1]:


# !pip install pandas numpy
import pandas as pd
import numpy as np
import pickle

# In[ ]:





# In[2]:


# df = pd.read_csv("/content/sample_data/tweet_AAPL.csv")
df = pd.read_csv("./theCleanedTweets.csv")


# In[3]:


df


# In[4]:


df = df[['date','tweetV2']]
df


# In[5]:


df = df.reindex(index = df.index[::-1]).reset_index(drop=True)
df


# In[6]:


# !pip install transformers


# In[7]:


from transformers import pipeline


# In[8]:


# Define pipeline
classifier = pipeline(task="zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device='cpu')


# In[9]:


# !pip3 uninstall torch torchvision torchaudio -y


# In[10]:


# !pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6


# In[11]:


#
small_df = df.sample(frac=.05, random_state=42)
print(small_df.shape)
print(small_df.head())


# In[12]:


df['tweetV2'].map(type) != str
df = df[df['tweetV2'].apply(lambda x: isinstance(x, str))]


# ### Neutral

# In[ ]:


from tqdm import tqdm
# Put reviews in a list
#sequences = small_df.head(20)['tweetV2'].to_list()

sequences = df['tweetV2'].to_list()

candidate_labels = ["positive", "neutral", "negative"]

hypothesis_template = "The sentiment of this review is {}."

predictions = []

for sequence in tqdm(sequences, desc="Classifying Sentiments"):
    result = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
    predictions.append(result)

hf_prediction = pd.DataFrame(predictions)
hf_prediction.to_pickle("hf_prediction.pkl")

# In[ ]:


# import concurrent.futures
# from tqdm import tqdm

# sequences = df['tweetV2'].to_list()
# candidate_labels = ["positive", "neutral", "negative"]
# hypothesis_template = "The sentiment of this review is {}."

# def classifySentiment(sequence):
#     return classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)

# num_threads = 50

# sequence_chunks = [sequences[i:i + num_threads] for i in range(0, len(sequences), num_threads)]

# predictions = []

# with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#     for chunk in tqdm(sequence_chunks, desc="Classifying Sentiments"):
#         results = list(executor.map(classifySentiment, chunk))
#         predictions.extend(results)

# hf_prediction = pd.DataFrame(predictions)


# In[ ]:


# Take a look at the data
hf_prediction.head(10)


# In[ ]:


hf_prediction.iloc[2]


# In[ ]:


hf_prediction.iloc[2].sequence


# ### TO DO: We need to sort the column "labels" and "scores" to have the labels "positive, neutral, and negative" matching the scores.
# 
# ### If possible, have three columns, "positive", "neutral", and "negative"

# In[ ]:


labelOreo = ['positive', 'neutral', 'negative']

def labelSort(row):
    labels, scores = row['labels'], row['scores']
    
    label_index = {label: labelOreo.index(label) for label in labels}
    
    sorted_labels = sorted(labels, key=lambda x: label_index[x])
    
    sorted_scores = [scores[labels.index(label)] for label in sorted_labels]
    
    return sorted_labels, sorted_scores

hf_prediction[['labels','scores']] = hf_prediction.apply(labelSort, axis=1, result_type='expand')


# In[ ]:


hf_prediction


# In[ ]:


hf_prediction.to_csv("./200HoursOfPain.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




