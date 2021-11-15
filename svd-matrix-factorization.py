#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate


# In[3]:


# movies and their ratings have been merged

movie = pd.read_csv('../input/moviee-rating/movie.csv')
rating = pd.read_csv('../input/moviee-rating/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()


# In[4]:


df.shape


# In[5]:


# random 4 movies ID have been chosen

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]


# In[6]:


sample_df = df[df.movieId.isin(movie_ids)]


# In[7]:


sample_df.shape


# In[8]:


sample_df.head()


# In[9]:


# creating user movie matirix

user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating")


# In[10]:


user_movie_df.shape
user_movie_df.head()


# In[11]:


# surprise module wants to have specific scale

reader = Reader(rating_scale=(1, 5))


# In[12]:


# surprise module wants to have specific df shape

data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)


# # MODELLING

# In[13]:


trainset, testset = train_test_split(data, test_size=.25)


# In[14]:


# Create empty SVD model
svd_model = SVD()

# fitting
svd_model.fit(trainset)


# In[15]:


# prediction
predictions = svd_model.test(testset)


# In[16]:


accuracy.rmse(predictions)


# In[17]:


cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[18]:


user_movie_df.head()


# In[19]:


# Guessing ID=541 movie rating for user1
svd_model.predict(uid=1.0, iid=541, verbose=True)


# In[20]:


# Guessing ID=356 movie rating for user1
svd_model.predict(uid=1.0, iid=356, verbose=True)


# # MODEL TUNNING

# ![image.png](attachment:931196e6-4be8-4870-8090-7d8dade8cf39.png)

# In[21]:


# GridSearchCV
param_grid = {'n_epochs': [5, 50], 'lr_all': [0.002, 0.004]}

# number of epochs = number of iteration
# lr_all = learning rate


# In[22]:


gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1, joblib_verbose=True)


# In[23]:


gs.fit(data)


# In[24]:


gs.best_score['rmse']


# In[25]:


gs.best_params['rmse']


# # Final model ve Prediction

# In[26]:


svd_model = SVD(**gs.best_params['rmse'])
data = data.build_full_trainset()
svd_model.fit(data)


# In[27]:


svd_model.predict(uid=1.0, iid=541, verbose=True)


# In[28]:


svd_model.predict(uid=1.0, iid=356, verbose=True)

