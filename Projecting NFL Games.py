#!/usr/bin/env python
# coding: utf-8

# # Gabriel Murazzi's ISE 435 Project:
# # Attempting to predict NFL games based on current odds
# 
# What I will be looking at is the vegas odds for every NFL game since 2000. 
# This consists of a spread (the difference between the projected losing team and projected winning team)
# and the Over/Under Line (the predicted sum of the scores of both teams).  
# I will also look at whether the favorite team is home/away, if the game is played inside a domed stadium, 
# and if the game is a playoff game.
# 
# Go ahead and "run all" :D

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import seaborn as sns 
import statsmodels.api as sm

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox



from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

import warnings
from collections import Counter
get_ipython().system('pip install imblearn ')
from imblearn.datasets import fetch_datasets
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from scipy import stats
from scipy.stats import kurtosis, skew

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Import the team key
pathKEY = "nfl_teams.csv"
team_key = pd.read_csv(pathKEY)
team_key.drop(['team_name_short', 'team_id_pfr', 'team_conference', 'team_division', 'team_conference_pre2002', 'team_division_pre2002'], axis=1, inplace=True)
display(team_key.head())


#Path for Scores, Betting odds, Weather
path = "spreadspoke_scores.csv"
game_data = pd.read_csv(path)

game_data.drop(['stadium_neutral','weather_wind_mph','weather_humidity'], axis=1, inplace=True)

#Only looking at games with betting data:
game_data.replace(' ', np.nan, inplace=True)
indexRemove = game_data[game_data['spread_favorite'] == np.nan].index
game_data.dropna(subset=['spread_favorite'] , inplace=True)
game_data.dropna(subset=['over_under_line'] , inplace=True)

#Only going to use data from the 2000 season onwards: Modern sports betting with the internet 
#will provide much more accurate data as the tech for online gambling is better.
game_data.query("schedule_season > 1999", inplace=True)

#fix over_under_line data type
game_data[['over_under_line']] = game_data[['over_under_line']].apply(pd.to_numeric) 

game_data.tail()


# In[3]:


#Create further useful columns

game_data.loc[:,"combined_score"] = abs(game_data.loc[:, "score_home"] + game_data.loc[:, "score_away"])

team_key.rename(columns = {'team_name':'team_home'}, inplace = True) 
game_data = pd.merge(game_data, team_key, on ='team_home', how ='left') 
game_data.rename(columns = {'team_id':'home_team_id'}, inplace = True) 

game_data.loc[:,"actual_spread"] = 0
game_data['actual_spread'] = np.where(game_data['home_team_id'] == game_data['team_favorite_id'], game_data['score_away']-game_data['score_home'],game_data['score_home']-game_data['score_away'])

#INTERPRETING actual_spread: THE SCORE DIFFERENE BETWEEN THE WINNING TEAM AND THE LOSING TEAM.  
        # if (-) then the favorite team won.  If (+) then the non-favorite team won
    
game_data.loc[:,"winning_bet"] = 0
game_data['winning_bet'] = np.where(game_data['actual_spread'] > 0, 0 , 1)
#0 means the favorite lost
#1 means the favorite won
#THIS IS WHAT I WILL BE ULTIMATELY PREDICTING


#0 means the away tam won
#1 means home team won
game_data.loc[:,"home_win"] = 0
game_data['home_win'] = np.where((game_data['score_home']>game_data['score_away']), 1,0)


#0 means favorite is not home
#1 means favorite is home
game_data.loc[:,"is_home"] = 0
game_data['is_home'] = np.where(game_data['home_team_id'] == game_data['team_favorite_id'], 1 , 0)

#0 means game is not played in a domed stadium
#1 means game is  played in a domed stadium
game_data.loc[:,"is_dome"] = 0
game_data['is_dome'] = np.where(game_data['weather_detail'] == "DOME", 1 , 0)

#0 means game is not a playoff game
#1 means game is a playoff game
game_data[['schedule_playoff']] = game_data[['schedule_playoff']].astype(int)

#Make the index the date:
game_data.index = pd.to_datetime(game_data['schedule_date'])
game_data = game_data.drop(['schedule_date'], axis = 1)


# In[4]:


#CHECK WITH THIS NOT ^THAT
game_data


# In[5]:


#See how over/under correlates with the actual score

get_ipython().run_line_magic('matplotlib', 'notebook')
x = game_data["over_under_line"]
y = game_data["combined_score"]
plt.plot(x,y,'o', color = "olivedrab", label = "Over/Under vs. Actual Total Score") 
plt.xlabel("Over/Under")
plt.ylabel("Actual total score") 
plt.title("Over/Under vs. Actual Total Score")
plt.xlim([25, 65])


# In[6]:


#And check spread correlation before and after the games
get_ipython().run_line_magic('matplotlib', 'notebook')
x2 = game_data["spread_favorite"]
y2 = game_data["actual_spread"]
plt.plot(x2,y2,'o', color = "blue", label = "spread favorite vs actual spread") 
plt.xlabel("Spread(favorite)")
plt.ylabel("Actual spread") 
plt.title("spread favorite vs actual spread")
plt.xlim([-30,1])


# In[7]:


display(game_data.corr())
game_data.describe()


# In[8]:


sns.pairplot(game_data, vars = ['spread_favorite', 'actual_spread', 'over_under_line', 'weather_temperature', 'winning_bet','home_win','combined_score', 'is_home'])


# #### By the looks of it, the data doesn't have much correlation.  I will check outliers/skews

# In[9]:


#check outliers/skew for Over Under Line and Combined Score:
OU_line_skew = skew(game_data['over_under_line'])
combined_skew = skew(game_data['combined_score'])
display("Combined Score Skew: {:.2}".format(combined_skew))
display("Over/Under Line Skew: {:.2}".format(OU_line_skew))

OU_line_kurtosis = kurtosis(game_data['over_under_line'], fisher = True)
combined_kurtosis = kurtosis(game_data['combined_score'], fisher = True)
display("Combined Score Kurtosis: {:.2}".format(combined_kurtosis))
display("Over/Under Line Kurtosis: {:.2}".format(OU_line_kurtosis))


# #### Skews look to be approximately symetric for each (between -.5 and +.5)
# #### Kurtosis < 3 is platykurtic for each, and means tails are shorter and thinner, and central peak is lower and broader

# In[10]:


#check outliers/skew for the spread

spread_skew = skew(game_data['spread_favorite'])
actual_spread_skew = skew(game_data['actual_spread'])
display("Spread Skew: {:.2}".format(spread_skew))
display("Actual Spread Skew: {:.2}".format(actual_spread_skew))

spread_kurtosis = kurtosis(game_data['spread_favorite'], fisher = True)
actual_spread_kurtosis = kurtosis(game_data['actual_spread'], fisher = True)
display("Spread Kurtosis: {:.2}".format(spread_kurtosis))
display("Actual Spread Kurtosis: {:.2}".format(actual_spread_kurtosis))


# #### Pre-game spread is skewed but the actual score spread seems to be rather normal

# ## Linear Regression:
# 
# #### I'm going to be using these 5 columns from my data:
# ####   - Spread Favorite (Pre-game spread)
# ####   - Over/Under Line
# ####   - Is the favorite team playing at home?
# ####   - Is the game played inside a domed stadium?
# ####   - Is this a playoff game?

# ### In this first try, I wanted to predict the actual spread (the score difference between the favorited team and the underdog team after the game was over
# ##### As I said when I created the column, if it is  (-) then the favorite team won.  If (+) then the non-favorite team won.

# In[11]:


X = game_data[['spread_favorite','over_under_line','is_home','is_dome','schedule_playoff']]
Y = game_data[["actual_spread"]]
#I will use some of the data to train, and some of the data to test on.
#I have chosen to use 20% to test on and 80% to train on (I found this is convention)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , train_size = 0.8, random_state = 1)

regress = linear_model.LinearRegression()
regress.fit(X_train, Y_train)

print('Intercept: \n', regress.intercept_)
print('Coefficients: \n', regress.coef_)


# In[12]:


regress = linear_model.LinearRegression()
regress.fit(X_train, Y_train)
print('Intercept: \n', regress.intercept_)
print('Coefficients: \n', regress.coef_)
predictedSpread = regress.predict(X_test)


# In[13]:


Y_predict = X_test.drop(['over_under_line', 'is_home', 'is_dome', 'schedule_playoff'], axis = 1)
display(Y_predict)


# In[14]:


##########
#HERE I DEFINE A FUNCTION TO PRINT EVALUATION METRICS FOR IMBALANCED DATA:

def print_results(headline, true_value, pred):
    print(headline)
    print("Accuracy: {}".format(accuracy_score(true_value, pred)))
    print("Precision: {}".format(precision_score(true_value, pred)))
    print("Recall: {}".format(recall_score(true_value, pred)))
    print("F Score: {}".format(f1_score(true_value, pred)))


# In[15]:


#Further evaluation of my data:


# In[16]:



print(sum(Y_train['actual_spread']>0))
print(sum(Y_train['actual_spread']<0))


# In[17]:


#Looks like in my training data, the favorite team is winning about 1/3 of the time
#This might be causing some unbalance in how it predicts


# In[18]:


#I've decided to just straight up predict a win or a loss of the bet

#Regression:

X = game_data[['spread_favorite','over_under_line','is_home','is_dome','schedule_playoff']]
Y = game_data[['winning_bet']]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , train_size = 0.8, random_state = 1)
LogisticRegression(solver='lbfgs', max_iter=1000)
clf = LogisticRegression().fit(X_train, Y_train.values.ravel())
warnings.filterwarnings('ignore')
Y_test_pred = clf.predict(X_test)

print_results("Unbalanced Evaluation Metrics", Y_test, Y_test_pred)


# # Checking for imbalance in spread_favorite
# #### Imbalance in data hurts machine learning, as the predictor will be very likely to predict the most common outcome. 

# # Undersampling Algorithms:
# #### I decided to try a few different algorithms to correct my imbalanced data.
# 
# ### Percision = (number of true positives) / (sum of number of true positives and false positives)
# ### Recall = (number of true positives) / (sum of number of true positives and false negatives)
# #### I am paying attention to the F-score.  This F-score combines the two metrics by calculating the geometric mean 
# ### F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

# ### Let's see if I can improve that F1 score at all:

# In[19]:


#Manually removing data from majority class (wins) until there are the same amount of wins as losses in the data

majority_class_indices = game_data[game_data['winning_bet']==1].index
minority_class_len = len(game_data[game_data['winning_bet']==0])
random_majority_indices = np.random.choice(majority_class_indices, minority_class_len, replace = False)
print(len(random_majority_indices))

minority_class_indices = game_data[game_data['winning_bet']==0].index
under_sample_indices = np.concatenate([minority_class_indices,random_majority_indices])
print(len(under_sample_indices))

under_sample = game_data.loc[under_sample_indices]
under_sample['spread_favorite'].value_counts


# In[20]:


#Now run regression on this new dataset
X = under_sample[['spread_favorite','over_under_line','is_home','is_dome','schedule_playoff']]
Y = under_sample.loc[:, game_data.columns == 'winning_bet']
Xu_train, Xu_test, Yu_train, Yu_test = train_test_split(X,Y , train_size = 0.8, random_state = 1)

LogisticRegression(solver='lbfgs', max_iter=1000)
clf_bal = LogisticRegression().fit(Xu_train, Yu_train.values.ravel())
warnings.filterwarnings('ignore')

Yu_test_pred = clf_bal.predict(Xu_test)
print_results("Balanced Evaluation Metrics", Yu_test, Yu_test_pred)


# ### CHECKING SMOTE VS Near Miss algorithms using pipeline

# In[21]:


classifier = RandomForestClassifier
data = fetch_datasets()


X = game_data[['spread_favorite','over_under_line','is_home','is_dome','schedule_playoff']]
Y = game_data[['winning_bet']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , train_size = 0.80, random_state = 2)


#Normal Model:
pipeline = make_pipeline(classifier(random_state = 42))
model = pipeline.fit(X_train,Y_train)
prediction = model.predict(X_test)


#SMOTE Model:
smote_pipeline = make_pipeline_imb(SMOTE(random_state = 4),classifier(random_state = 42))
smote_model = smote_pipeline.fit(X_train,Y_train)
smote_prediction = smote_model.predict(X_test)

# classification report
print()
print(classification_report(Y_test, prediction))
print(classification_report_imbalanced(Y_test, smote_prediction))
print()
print('Normal Pipeline Score {}'.format(pipeline.score(X_test, Y_test)))
print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, Y_test)))
print()
print_results("Normal Classification Evaluation Metrics", Y_test, prediction)
print()
print_results("SMOTE classification Evaluation Metrics", Y_test, smote_prediction)


# In[22]:


#I believe I'm not importing this properly and kept getting an error so I left it commented out
#the NearMiss should take the 'random_state' argument but I do not know how to troubleshoot this
#Try running this uncommented and see if you also get this error:
#__init__() got an unexpected keyword argument 'random_state'

'''


#Near Miss Model:
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state=42), classifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train,Y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)

print(classification_report_imbalanced(Y_test, nearmiss_prediction))
print()
print('Near Miss Pipeline Score {}'.format(nearmiss_pipeline.score(X_test, Y_test)))
print()
print_results("Near Miss classification", Y_test, nearmiss_prediction)

'''


# #### Unbalanced Evaluation Metrics
# Accuracy: 0.6394366197183099
# 
# Precision: 0.6420454545454546
# 
# Recall: 0.9912280701754386
# 
# ##### F Score: 0.7793103448275862
# 
# #### Manually Balanced Evaluation Metrics
# 
# Accuracy: 0.6491695963147048
# 
# Precision: 0.655488198116858
# 
# Recall: 0.9579641847313854
# 
# F Score: 0.7783734109358248
# 
# #### Normal Classification Evaluation Metrics
# 
# Accuracy: 0.5981220657276995
# 
# Precision: 0.6691729323308271
# 
# Recall: 0.7650429799426934
# 
# F Score: 0.713903743315508
# 
# #### SMOTE classification Evaluation Metrics
# 
# Accuracy: 0.5727699530516432
# 
# Precision: 0.6738197424892703
# 
# Recall: 0.6747851002865329
# 
# F Score: 0.674302075876879

# In[23]:


#What it looks like is that I was wrong about the imbalance of the data.  
#The highest F score belongs to the first unbalanced set
#I will use this original, un-tampered-with data to train my predictions


# # Final Predictor:

# ### Created using tkinter

# In[24]:


#Run Me :)

win = tk.Tk()
win.geometry("300x399")

label1 = ttk.Label(win, text = 'Enter the spread: ')
label1.grid(row=0,column=1, sticky = tk.W, pady = 5, padx = 5)

label2 = ttk.Label(win, text = 'Enter the Over/Under Line: ')
label2.grid(row=1,column=1, sticky = tk.W, pady = 5, padx = 5)

label3 = ttk.Label(win, text = 'Is the favorite the home team? ')
label3.grid(row=2,column=1, sticky = tk.W, pady = 5, padx = 5)

label4 = ttk.Label(win, text = 'Is the stadium in a dome? ')
label4.grid(row=3,column=1, sticky = tk.W, pady = 5, padx = 5)

label5 = ttk.Label(win, text = 'Is this a playoff game? ')
label5.grid(row=4,column=1, sticky = tk.W, pady = 5, padx = 5)

var1 = tk.StringVar()
var1_ent = ttk.Entry(win, width = 10, textvariable = var1)
var1_ent.grid(row=0, column= 2)
var1.set('')
var2 = tk.IntVar()
var2_ent = ttk.Entry(win, width = 10, textvariable = var2)
var2_ent.grid(row=1, column= 2)
var2.set('')

var3 = tk.IntVar()
ishomebtn1 = ttk.Radiobutton(win, text = 'Yes', variable = var3, value = 1)
ishomebtn1.grid(row = 2, column = 2)
ishomebtn2 = ttk.Radiobutton(win, text = 'No', variable = var3, value = 0)
ishomebtn2.grid(row = 2, column = 3)

var4 = tk.IntVar()
isdomebtn1 = ttk.Radiobutton(win, text = 'Yes', variable = var4, value = 1)
isdomebtn1.grid(row = 3, column = 2)
isdomebtn2 = ttk.Radiobutton(win, text = 'No', variable = var4, value = 0)
isdomebtn2.grid(row = 3, column = 3)

var5 = tk.IntVar()
isplayoffbtn1 = ttk.Radiobutton(win, text = 'Yes', variable = var5, value = 1)
isplayoffbtn1.grid(row = 4, column = 2)
isplayoffbtn2 = ttk.Radiobutton(win, text = 'No', variable = var5, value = 0)
isplayoffbtn2.grid(row = 4, column = 3)

txt = tk.Text(win, width = 25, height = 10)
txt.grid(row = 7, column = 0, columnspan = 4, pady = 12)

def action():
    if var1.get().lstrip('-+').replace('.', '', 1).isdigit() and float(var1.get())<= 0:
        global input_spread
        input_spread = float(var1.get())
    
        global input_OL
        input_OL = var2.get()

        global input_ishome
        if var3.get() == 1:
            input_ishome = "Yes"
        else:
            input_ishome = "No"

        global input_isdome
        if var4.get() == 1:
            input_isdome = "Yes"
        else:
            input_isdome = "No"

        global input_isplayoff
        if var5.get() == 1:
            input_isplayoff = "Yes"
        else:
            input_isplayoff = "No"
            
#make sure to run the regression with the correctly balanced/trained data (clf)
        predictionbin = clf.predict([[input_spread,input_OL, var3.get(),var4.get(),var5.get()]])
        if predictionbin == 1:
            prediction = "You should bet on this game"
        else:
            prediction = "You should not bet on this game"
        txt.delete(0.0, 'end')
        txt.insert(0.0, "Current spread: {}\n"
                   "Current Over/Under: {}\n"
                  "Home game? {}\n"
                  "Played in a dome? {}\n"
                  "Playoff game? {}\n\n"
                   "ADVICE: {}".format(input_spread,input_OL, input_ishome,input_isdome,input_isplayoff, prediction))
        return True
    else:
        messagebox.showwarning("Wrong input", "Spread must be a negative number")
        var1.set("")
        return False


submit_button = ttk.Button(win, text = "SUBMIT", command = action)
submit_button.grid(row=5, column =2)

exit_button = ttk.Button(win, text= "Close", command = win.destroy)
exit_button.grid(row = 10, column = 2)

win.title('PREDICTOR')
win.resizable(False,False)
win.mainloop()


# # What I've come to learn:
# #### After analyzing and running linear regression on my data, I've come to learn that the data I chose was not very accurate at predicting games.
# #### The data seemed to be unbalanced at first, but I think the real issue is that there just might not be a strong correlation to be able to predict using the pipeline and linear regression toolkits I chose.
# #### Future analysis and prediction should also include more variables rather than the 5 I chose to analyze.  I had to stick with 5 because of the nature of the data I was given, but as I take this work further after graduation, I will surely search for more databases and increase the complexity.

# In[ ]:




