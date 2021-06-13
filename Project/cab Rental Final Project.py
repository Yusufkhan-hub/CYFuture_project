#!/usr/bin/env python
# coding: utf-8

# # A Cab Rental Project, predict fare of ride
# **By Yusuf Khan**

# **Imports**
# 
# Before we get started, let's import all the required packages/libraries that we will need for this analysis.

# In[1]:


import os
import folium
import colorcet as cc
import datashader as ds
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sin,cos,asin,sqrt,radians
from collections import defaultdict, OrderedDict
from folium.plugins import HeatMapWithTime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse,r2_score
from sklearn.svm import SVR
from sklearn import preprocessing


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15,10)


# ## The Data
# This data contains various attributes, it collected from pilot startup
# :
# 
# 
# Number of attributes:
# 
# · pickup_datetime - timestamp value indicating when the cab ride started.
# 
# · pickup_longitude - float for longitude coordinate of where the cab ride started.
# 
# · pickup_latitude - float for latitude coordinate of where the cab ride started.
# 
# · dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# 
# · dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# 
# · passenger_count - an integer indicating the number of passengers in the cab ride.

# In[3]:


df=pd.read_csv("H:\\Yusuf\\Edwisor\\Project\\train_cab.csv")


# In[4]:


df.describe()


# We can clearly see that there are outlier and missing values in passenger_count such as max is 5345 and min is 0. Which is not possible. Let's move further, we will resolve them later

# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df1=df.copy()


# Check the count of Null values in DataFrame!

# In[8]:


df1.isnull().sum()


# In[9]:


df1.head()


# # Data Wrangling.

# Splitting pickup_datetime feature in date, time for future analysis and prediction

# In[10]:


df1[['date','time','utc']]=df1.pickup_datetime.str.split(" ",expand=True)


# In[11]:


df1.head()


# In[12]:


df1.info()


# The below code is commented because if we run this code it will through an error because at instance(1327) the value is missing.

# In[13]:


# df1['date_time'] = df1[['date','time']].apply(lambda x: " ".join(x), axis=1)# It is now showing an error of missing values at row 1327


# In[14]:


df1.iloc[1327]


# In[15]:


df1=df1.drop(labels=[1327],axis=0)# Droping an missing value of time and utc at instance=1327


# Merging two column to use it as datetime feature

# In[16]:


df1['date_time'] = df1[['date','time']].apply(lambda x: " ".join(x), axis=1)


# In[17]:


df1.head()


# In[18]:


df1=df1.drop(columns=['date','time','utc','pickup_datetime'],axis=1)


# In[19]:


df1.head()


# To extract information from date_time column we need to separate it into month, year, time, day, weekdays, hour.

# In[20]:


df1['date_time'] = pd.to_datetime(df1['date_time'])
df1['month'] = df1['date_time'].dt.month
df1['day'] = df1['date_time'].dt.day
df1['time'] = df1['date_time'].dt.time
df1['year'] = df1['date_time'].dt.year
df1['weekdays'] = df1['date_time'].dt.dayofweek
df1['hour'] = df1['date_time'].dt.hour


# In[21]:


df1.head()


# # Trip Distance

# In[22]:


#Function use for calculating distance in km from longitude and latitude
def distance(pickup_latitude,dropoff_latitude,pickup_longitude,dropoff_longitude):
    
    #converting latitudes and longitudes degree values in radian.
    pickup_latitude = radians(pickup_latitude)
    dropoff_latitude = radians(dropoff_latitude)
    pickup_longitude = radians(pickup_longitude)
    dropoff_longitude = radians(dropoff_longitude)
    
    #arranging formula
    distance_latitude = pickup_latitude - dropoff_latitude
    distance_longitude = pickup_longitude - dropoff_longitude
    
    #calculating a value
    a = sin(distance_latitude/2)**2 + cos(pickup_latitude)*cos(dropoff_latitude)*sin(distance_longitude/2)**2
    #calculating c value
    c = 2*asin(sqrt(a))
    # radius of eart is 6371 km
    radius = 6371
    return(c*radius)
# distance(a,b,c,d)


# # Trip Distance is new feature with calculated distance

# In[23]:


df1['trip_distance']=df1.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


# In[24]:


df1.drop(columns=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','date_time','time'],axis=1,inplace=True)


# # Final Dataset will be use for analysis and prediction

# In[25]:


df1.head()


# In[26]:


df1.shape[1]


# In[27]:


#converting years into words
month_map = {
    1:'january',
    2:'februrary',
    3:'march',
    4:'april',
    5:'may',
    6:'june',
    7:'july',
    8:'august',
    9:'september',
    10:'october',
    11:'november',
    12:'december'
}
#converting week days into words
weekend_map = {
    0: 'monday',
    1: 'tuesday',
    2: 'wednesday',
    3: 'thursday', 
    4: 'friday', 
    5: 'saturday',
    6: 'sunday'
}
df1['month'] = df1['month'].replace(month_map)
df1['weekdays'] = df1['weekdays'].replace(weekend_map)


# In[28]:


df1.head()


# In[29]:


df1['trip_distance'] = df1['trip_distance'].round(2)


# In[30]:


df1.head()


# In[31]:


df1.shape


# In[32]:


# Check for missing Value


# In[33]:


#missing values check
num_missing = df1.isnull().sum()
num_missing


# In[34]:


#drop missing value it will disturb the pattern
df1=df1.dropna(how='any')


# In[35]:


df1.isnull().sum()


# In[36]:


#Check any anomali in hour column
hour_check = df1['hour'].between(0,23,inclusive=True).all()
assert hour_check, "INvalid hour value exist"

#Check any anomali in day column
day_check=df1['day'].between(1,31,inclusive=True).all()
assert day_check, "Invalid days values exist"

print("all values ranges valid")


# In[37]:


#Check any anomali in month column
month_check = not set(df1['month']).difference(month_map.values())
assert month_check, 'Invalid month values exist'

#Check any anomali in weekdays column
weekday_check = not set(df1['weekdays']).difference(weekend_map.values())
assert weekday_check, 'Invalid weekday values exist'

print("values categories valid")


# In[38]:


#duplicate values
duplicates_df1 = df1[df1.duplicated(keep=False)]
duplicates_df1


# Now we have checked that datasets had some missing values and we removed it because there were less than 1%. So this is simple instead of replacing them with mean or mode value.  It was better to remove them.

# Lets check for outliers for each feauter, and for label as well.

# # Checking Outliers for each feature

# In[39]:


df1.shape


# In[40]:


df1.columns


# In[41]:


df1.passenger_count.describe()


# # Haha!! We can see that the maximum value is 5345. Which is not possible. We know that in India maximum passenger is 6 for taxi or cab. Above 6, need to hire traveller or mini bus. 

# So, lets set the limit of the passenger_count feature from 1 to 6. And check how many is not in the range.

# In[42]:


df1[(df1['passenger_count']<1) | (df1['passenger_count']>6)].count()


# We see that there are 77 outliers in this feautre, we will remove them. clearly. Out of 16k it is less than 1%.

# In[43]:


df1=df1[(df1['passenger_count']>0) & (df1['passenger_count']<7)]


# In[44]:


df1.passenger_count.unique()


# Haha!!,, we again found anomalies. It is good that there are only two. And we are gonna remove them

# In[45]:


df1.passenger_count.value_counts()


# In[46]:


df1[(df1['passenger_count']==1.3) & (df1['passenger_count']==0.12)]


# we got the instances of the anamolies.. we can remove them.

# In[47]:


df1.drop(labels=[8790],axis=0,inplace=True)
df1.drop(labels=[8862],axis=0,inplace=True)


# In[48]:


df1.passenger_count.value_counts()


# # Lets check Outliers for trip_distance.

# In[49]:


df1[['trip_distance']].describe ()


# We can see that the maximum value is 8667.54 km, which may be possible but it consist various circumtances like how many passenger and what is the fare amount. so lets check

# In[50]:


df1[df1['trip_distance']==8667.500]


# It clear, that this is outlier. so let's check how many of them

# # EDA of trip_distance for outliers
# 

# In[51]:


df1.trip_distance.hist(bins=40,range=[10,15],facecolor='red',rwidth=0.90)


# In[52]:


df1.trip_distance.hist(bins=40,range=[15,25],facecolor='red',rwidth=0.90)


# In[53]:


df1.trip_distance.hist(bins=40,range=[25,35],facecolor='green',rwidth=0.90)


# In[54]:


df1.trip_distance.hist(bins=40,range=[35,100],facecolor='green',rwidth=0.90)


# In[55]:


df1.trip_distance.hist(bins=40,range=[100,1000],facecolor='green',rwidth=0.90)


# In[56]:


df1.trip_distance.hist(bins=40,range=[1000,9000],facecolor='green',rwidth=0.90)


# Well, we see that trip distance ranges from, first 0 to 45 km. It can be possible that cab service is hire for such distance, next range is 90 to 110 km which is also possible but with some critieria like passenger count and fare amount. So lets check them whether they are valid or outliers.

# In[57]:


df1[df1['trip_distance']>90].count()


# After checking trip distance above 90km, it is clear that all instances are outlier which will drastically destroy our training dataset. So we will remove them and keep them below 90 km. and we also noticed that there are some intances are 0. it is not possible that, without moving a cab can charge fare. so lets check them as well

# In[58]:


# checking trip_distance of 0 km.
df1[df1['trip_distance']==0].count()


# In[59]:


df1[df1['trip_distance']==0]


# Now, lets check the quantile of trip_distance feature.

# In[60]:


out=df1.trip_distance.quantile(0.99780)
out


# In[61]:


out1=df1.trip_distance.quantile(0.99790)
out1


# In[62]:


df1[df1['trip_distance']>out].count()


# In[63]:


df1[df1['trip_distance']>out1].count()


# So, we see that there are figures of out and out1 variable. We can see that more than 99.78% data are below 40km. so, to make our data meaning full, we have to remove outliers. It is because the fare_amount of such distance are not justifying any pattern or association with others features.

# In[64]:


# Removing outliers for distance
df1=df1[(df1['trip_distance']>0) &(df1['trip_distance']<out)]


# In[65]:


df1.shape


# In[66]:


df1.columns


# So, till now we verified and made datasets meaningfull, now there is no missing values, no any outliers in any features.
# At last it is necessary to check  the label of datasets. There might be any faulty entry.

# In[67]:


df1.fare_amount.describe()


# HaHa!, it is because fare_amount is object type. lets change it.

# In[68]:


df1['fare_amount']=df1['fare_amount'].astype('float64')# actually it has anomali


# During converting we found that there is an intance which is actually anomali. so lets remove it('430-')

# In[69]:


df1[df1['fare_amount']=='430-']


# In[70]:


df1.drop(labels=[1123],axis=0,inplace=True)


# In[71]:


df1.fare_amount.describe()


# In[72]:


df1['fare_amount']=df1['fare_amount'].astype('float64')


# In[73]:


df1.fare_amount.describe()


# The maximum and minimum values are definetly doubtfull, we need to check them.

# In[74]:


df1[df1['fare_amount']<0].count()


# In[75]:


Q98=df1['fare_amount'].quantile(0.980)
Q985=df1['fare_amount'].quantile(0.985)
Q99=df1['fare_amount'].quantile(0.990)
Q995=df1['fare_amount'].quantile(0.995)
print("The 98% of fare_amount is below than {}, and the 98.50% is below than {},and the 99.0% is below than {},and the 99.5% is below than {}".format(Q98,Q985,Q99,Q995))


# In[76]:


df1.describe()


# In[77]:


# Checking count for various quantile of fare_amount


# In[78]:


df1[df1['fare_amount']>Q98].count()


# In[79]:


df1[df1['fare_amount']>Q985].count()


# In[80]:


df1[df1['fare_amount']>Q99].count()


# In[81]:


df1[df1['fare_amount']>Q995].count()


# In[82]:


df1[df1['fare_amount']>Q98]


# We can see that above 98% quantile the data has anomalies,below the 98% quantile we can say that pattern may recognise association between features.

# # Removing outliers of fare_amount

# In[83]:


df1['fare_amount']=df1[(df1['fare_amount']>0) & (df1['fare_amount']<Q98)]


# In[84]:


df1.describe()


# In[85]:


df1.fare_amount.describe()


# In[86]:


df1.info()


# In[87]:


df1['fare_amount']=df1.fare_amount.astype('float64')


# In[88]:


df1.info()


# While describing it is checked that trip distance have some instances equal to 0.01 km. lets check them

# In[89]:


df1[df1['trip_distance']==0.01].count()


# In[90]:


df1[df1['trip_distance']==0.01]


# Here is the two scnerios which is very clearly visible that with 0.01km trip distance the fare amount is 2.5 and above, so this might be possible that for 0.01km trip distance the fare_amount is 2.5, so lets remove the above fare them

# Or why not we should remove all of them because there is no chance that somebody hire cab for such distance

# But it will train our model with such distance as well, so just remove the fare_amount above than 2.5

# In[91]:


df1[(df1['fare_amount']==2.5) & (df1['trip_distance']==0.01)].shape


# In[92]:


df1=df1.drop(df1[(df1['fare_amount']!=2.5) & (df1['trip_distance']==0.01)].index)


# In[93]:


df1[df1['trip_distance']==0.01]


# In[94]:


df1.shape


# Now it is not fare as well, because the fare_amount with 2.5 and trip_distance with 0.01 the passenger_count is varying, lets set them

# In[95]:


df1[(df1['fare_amount']==2.5) & (df1['trip_distance']==0.01) & (df1['passenger_count']!=1)]


# In[96]:


df1=df1.drop(df1[(df1['fare_amount']==2.5) & (df1['trip_distance']==0.01) & (df1['passenger_count']!=1)].index)


# In[97]:


df1.fare_amount.shape[0]


# In[98]:


df1[df1['fare_amount']=='Nan']


# In[99]:


df1.isnull().any()


# In[100]:


df1=df1.dropna(how='any')


# In[101]:


df1.isnull().any()


# In[102]:


df1.shape


# # Now our data is ready for EDA and modeling or applying algorith , for training and testing

# In[103]:


num_df1=df1.shape[0]
num_days = len(df1[['month','day']].drop_duplicates())
daily_avg = np.round(num_df1/num_days,0)
stats_raw = 'Number of instances of df1: {}\nNumber of days: {}\nDaily avg pickup: {}'
print(stats_raw.format(num_df1,num_days,daily_avg))


# # Total pickups per month

# In[104]:


monthly_pickups = df1['month'].value_counts(ascending=True)
monthly_pickups.plot(kind='bar',rot=0)
plt.title('Total Pickups per month')
plt.xlabel("Month")
plt.ylabel('pickups in millions')


# # Total pickups per year

# In[105]:


monthly_pickups = df1['year'].value_counts(ascending=True)
monthly_pickups.plot(kind='bar',rot=0)
plt.title('Total Pickups per year')
plt.xlabel("Years")
plt.ylabel('pickups in millions')


# # Total pickups per week day

# In[106]:


monthly_pickups = df1['weekdays'].value_counts(ascending=True)
monthly_pickups.plot(kind='bar',rot=0)
plt.title('Total Pickups per week day')
plt.xlabel("Days")
plt.ylabel('pickups in millions')


# # Total pickups per day
# 

# In[107]:


monthly_pickups = df1['hour'].value_counts().sort_index()
monthly_pickups.plot(kind='bar',rot=0)
plt.title('Total Pickups per hour')
plt.xlabel("Timing in hour")
plt.ylabel('pickups in millions')


# # Check for maximum pickup of the data in a day 
# 

# In[108]:


daily_max_pickup = df1.groupby(['month','day'])['hour'].count()
print("The bussiest day was: {}".format(daily_max_pickup.idxmax()))
print("Number of pickups: {}".format(daily_max_pickup.max()))


# In[109]:


df.shape


# In[110]:


df2=df1.copy()


# It is because for training a model it needs a numeric values.

# In[111]:


month_map = {
    'january':1,
    'februrary':2,
    'march':3,
    'april':4,
    'may':5,
    'june':6,
    'july':7,
    'august':8,
    'september':9,
    'october':10,
    'november':11,
    'december':12
}

weekend_map = {
    'monday':0,
    'tuesday':1,
    'wednesday':2,
    'thursday':3, 
    'friday':4, 
    'saturday':5,
    'sunday':6
}
df2['month'] = df2['month'].replace(month_map)
df2['weekdays'] = df2['weekdays'].replace(weekend_map)


# In[112]:


df2.head()


# # Feature Selection

# In[113]:


X=df2.drop(columns=['fare_amount'],axis=1)
y=df2.fare_amount


# # Model Selection

# In[114]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[115]:


avg_fare=round(np.mean(y_train),2)
baseline_pred=np.repeat(avg_fare,y_test.shape[0])
baseline_rmse=np.sqrt(mse(baseline_pred,y_test))
print("Baseline RMSE of Validation data: ",baseline_rmse)


# # Applying Linear Regression

# In[116]:


lm=LinearRegression()
lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)
rmse_test=np.sqrt(mse(y_test,y_pred))
y_pred_train=lm.predict(X_train)
rmse_train = np.sqrt(mse(y_train,y_pred_train))
lm_variance=abs(rmse_train-rmse_test)
print("The RMSE of test datasets is: ",rmse_test)
print("The RMSE of train datasets is: ",rmse_train)
print("The variance is: ",lm_variance)
print("The r2_score is: ",r2_score(y_test,y_pred))


# # Applying Random Forest

# In[117]:


rf=RandomForestRegressor(n_estimators = 40,random_state=82,n_jobs=-1)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
rmse_test=np.sqrt(mse(y_test,y_pred))
y_pred_train = rf.predict(X_train)
rmse_train = np.sqrt(mse(y_train,y_pred_train))
variance = abs(rmse_train-rmse_test)
print("The RMSE of Train dadasets is: ",rmse_train)
print("The RMSE of Test dadasets is: ",rmse_test)
print("The Variance is: ",variance)
print("The r2_score is: ",r2_score(y_test,y_pred))


# # Applying SVR

# In[118]:


from sklearn.svm import SVR
svm=SVR(C=10,gamma='auto')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
rmse_train=np.sqrt(mse(y_test,y_pred))
print("the rmse of train datasets is: ",rmse_train)
print("The r2_score is: ",r2_score(y_test,y_pred))


# # Applying Various Ensemble techniques

# In[ ]:


#applying AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
ada_boost=AdaBoostRegressor()
ada_boost.fit(X_train,y_train)
ada_boost.score(X_test,y_test)


# In[118]:


#applying BaggingRegressor
from sklearn.ensemble import BaggingRegressor
bagg_reg=BaggingRegressor(random_state=1)
bagg_reg.fit(X_train,y_train)
bagg_reg.score(X_test,y_test)


# In[119]:


#applying GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[120]:


#applying xtream gradient boost
import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[121]:


import lightgbm as lgb
train_data = lgb.Dataset(X_train,label=y_train)
params= {'learning_rate':0.001}
model = lgb.train(params,train_data,100)
rmse=mse(y_pred,y_test)**0.5
rmse


# # Hyper parameter tuning using GridSearchCV
# 

# In[122]:


from sklearn.tree import DecisionTreeRegressor
model_params = {
    'LinearRegression': {
        'model': LinearRegression(n_jobs=-1),
        'params' : {
        }  
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=82,n_jobs=-1),
        'params' : {
            'n_estimators': [30,40,50]
        }
    },
    'SVR' : {
        'model': SVR(gamma='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(),
        'params': {
            'n_estimators':[40,50,65],
            'learning_rate':[1.0,1.2,1.4]
        }
    },
    'BaggingRegressor': {
        'model': BaggingRegressor(random_state=1),
        'params': {
            'n_estimators':[10,15,20]            
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(),
        'params': {
            'loss':['ls','lad', 'huber'],
            'learning_rate':[0.1,0.2,0.3],
            'n_estimators':[100,150,170],
            'alpha':[0.9,1.0,1.2],            
        }
    },
    'XGBRegressor': {
        'model': xgb.XGBRegressor(),
        'params': {
            'n_estimators': [30,40,50]            
        }
    }
}


# In[ ]:


from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_best_algorith = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_best_algorith


# # It is easy to say now which algorithm is best for our dataset to predict the fare charges. Random Forest and Gradient Boosting Regressor are trained well. which is because both uses decision tree. I did not use Decision Tree because Random forest and various others ensemble algorithms already used decsion tree with various parameter.

# In[ ]:


get_ipython().system('pip install ipython')


# In[ ]:


pip install nbconvert


# In[ ]:


get_ipython().system('ipython nbconvert --to script Cab_Fare_Prediction_Project.ipynb')


# In[ ]:


get_ipython().system('pip install pipreqs')


# In[ ]:


get_ipython().system('pipreqs ./')


# In[126]:


import pickle


# In[127]:


GBR=GradientBoostingRegressor()
GBR.fit(X_train,y_train)


# In[129]:


score=GBR.score(X_test,y_test)*100


# In[130]:


score


# In[138]:


pickle.dump(GBR,open("ML_Cab_Fare_price_pred_model.sav","wb"))


# In[ ]:




