#!/usr/bin/env python
# coding: utf-8

# In[98]:


# Importing necessary libraries
import sklearn
import numpy             as  np
import pandas            as  pd
import matplotlib.pyplot as  plt
import seaborn           as  sns
from sklearn.preprocessing import StandardScaler

# Importing High yield corporate bonds dataset using pandas
HY_CB_df = pd.read_csv("HY_corp_bond_data.csv")


# In[99]:


HY_CB_df.head()


# In[100]:


HY_CB_df.shape


# In[101]:


HY_CB_df.dtypes


# In[102]:


# Summary Statistics of the entire dataset

HY_CB_df.describe()


# In[103]:


# Including variables to do EDA

X = HY_CB_df[['Coupon','LiquidityScore','n_trades','volume_trades',
             'total_median_size','total_mean_size','n_days_trade',
             'days_diff_max','percent_intra_dealer','percent_uncapped',
            'Client_Trade_Percentage','weekly_mean_volume',
            'weekly_median_volume','weekly_max_volume','weekly_min_volume',
            'weekly_mean_ntrades','weekly_median_ntrades']]

y = HY_CB_df[['Industry','IN_ETF','bond_type']]


# In[104]:


# Scaling the data
scaler = StandardScaler()
scaler.fit(X)
X_std  = scaler.transform(X)
X_std  = pd.DataFrame(X_std,columns = X.columns)


# In[105]:


print (X.head())
print (X_std.head())


# In[106]:


y.head()


# In[107]:


# Summary Statistics

X_std.describe()


# In[108]:


# Skewness

X_std.skew()


# In[109]:


# Boxplots to visualise outliers

sns.boxplot(data = X_std, orient = 'h')


# In[110]:


# Swarmplots 

sns.swarmplot(y = 'LiquidityScore', data = X_std)


# In[111]:


sns.swarmplot(y = 'Client_Trade_Percentage', data = X_std)


# In[112]:


# pairwise scatter plots 

X_std.plot.scatter(x = 'LiquidityScore', y = 'weekly_median_ntrades', 
                   c = 'DarkBlue')
X_std.plot.scatter(x = 'LiquidityScore', y = 'n_trades', c = 'Green')

X_std.plot.scatter(x = 'LiquidityScore', y = 'Client_Trade_Percentage', 
                   c = 'Red')

X_std.plot.scatter(x = 'weekly_median_ntrades', y = 'Client_Trade_Percentage', 
                   c = 'Orange')

X_std.plot.scatter(x = 'percent_intra_dealer', y = 'weekly_median_volume', 
                   c = 'Purple')



# In[114]:


# Correlation heat map to determine the variables we need to do analysis on

corr_mat = pd.DataFrame(X.corr())
plt.pcolor(corr_mat)
plt.show()


# In[115]:


sns.set(font_scale = 1.2)
heat_map = sns.heatmap(corr_mat, 
                       cbar=True, 
                       annot=True, 
                       square=True, 
                       fmt='.2f',
                       annot_kws={'size':5},
                      yticklabels=True,
                      xticklabels=True)
plt.show()


# In[116]:


# Boxplots using bond_type and IN_ETF

sns.boxplot(x='bond_type', y='weekly_median_ntrades', data=HY_CB_df)


# In[117]:


sns.boxplot(x='bond_type', y='percent_intra_dealer', data=HY_CB_df)


# In[118]:


sns.boxplot(x='IN_ETF', y='weekly_median_ntrades', data=HY_CB_df)


# In[119]:


sns.boxplot(x='IN_ETF', y='percent_intra_dealer', data=HY_CB_df)


# In[120]:


# Histogram of bond_type

y.hist(column='bond_type')


print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is: rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# End



