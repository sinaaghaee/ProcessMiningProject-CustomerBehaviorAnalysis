#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/Sharif_Logo.png?raw=true" width="250" alt="cognitiveclass.ai logo"  />
# </center>
# 
# # Process Mining Project - Customer Behavior Analysis Based on Click Data
# 

#  ## Course Info:
#  
#  **Student/Analyst:**  Sina Aghaee <br>
#  **Course:** Business Process Management 1400-1401 <br>
#  **Institution:** Sharif University of Technology, Department of Industrial Engineering <br>
#  **Instructor:**  Dr. Erfan Hassannayebi 
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#about_dataset">About the Dataset and the Paper</a></li>
#         <li><a href="#pre-processing">Pre-processing</a></li>
#         <li><a href="#Process_discovery">Challenge 1: Process Discovery</a></li>
#         <li><a href="#Customer_behaviour">Challenge 2: Customer Behaviour</a></li>
#         <li><a href="#Transition_expensive_channels">Challenge 4: Transition to More Expensive Channels</a></li>
#     </ol>
# </div>
# <br>
# <hr>

# <div id="about_dataset">
#     <h2>About the Dataset and the Paper</h2>
# 
# </div>
#     
# Our data belongs to UWV and presented in BPI 2016 Challenge.
# 
# <h3>About UWV</h3>
#     
# UWV (Employee Insurance Agency) is a Dutch autonomous administrative authority (ZBO) and is commissioned by the Ministry of Social Affairs and Employment (SZW) to implement employee insurances and provide labour market and data services in the Netherlands.
# 
# The Dutch employee insurances are provided for via laws such as the WW (Unemployment Insurance Act), the WIA (Work and Income according to Labour Capacity Act, which contains the IVA (Full Invalidity Benefit Regulations), WGA (Return to Work (Partially Disabled) Regulations), the Wajong (Disablement Assistance Act for Handicapped Young Persons), the WAO (Invalidity Insurance Act), the WAZ (Self-employed Persons Disablement Benefits Act), the Wazo (Work and Care Act) and the Sickness Benefits Act.
# 
# <h3>Data</h3>
#     
# The data in this collection pertains to customer contacts over a period of 8 months and UWV is looking for insights into their customers' journeys. The data is focused on customers in the WW (unemployment benefits) process.
# 
# Data has been collected from several different sources, namely:
# 
# 1) Click data from the site www.werk.nl collected from visitors that were not logged in: 
# * [BPI Challenge 2016: Clicks NOT Logged In](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Clicks_NOT_Logged_In/12708596/1)
# 
# 
# 2) Click data from the customer specific part of the site www.werk.nl (a link is made with the customer that logged in):
# * [BPI Challenge 2016: Clicks Logged In](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Clicks_Logged_In/12674816/1)
#  
#     
# 3) Werkmap Message data, showing when customers contacted the UWV through a digital channel:
# * [BPI Challenge 2016: Questions](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Questions/12687320/1)
#     
#     
# 4) Call data from the call center, showing when customers contacted the call center by phone:
# * [BPI Challenge 2016: Werkmap Messages](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Werkmap_Messages/12714569/1)    
#     
# 5) Complaint data showing when customers complained:
# * [BPI Challenge 2016: Complaints](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Complaints/12717647/1)
#       
# 
# <h3>Paper</h3>
#     
# The Following is the paper that we chose as our base paper and we will try to analyze the data in the same way:
# * [Identification of Distinct Usage Patterns and Prediction of Customer Behavior](https://www.win.tue.nl/bpi/lib/exe/fetch.php?media=2016:bpic2016_paper_1.pdf) by Sharam Dadashnia, Tim Niesen, Philip Hake, Peter Fettke, Nijat Mehdiyev and Joerg Evermann    
# 
# **Note**: The Author of the above article didn't use the Not_Logged_In dataset in the analysis since it doesn't contain any customer ID. We will do the same since this data won't give us much information about customers' behavior over time and in different sessions.        
#     
#     

# <div id="pre-processing">
#     <h2>Pre-processing</h2>
#     
# </div>
# 

# ### Reading and cleaning the data
# 

# We will import the required libraries for pre-processing, data analysis, process mining, and discovery in the next cell:

# In[1]:


# The following library and code is to igonre warnings 
import warnings
warnings.filterwarnings('ignore')

# python ######################################################################
import sys
import os
import datetime

# basics ######################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# widgets #####################################################################
import ipywidgets as widgets
from ipywidgets import interact

# process mining ##############################################################
import pm4py

# object.log
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer

# object.conversion
from pm4py.objects.conversion.dfg import converter as dfg_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter

# algo.discovery
from pm4py.algo.discovery.alpha import variants
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

# algo.filtering
from pm4py.algo.filtering.log.auto_filter.auto_filter import apply_auto_filter

# algo.conformance
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
# vizualization
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# statistics
from pm4py.statistics.traces.log import case_statistics

# util
from pm4py.util import vis_utils


# Reading the data and saving it in a datafram:

# In[2]:


clicks_logged_in = pd.read_csv('BPI2016_Clicks_Logged_In.csv', sep = ';', encoding = 'latin', parse_dates=['TIMESTAMP'] )
clicks_logged_in.shape


# There is about 7 milloin Records in clicks_logged_in dataset!!! Now let's check out a sample of records in the dataset, here is the first 10 rows:

# In[3]:


clicks_logged_in.head(10)


# We only need these columns for our analysis and process discovery:
# 
# * CustomerID
# * SessionID
# * AgeCategory
# * Gender
# * TIMESTAMP
# * PAGE_NAME
# 
# So we will copy these cloumns and save them in a new variable named **"clicks_logged_in_SelectedColumns"**

# In[4]:


clicks_logged_in_SelectedColumns = clicks_logged_in[['CustomerID','SessionID', 'AgeCategory', 'Gender', 'TIMESTAMP', 'PAGE_NAME']].copy()


# Now let's check the types of each column:

# In[5]:


clicks_logged_in_SelectedColumns.dtypes


# Everything seems normal!

# Let's check to see if we have any NA values:

# In[6]:


clicks_logged_in_SelectedColumns.isna().sum()


# Lucky us! there is no NA value in the dataset.

# Let's see how many activites we have in total:

# In[7]:


clicks_logged_in_SelectedColumns['PAGE_NAME'].nunique()


# WOW!!! We have 600 activities!!!! That's too much! We will probably work with the most frequent ones, and since our activities are the web pages that users visited, it makes sense!

# Here we export the cleaned data in CSV format for further analysis(We will use Microsoft Power BI for some visualization, so that's why we export data here and later):

# In[8]:


clicks_logged_in_SelectedColumns.to_csv ('clicks_logged_in_SelectedColumns.csv', index = False)


# ### Segmentation of customer basis with respect to demographic features
# 
# In this part, we will segment our data in the same way our chosen article has done (the article's tables only show the result for the first four );  based on the demographic information. We will segment our data into six different data sets. We export all in CSV format for further analysis (Visualization with PowerBI) :
# 
# * Segment 1: Age 18-29
# * Segment 2: Age 30-39
# * Segment 3: Age 40-49
# * Segment 4: Age 50-65
# * Segment 5: Females
# * Segment 6: Males

# #### Segment 1: Age 18-29

# In[9]:


clicks_logged_in_SelectedColumns_Age18_29 = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['AgeCategory']== '18-29']


# In[10]:


clicks_logged_in_SelectedColumns_Age18_29.head()


# In[11]:


# exporting csv file
clicks_logged_in_SelectedColumns_Age18_29.to_csv ('clicks_logged_in_SelectedColumns_Age18_29.csv', index = False)


# #### Segment 2: Age 30-39

# In[12]:


clicks_logged_in_SelectedColumns_Age30_39 = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['AgeCategory']== '30-39']


# In[13]:


clicks_logged_in_SelectedColumns_Age30_39.head()


# In[14]:


# exporting csv file
clicks_logged_in_SelectedColumns_Age30_39.to_csv ('clicks_logged_in_SelectedColumns_Age30_39.csv', index = False)


# #### Segment 3: Age 40-49

# In[15]:


clicks_logged_in_SelectedColumns_Age40_49 = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['AgeCategory']== '40-49']


# In[16]:


clicks_logged_in_SelectedColumns_Age40_49.head()


# In[17]:


# exporting csv file
clicks_logged_in_SelectedColumns_Age40_49.to_csv ('clicks_logged_in_SelectedColumns_Age40_49.csv', index = False)


# #### Segment 4: Age 50-65

# In[18]:


clicks_logged_in_SelectedColumns_Age50_65 = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['AgeCategory']== '50-65']


# In[19]:


clicks_logged_in_SelectedColumns_Age50_65.head()


# In[20]:


# exporting csv file
clicks_logged_in_SelectedColumns_Age50_65.to_csv ('clicks_logged_in_SelectedColumns_Age50_65.csv', index = False)


# #### Segment 5: Females

# In[21]:


clicks_logged_in_SelectedColumns_Female = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['Gender']== 'V']


# In[22]:


clicks_logged_in_SelectedColumns_Female.head()


# In[23]:


# exporting csv file
clicks_logged_in_SelectedColumns_Female.to_csv ('clicks_logged_in_SelectedColumns_Female.csv', index = False)


# #### Segment 6: Males

# In[24]:


clicks_logged_in_SelectedColumns_Male =  clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['Gender']== 'M']


# In[25]:


clicks_logged_in_SelectedColumns_Male.head()


# In[26]:


# exporting csv file
clicks_logged_in_SelectedColumns_Male.to_csv ('clicks_logged_in_SelectedColumns_Male.csv', index = False)


# ### Segmentations Comparison

# in the next cell, we are going to write a code to compare the segments based on age categories:

# In[27]:


# create a dictionary contaning information about each age category
Segments_Comparison = {'Segment': ['Age category 18-29', 'Age category 30-39', 'Age category 40-49', 'Age category 50-65'],
                     'Number Of Sessions': [clicks_logged_in_SelectedColumns_Age18_29['SessionID'].nunique(), clicks_logged_in_SelectedColumns_Age30_39['SessionID'].nunique(), clicks_logged_in_SelectedColumns_Age40_49['SessionID'].nunique() , clicks_logged_in_SelectedColumns_Age50_65['SessionID'].nunique()] ,
                     'Number Of Customers': [clicks_logged_in_SelectedColumns_Age18_29['CustomerID'].nunique(), clicks_logged_in_SelectedColumns_Age30_39['CustomerID'].nunique(), clicks_logged_in_SelectedColumns_Age40_49['CustomerID'].nunique() , clicks_logged_in_SelectedColumns_Age50_65['CustomerID'].nunique()] ,
                     'Number of Events': [clicks_logged_in_SelectedColumns_Age18_29['SessionID'].count(), clicks_logged_in_SelectedColumns_Age30_39['SessionID'].count(), clicks_logged_in_SelectedColumns_Age40_49['SessionID'].count() , clicks_logged_in_SelectedColumns_Age50_65['SessionID'].count()],
                    }

# convert the dictionary into dataframe and add the sum of each column to the end of dataframe
Segments_Comparison = pd.DataFrame(data=Segments_Comparison)
Segments_Comparison = Segments_Comparison.append(Segments_Comparison[['Number Of Sessions','Number Of Customers','Number of Events' ]].sum(),ignore_index=True)
Segments_Comparison.iloc[4,0] = 'Total'

# we don't want decimals to be diplayed
Segments_Comparison['Number Of Sessions']=Segments_Comparison['Number Of Sessions'].apply('{:,.0f}'.format)
Segments_Comparison['Number Of Customers']=Segments_Comparison['Number Of Customers'].apply('{:,.0f}'.format)
Segments_Comparison['Number of Events']=Segments_Comparison['Number of Events'].apply('{:,.0f}'.format)

Segments_Comparison


# As you can see, the numbers in the above table are precisely the same as the numbers in the following table, which is in our base paper except for the number of events which means the author of the article deleted some records, but they didn't explain which and why.
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/01_Table3.png?raw=true"  />
# </center>

# Segmentation info based on Gender: 

# In[28]:


# create a dictionary contaning information about each gender category
Segments_Comparison = {'Segment': ['Female' ,'Male'],
                     'Number Of Sessions': [clicks_logged_in_SelectedColumns_Female['SessionID'].nunique() , clicks_logged_in_SelectedColumns_Male['SessionID'].nunique()] ,
                     'Number Of Customers': [ clicks_logged_in_SelectedColumns_Female['CustomerID'].nunique() , clicks_logged_in_SelectedColumns_Male['CustomerID'].nunique()] ,
                     'Number of Events': [clicks_logged_in_SelectedColumns_Female['SessionID'].count() , clicks_logged_in_SelectedColumns_Male['SessionID'].count()],
                    }

# convert the dictionary into dataframe and add the sum of each column to the end of dataframe
Segments_Comparison = pd.DataFrame(data=Segments_Comparison)
Segments_Comparison=Segments_Comparison.append(Segments_Comparison[['Number Of Sessions','Number Of Customers','Number of Events' ]].sum(),ignore_index=True)
Segments_Comparison.iloc[2,0] = 'Total'

# we don't want decimals to be diplayed
Segments_Comparison['Number Of Sessions']=Segments_Comparison['Number Of Sessions'].apply('{:,.0f}'.format)
Segments_Comparison['Number Of Customers']=Segments_Comparison['Number Of Customers'].apply('{:,.0f}'.format)
Segments_Comparison['Number of Events']=Segments_Comparison['Number of Events'].apply('{:,.0f}'.format)

Segments_Comparison


# ### Activities frequency for all the logged_in dataset

# In the next cell we write a code to find the most frequent activities in all logged in data:

# In[29]:


# counting the repetitions of each activity for all data
activity_counts_all_logged_in = pd.DataFrame(clicks_logged_in_SelectedColumns['PAGE_NAME'].value_counts())

# calculating the relative frequency for all data
activity_counts_all_logged_in['Relative Frequency(%)'] = round(activity_counts_all_logged_in['PAGE_NAME']/len(clicks_logged_in_SelectedColumns)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_all_logged_in.reset_index(level=0, inplace=True)
activity_counts_all_logged_in=activity_counts_all_logged_in.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_all_logged_in['Absolute Frequency']=activity_counts_all_logged_in['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_all_logged_in[activity_counts_all_logged_in['Relative Frequency(%)'] >= 1]


# In the above table, you can see the most frequent activities for all logged-in customers (those with more than one percent relative frequency), as you see only 14 out of 600 hundred webpages visited in more than one percent of the time. 

# ### Activities frequency for the segment 1

# Our paper only printed the table for the first segment means age between 18 to 29, so let's check out the frequency of this segment and see how close we are to what our article has done!

# In[30]:


# counting the repetitions of each activity for segment 1: Age 18-29
activity_counts_logged_in_18_29 = pd.DataFrame(clicks_logged_in_SelectedColumns_Age18_29['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 1: Age 18-29
activity_counts_logged_in_18_29['Relative Frequency(%)'] = round(activity_counts_logged_in_18_29['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Age18_29)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_18_29.reset_index(level=0, inplace=True)
activity_counts_logged_in_18_29=activity_counts_logged_in_18_29.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_18_29['Absolute Frequency']=activity_counts_logged_in_18_29['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_18_29[activity_counts_logged_in_18_29['Relative Frequency(%)'] >= 1]


# As you see in the above table, the absolute frequencies have a slight difference with the following table. We mentioned this before that the reason is the authors deleted some rows which we dont know why!! However the relative frequency we calculated is the same as the numbers in the articles table.
# 
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/02_Table4.png?raw=true"    />
# </center>
# 
# 
# The following chart created by **PowerBI** on the same dataset in seperate analysis:
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/03-Segment1_Frequent_Activities.png?raw=true"    />
# </center>
# 
# 
# 
# 

# ### Activities frequency for the segment 2

# In[31]:


# counting the repetitions of each activity for segment 2: Age 30-39
activity_counts_logged_in_30_39 = pd.DataFrame(clicks_logged_in_SelectedColumns_Age30_39['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 2: Age 30-39
activity_counts_logged_in_30_39['Relative Frequency(%)'] = round(activity_counts_logged_in_30_39['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Age30_39)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_30_39.reset_index(level=0, inplace=True)
activity_counts_logged_in_30_39=activity_counts_logged_in_30_39.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_30_39['Absolute Frequency']=activity_counts_logged_in_30_39['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_30_39[activity_counts_logged_in_30_39['Relative Frequency(%)'] >= 1]


# 
# The following chart created by **PowerBI** on the same dataset in seperate analysis:
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/04-Segment2_Frequent_Activities.png?raw=true"    />
# </center>
# 
# 

# ### Activities frequency for the segment 3

# In[32]:


# counting the repetitions of each activity for segment 3: Age 40-49
activity_counts_logged_in_40_49 = pd.DataFrame(clicks_logged_in_SelectedColumns_Age40_49['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 3: Age 40-49
activity_counts_logged_in_40_49['Relative Frequency(%)'] = round(activity_counts_logged_in_40_49['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Age40_49)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_40_49.reset_index(level=0, inplace=True)
activity_counts_logged_in_40_49=activity_counts_logged_in_40_49.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_40_49['Absolute Frequency']=activity_counts_logged_in_40_49['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_40_49[activity_counts_logged_in_40_49['Relative Frequency(%)'] >= 1]


# 
# The following chart created by **PowerBI** on the same dataset in seperate analysis:
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/05-Segment3_Frequent_Activities.png?raw=true"    />
# </center>
# 
# 

# ### Activities frequency for the segment 4

# In[33]:


# counting the repetitions of each activity for segment 4: Age 50-65
activity_counts_logged_in_50_65 = pd.DataFrame(clicks_logged_in_SelectedColumns_Age50_65['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 4: Age 50-65
activity_counts_logged_in_50_65['Relative Frequency(%)'] = round(activity_counts_logged_in_50_65['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Age50_65)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_50_65.reset_index(level=0, inplace=True)
activity_counts_logged_in_50_65=activity_counts_logged_in_50_65.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_50_65['Absolute Frequency']=activity_counts_logged_in_50_65['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_50_65[activity_counts_logged_in_50_65['Relative Frequency(%)'] >= 1]


# 
# The following chart created by **PowerBI** on the same dataset in seperate analysis:
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/06-Segment4_Frequent_Activities.png?raw=true"    />
# </center>
# 
# 

# ### Activities frequency for the segment 5

# In[34]:


# counting the repetitions of each activity for segment 5: Female
activity_counts_logged_in_Female = pd.DataFrame(clicks_logged_in_SelectedColumns_Female['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 5: Female
activity_counts_logged_in_Female['Relative Frequency(%)'] = round(activity_counts_logged_in_Female['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Female)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_Female.reset_index(level=0, inplace=True)
activity_counts_logged_in_Female=activity_counts_logged_in_Female.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_Female['Absolute Frequency']=activity_counts_logged_in_Female['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_Female[activity_counts_logged_in_Female['Relative Frequency(%)'] >= 1]


# ### Activities frequency for the segment 6

# In[35]:


# counting the repetitions of each activity for segment 6: Male
activity_counts_logged_in_Male = pd.DataFrame(clicks_logged_in_SelectedColumns_Male['PAGE_NAME'].value_counts())

# calculating the relative frequency for segment 6: Male
activity_counts_logged_in_Male['Relative Frequency(%)'] = round(activity_counts_logged_in_Male['PAGE_NAME']/len(clicks_logged_in_SelectedColumns_Female)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_logged_in_Male.reset_index(level=0, inplace=True)
activity_counts_logged_in_Male=activity_counts_logged_in_Male.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_logged_in_Male['Absolute Frequency']=activity_counts_logged_in_Male['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than one percent Relative Frequency
activity_counts_logged_in_Male[activity_counts_logged_in_Male['Relative Frequency(%)'] >= 1]


# <div id="Process_discovery">
#     <h2>Challenge1: Process Discovery: Distinct Usage Patterns for www.werk.nl</h2>
#     
# 
# </div>

# ## Segment 1: Age 18-29

# In[36]:


# saving the most frequent activites of segment 1 into a list
most_frequent_activites_list_segment1 = activity_counts_logged_in_18_29[activity_counts_logged_in_18_29['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment1


# In[37]:


clicks_logged_in_SelectedColumns_Age18_29.head()


# In[38]:


# copying required columns into new data frames and renaming the columns
segment_1 = clicks_logged_in_SelectedColumns_Age18_29[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_1=segment_1.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_1.head()


# In[39]:


segment_1.shape


# In[40]:


segment_1_most_frequent = segment_1.copy()

# removing records for non-frequent activities:
segment_1_most_frequent = segment_1_most_frequent[segment_1_most_frequent['activity'].isin(most_frequent_activites_list_segment1)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_1_most_frequent.loc[~segment_1_most_frequent['activity'].isin(most_frequent_activites_list_segment1), 'activity'] = 'other'

segment_1_most_frequent.head()


# In[41]:


segment_1_most_frequent.shape


# In[42]:


# creating Event Log
event_log_segment_1 = pm4py.format_dataframe(
    segment_1_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[43]:


event_log_segment_1.head(7)


# In[44]:


start_activities_segment1 = pm4py.get_start_activities(event_log_segment_1)
end_activities_segment1 = pm4py.get_end_activities(event_log_segment_1)


# In[45]:


print(f'Start activities: {start_activities_segment1}')
print(f'\nEnd activities  : {end_activities_segment1}')


# In[46]:


xes_exporter.apply(event_log_segment_1, 'event_log_segment_1.xes')


# In[47]:


log_segment_1 = xes_importer.apply('event_log_segment_1.xes')


# In[48]:


# EventLog
type(log_segment_1)


# In[49]:


# Trace
type(log_segment_1[0])


# In[50]:


# Event
type(log_segment_1[0][0])


# In[51]:


# Start activities
pm4py.get_start_activities(log_segment_1)


# In[52]:


# End activities
pm4py.get_end_activities(log_segment_1)


# In[53]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_1, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment1_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# 
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/07-ProcessMap_Segment1.png?raw=true"    />
# </center>
# 
# 

# ## Segment 2: Age 30-39

# In[54]:


# saving the most frequent activites of segment 2 into a list
most_frequent_activites_list_segment2 = activity_counts_logged_in_30_39[activity_counts_logged_in_30_39['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment2


# In[55]:


clicks_logged_in_SelectedColumns_Age30_39.head()


# In[56]:


# copying required columns into new data frames and renaming the columns
segment_2 = clicks_logged_in_SelectedColumns_Age30_39[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_2=segment_2.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_2.head()


# In[57]:


segment_2.shape


# In[58]:


segment_2_most_frequent = segment_2.copy()

# removing records for non-frequent activities:
segment_2_most_frequent = segment_2_most_frequent[segment_2_most_frequent['activity'].isin(most_frequent_activites_list_segment2)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_2_most_frequent.loc[~segment_2_most_frequent['activity'].isin(most_frequent_activites_list_segment2), 'activity'] = 'other'

segment_2_most_frequent.head()


# In[59]:


segment_2_most_frequent.shape


# In[60]:


# creating Event Log
event_log_segment_2 = pm4py.format_dataframe(
    segment_2_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[61]:


event_log_segment_2.head(7)


# In[62]:


start_activities_2 = pm4py.get_start_activities(event_log_segment_2)
end_activities_2 = pm4py.get_end_activities(event_log_segment_2)


# In[63]:


print(f'Start activities: {start_activities_2}')
print(f'\nEnd activities  : {end_activities_2}')


# In[64]:


xes_exporter.apply(event_log_segment_2, 'event_log_segment_2.xes')


# In[65]:


log_segment_2 = xes_importer.apply('event_log_segment_2.xes')


# In[66]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_2, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment2_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# 
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/08-ProcessMap_Segment2.png?raw=true"    />
# </center>
# 
# 

# ## Segment 3: Age 40-49

# In[67]:


# saving the most frequent activites of segment 3 into a list
most_frequent_activites_list_segment3 = activity_counts_logged_in_40_49[activity_counts_logged_in_40_49['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment3


# In[68]:


# copying required columns into new data frames and renaming the columns
segment_3 = clicks_logged_in_SelectedColumns_Age40_49[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_3=segment_3.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_3.head()


# In[69]:


segment_3.shape


# In[70]:


segment_3_most_frequent = segment_3.copy()

# removing records for non-frequent activities:
segment_3_most_frequent = segment_3_most_frequent[segment_3_most_frequent['activity'].isin(most_frequent_activites_list_segment3)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_3_most_frequent.loc[~segment_3_most_frequent['activity'].isin(most_frequent_activites_list_segment3), 'activity'] = 'other'

segment_3_most_frequent.head()


# In[71]:


segment_3_most_frequent.shape


# In[72]:


# creating Event Log
event_log_segment_3 = pm4py.format_dataframe(
    segment_3_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[73]:


event_log_segment_3.head(7)


# In[74]:


start_activities_3 = pm4py.get_start_activities(event_log_segment_3)
end_activities_3 = pm4py.get_end_activities(event_log_segment_3)


# In[75]:


print(f'Start activities: {start_activities_3}')
print(f'\nEnd activities  : {end_activities_3}')


# In[76]:


xes_exporter.apply(event_log_segment_3, 'event_log_segment_3.xes')


# In[77]:


log_segment_3 = xes_importer.apply('event_log_segment_3.xes')


# In[78]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_3, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment3_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# 
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/09-ProcessMap_Segment3.png?raw=true"    />
# </center>
# 
# 

# ## Segment 4: Age 50-65

# In[79]:


# saving the most frequent activites of segment 4 into a list
most_frequent_activites_list_segment4 = activity_counts_logged_in_50_65[activity_counts_logged_in_50_65['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment4


# In[80]:


clicks_logged_in_SelectedColumns_Age50_65.head()


# In[81]:


# copying required columns into new data frames and renaming the columns
segment_4 = clicks_logged_in_SelectedColumns_Age50_65[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_4=segment_4.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_4.head()


# In[82]:


segment_4.shape


# In[83]:


segment_4_most_frequent = segment_4.copy()

# removing records for non-frequent activities:
segment_4_most_frequent = segment_4_most_frequent[segment_4_most_frequent['activity'].isin(most_frequent_activites_list_segment4)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_4_most_frequent.loc[~segment_4_most_frequent['activity'].isin(most_frequent_activites_list_segment4), 'activity'] = 'other'

segment_4_most_frequent.head()


# In[84]:


segment_4_most_frequent.shape


# In[85]:


# creating Event Log
event_log_segment_4 = pm4py.format_dataframe(
    segment_4_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[86]:


event_log_segment_4.head(7)


# In[87]:


start_activities_4 = pm4py.get_start_activities(event_log_segment_4)
end_activities_4 = pm4py.get_end_activities(event_log_segment_4)


# In[88]:


print(f'Start activities: {start_activities_4}')
print(f'\nEnd activities  : {end_activities_4}')


# In[89]:


xes_exporter.apply(event_log_segment_4, 'event_log_segment_4.xes')


# In[90]:


log_segment_4 = xes_importer.apply('event_log_segment_4.xes')


# In[91]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_4, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment4_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# 
# 
# 
# 
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/10-ProcessMap_Segment4.png?raw=true"    />
# </center>
# 
# 

# ## Segment 5: Female

# In[92]:


# saving the most frequent activites of segment 5 into a list
most_frequent_activites_list_segment5 = activity_counts_logged_in_Female[activity_counts_logged_in_Female['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment5


# In[93]:


clicks_logged_in_SelectedColumns_Female.head()


# In[94]:


# copying required columns into new data frames and renaming the columns
segment_5 = clicks_logged_in_SelectedColumns_Female[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_5=segment_5.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_5.head()


# In[95]:


segment_5.shape


# In[96]:


segment_5_most_frequent = segment_5.copy()

# removing records for non-frequent activities:
segment_5_most_frequent = segment_5_most_frequent[segment_5_most_frequent['activity'].isin(most_frequent_activites_list_segment5)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_5_most_frequent.loc[~segment_5_most_frequent['activity'].isin(most_frequent_activites_list_segment5), 'activity'] = 'other'

segment_5_most_frequent.head()


# In[97]:


segment_5_most_frequent.shape


# In[98]:


# creating Event Log
event_log_segment_5 = pm4py.format_dataframe(
    segment_5_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[99]:


event_log_segment_5.head(7)


# In[100]:


start_activities_5 = pm4py.get_start_activities(event_log_segment_5)
end_activities_5 = pm4py.get_end_activities(event_log_segment_5)


# In[101]:


print(f'Start activities: {start_activities_5}')
print(f'\nEnd activities  : {end_activities_5}')


# In[102]:


xes_exporter.apply(event_log_segment_5, 'event_log_segment_5.xes')


# In[103]:


log_segment_5 = xes_importer.apply('event_log_segment_5.xes')


# In[104]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_5, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment5_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# ## Segment 6: Male

# In[105]:


# saving the most frequent activites of segment 3 into a list
most_frequent_activites_list_segment6 = activity_counts_logged_in_Male[activity_counts_logged_in_Male['Relative Frequency(%)'] >= 1]['Activity'].tolist()
most_frequent_activites_list_segment6


# In[106]:


clicks_logged_in_SelectedColumns_Male.head()


# In[107]:


# copying required columns into new data frames and renaming the columns
segment_6 = clicks_logged_in_SelectedColumns_Male[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
segment_6=segment_6.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
segment_6.head()


# In[108]:


segment_6.shape


# In[109]:


segment_6_most_frequent = segment_6.copy()

# removing records for non-frequent activities:
segment_6_most_frequent = segment_6_most_frequent[segment_6_most_frequent['activity'].isin(most_frequent_activites_list_segment6)]

## renaming  acivity name to "other" for all records with non-frequent activities:
# segment_6_most_frequent.loc[~segment_6_most_frequent['activity'].isin(most_frequent_activites_list_segment6), 'activity'] = 'other'

segment_6_most_frequent.head()


# In[110]:


segment_6_most_frequent.shape


# In[111]:


# creating Event Log
event_log_segment_6 = pm4py.format_dataframe(
    segment_6_most_frequent,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[112]:


event_log_segment_6.head(7)


# In[113]:


start_activities_6 = pm4py.get_start_activities(event_log_segment_6)
end_activities_6 = pm4py.get_end_activities(event_log_segment_6)


# In[114]:


print(f'Start activities: {start_activities_6}')
print(f'\nEnd activities  : {end_activities_6}')


# In[115]:


xes_exporter.apply(event_log_segment_6, 'event_log_segment_6.xes')


# In[116]:


log_segment_6 = xes_importer.apply('event_log_segment_6.xes')


# In[117]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_segment_6, dependency_threshold=0.9999, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='Segment6_behavior_heuristicsminer.png') 
pm4py.view_heuristics_net(heu_net)


# <div id="Customer_behaviour">
#     <h2>Challenge2: Customer Behaviour: Changes of Usage Patterns Over Time</h2>
# 
# </div>

# In the next cell we import the "clicks_logged_in_SelectedColumns" dataset. we exported it as a csv file in the begining of this notebook:

# In[118]:


clicks_logged_in_SelectedColumns = pd.read_csv('clicks_logged_in_SelectedColumns.csv')
clicks_logged_in_SelectedColumns.head()


# In[119]:


#sorting values based on CustomerID and SessionID
clicks_logged_in_SelectedColumns.sort_values(['CustomerID', 'SessionID'], ascending=[True, True])


# In the next cell we are going to count number of sessions of each customer:

# In[120]:


# grouping the data by CustomerID and counting unique SessionIDs for each CustomerID 
number_of_sessions_per_customer = pd.DataFrame(clicks_logged_in_SelectedColumns.groupby('CustomerID')['SessionID'].nunique())

# reseting the datafraem index and renaming columns
number_of_sessions_per_customer.reset_index(level=0, inplace=True)
number_of_sessions_per_customer = number_of_sessions_per_customer.rename(columns={'index': 'CustomerID','SessionID': '# of Sessions' })

number_of_sessions_per_customer.head(10)


# In[121]:


# display the above dataframe sorted by number of sessions
number_of_sessions_per_customer.sort_values(['# of Sessions'], ascending=True)


# In the next cell we loop through the above dataset and calculate number of customers who have at least 1,2,3,...,15 seessions

# In[122]:


# creating an empty list
number_of_customer_min_sessions = list()

# looping through "number_of_sessions_per_customer" dataset and count number of customers who have at least 1,2,..,15 sessions
for one in range(1,16):
    number_of_customer_min_sessions.append(
        number_of_sessions_per_customer[number_of_sessions_per_customer['# of Sessions'] >= one]['CustomerID'].count())
    
number_of_customer_min_sessions


# In[123]:


# creating list of Session number names for our table

session_list = ['1 Session']

for one in range(2,16):
    session_list.append(str(one) +' ' + "Sessions")
    
session_list


# In the next cell we will calculate the change in number of customers over the sessions

# In[124]:


change_list = ['-']

for one in range(1,15):
    change_list.append(round((number_of_customer_min_sessions[one]-number_of_customer_min_sessions[one-1])/number_of_customer_min_sessions[one-1]*100,2))
    
change_list


# Now we can create our table!

# In[125]:


# creating a dictionary which has columns names as keys and the three lists we created above as values
least_one_session = pd.DataFrame(
    {'Session': session_list,
     'Customers': number_of_customer_min_sessions,
     'Change': change_list
    })

# converting the dictionary to a dataframe
least_one_session = pd.DataFrame(data=least_one_session)
least_one_session


# The above tables  introduces the number of customer having at least 15 sessions of using the website. The analysis of results suggests that 26,647 users had at least one session which significantly decreased to 14,800 who had at least 15 sessions. From this statistics, we can infer that the users don’t tend to use the website after they used it for the first time and this trends continues over time. E.g. Table 5 suggests that 3.54 % of the customers did not use the website again after they used it for the first time. The average drop rate between sessions is  about 4 %.

# The following is the same table from our base article; as you can see, the numbers are very similar, and the pattern is the same. There is only a slight difference between numbers because, as we mentioned before, the authors deleted some records in their preprocessing, which we don't know why.
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/11_Table5.png?raw=true"  />
# </center>    
# 
# 
# 

# **Now we want to aggregate the number of logs per session sorted according to the time.**

# In the next cell we sort the "clicks_logged_in_SelectedColumns" dataset by date and time:

# In[126]:


clicks_logged_in_SelectedColumns_SortedbyTime = clicks_logged_in_SelectedColumns.sort_values('TIMESTAMP', ascending=True)


# In[127]:


clicks_logged_in_SelectedColumns_SortedbyTime


# Now we need to list the unique sessions for each customer sorted by time, so we know which sessions are the first, second, ... and fifteenth sessions per customer. 

# In[128]:


# grouping "clicks_logged_in_SelectedColumns_SortedbyTime" dataset by customerID column and counting unique session value for each
sessions_per_customer_SortedbyTime = clicks_logged_in_SelectedColumns_SortedbyTime.groupby('CustomerID').apply(lambda x: x['SessionID'].unique())
sessions_per_customer_SortedbyTime


# In[129]:


# converting the above object into a data frame
sessions_per_customer_SortedbyTime = pd.DataFrame(sessions_per_customer_SortedbyTime)

# save column of sessions per customer in a list
sessions_per_customer_SortedbyTime_values = sessions_per_customer_SortedbyTime[0].to_list()

# Checking the output for first three customers
sessions_per_customer_SortedbyTime_values[1:4]


# In[130]:


# creating an empty dictionary, contaning 15 empty lists, named "Sessions" to store SessionIDs in the relted list
Sessions = {}
for i in range(1, 16):
    Sessions[str(i)+' '+ 'Sessions'] = []
Sessions


# In the next cell, we will loop through the lists in the "sessions_per_customer_SortedbyTime" list and store the desired SessionID in the related list; for example, the SessiondID of all first sessions will be stored in the 1 Sessions list.

# In[131]:


# looping through the "sessions_per_customer_SortedbyTime_values" list
for i in range(0,15):
    for one in sessions_per_customer_SortedbyTime_values:
        if len(one) > i: 
            Sessions[f'{i + 1} Sessions'].append(one[i])


# In[132]:


## checkig the first five items of Sessions dictionary
# list(Sessions.items())[:5]


# Now that we have the SessionIDs of first, second,... and fifteenth sessions stored in Sessions dictionary we can filter  the "clicks_logged_in_SelectedColumns" for ith session's sessionIDs and count the number of clicks (records).

# In[133]:


# creating another empty dictionary, containing 15 lists, to store the number of clicks for first, second, ... and fifteenth sessions
Sessions_num_of_clicks = {}
for i in range(1, 16):
    Sessions_num_of_clicks[str(i)+' '+ 'Sessions'] = []
Sessions_num_of_clicks


# In[134]:


# counting number of clicks for 1st, 2nd, .... 15th sessions and storing it in "Sessions_num_of_clicks" dictionary
for i in range(0,15):
     Sessions_num_of_clicks[f'{i + 1} Sessions'].append(len(clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['SessionID'].isin(Sessions[f'{i + 1} Sessions'])]))

Sessions_num_of_clicks


# Now that we know how many click records we have for the 1st, 2nd, ..., and 15th sessions for all customers altogether, we can calculate the drop number of clicks over sessions from the first to the fifteenth.

# In[135]:


# there is no drop for first sessions, so we just fill first row of change column with a dash
Sessions_num_of_clicks['1 Sessions'].append('-')

# calculating the percentage of drop of click records over sessions and rounding it we two decimals and storing it in the "Sessions_num_of_clicks" dictionary lists
for i in range(1,15):
    Sessions_num_of_clicks[f'{i + 1} Sessions'].append(round(((Sessions_num_of_clicks[f'{i+1} Sessions'][0]-Sessions_num_of_clicks[f'{i} Sessions'][0])/Sessions_num_of_clicks[f'{i} Sessions'][0])*100,2))    
    
Sessions_num_of_clicks


# In the next cell we are going to calculate the average number of clicks per customer pro session:

# In[136]:


for i in range(0,15):
    Sessions_num_of_clicks[f'{i + 1} Sessions'].append(round(len(clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['SessionID'].isin(Sessions[f'{i + 1} Sessions'])])/len(Sessions[f'{i + 1} Sessions']),2))

Sessions_num_of_clicks


# Now we will convert the above dictionary into a Pandas DataFrame:

# In[137]:


# converting the "Sessions_num_of_clicks" dictionary into dataframe
Click_trend_over_time = pd.DataFrame.from_dict(data=Sessions_num_of_clicks , orient='index')

# resetting the index of dataframe and renaming it columns  
Click_trend_over_time.reset_index(level=0, inplace=True)
Click_trend_over_time=Click_trend_over_time.rename(columns={'index': '# of Sessions', 0 : '# of Click Logs', 1 : 'Change(%)', 2 : 'Average # of Clicks per customer pro session'})
Click_trend_over_time


# In the above table  we have aggregated the number of logs per session sorted according to the time. The first column of this table indicates not the amount but the order of the sessions. E.g. the number of click logs in the first session of all individual customers is equal to 603,405. These results also propose that the number of click logs per session drops over time by decreasing to 143,238 in fifteenth session of all customers. This trend can be considered as reasonable since the numbers customers per session decreases as mentioned before. However, the velocity of negative change in number of clicks is much higher. Furthermore, to normalize the results we have also calculated the average numbers of clicks by individual customers per session (See the last column). A decreasing trend in this feature can also be easily observed. On these grounds we can argue that, not only the number of customers using the website decreases over time but also the average clicks per session follows a negative trend.

# The following is the same table from our base article; as you can see, the negative trend pattern is the same, but there is a noticeable difference in the numbers, especially for the first session. The reason again might be the authors' changes in the preprocessing phase. Since all customers have at least one session of using the website, deleting some records from the dataset impacts the number of clicks in the first sessions more than other sessions. 
# 
# **The important thing about this table is the trend pattern which complies with the article table, as you see below:**
# 
# <br>
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/12_Table6.png?raw=true"  />
# </center>    
# 
# 
# 
# 

# **We want to draw a chart that shows the change of the website's most frequent webpages usage over time.**

# In[138]:


# let's check out what are the most frequent activities in the whole datase 
activity_counts_all_logged_in[activity_counts_all_logged_in['Relative Frequency(%)'] >= 1.1]


# In[139]:


# storing the name of most frequent activities in a list
most_frequent_activites_list = activity_counts_all_logged_in[activity_counts_all_logged_in['Relative Frequency(%)'] >= 1.1]['Activity'].tolist()
most_frequent_activites_list


# In the next cell we will create dictionary containing activity names and empty lists for sessions to fill them with the relative frequency of each page:

# In[140]:


# creating the dictionary
activity_relative_frequency_over_sessions = {
    'Activities': most_frequent_activites_list,
    '1 Sessions': [],
    '2 Sessions': [],
    '3 Sessions': [],
    '4 Sessions': [],
    '5 Sessions': [],
    '6 Sessions': [],
    '7 Sessions': [],
    '8 Sessions': [],
    '9 Sessions': [],
    '10 Sessions': [],
    '11 Sessions': [],
    '12 Sessions': [],
    '13 Sessions': [],
    '14 Sessions': [],
    '15 Sessions': []
    }

activity_relative_frequency_over_sessions


# Now we are going to loop through the "most_frequent_activites_list" list, and for each page, we will calculate the relative frequency per session and store the number in the above dictionary:

# In[141]:


# writing two nested loops to calculate relative frequency per session for each page and storing in the dictionary
for activity in most_frequent_activites_list:
    for i in range(0,15):
        i_sessions = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['SessionID'].isin(Sessions[f'{i+1} Sessions'])]
        activity_counts_i_sessions = pd.DataFrame(i_sessions['PAGE_NAME'].value_counts())
        activity_counts_i_sessions['Relative Frequency(%)'] = round(activity_counts_i_sessions['PAGE_NAME']/len(i_sessions)*100,2)
        activity_counts_i_sessions.reset_index(level=0, inplace=True)
        activity_counts_i_sessions=activity_counts_i_sessions.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })
        activity_relative_frequency_over_sessions[f'{i+1} Sessions'].append(activity_counts_i_sessions[activity_counts_i_sessions['Activity'] == activity]['Relative Frequency(%)'].iloc[0])


# In[142]:


# converting the dictionary to a dataframe
activity_relative_frequency_over_sessions = pd.DataFrame.from_dict(data=activity_relative_frequency_over_sessions , orient='index')

# grab the first row for the header
new_header = activity_relative_frequency_over_sessions.iloc[0] 

#take the data less the header row
activity_relative_frequency_over_sessions =activity_relative_frequency_over_sessions[1:] 

# making the pages name as the columns names
activity_relative_frequency_over_sessions.columns = new_header
activity_relative_frequency_over_sessions


# In[143]:


## exporting csv file
# activity_relative_frequency_over_sessions.to_csv ('activity_relative_frequency_over_sessions.csv', index = False)


# In[144]:


# plotting the relative frequency of frequently visited webpages over the sessions
activity_relative_frequency_over_sessions.plot.line(figsize=(20,10))
plt.show()


# To answer the question of how the usage patterns change, we have aggregated the log data of customers beginning from their first session to the fifteenth session and analyzed the website's visits. The above figure provides valuable insights into changes in website usage behavior over time. We have introduced the visited web pages with a relative frequency higher than 1.1%. From the underlying diagram, we can detect a significant drop in the visit of the "mijn_cv" page. The relative frequency of **"mijn_cv"** decreased to less than 10% in the fifteenth, more than 25% in users' first session. A significant negative trend is also observed in the visit frequency of **"aanvragen-ww"** and **"inschrijven"**.
# 
# In contrast to **mijn_cv**, **"taken"** page follows an increasing preference trend over time. The relative frequency of taken has increased from 12.32% in the first sessions to 30.14% in the fifteenth sessions. In other web pages such as **"werkmap"**, **"mijn_sollicitaties"**, **"vacatures_zoeken"**, **"mijn_berichten"** and etc. we can observe an increasing trend however, the amplitude of the change is not significantly high.

# The following figure is the exact figure in the article; as you see, the results, patterns, and trends are pretty similar; however, there are slight differences in some of the numbers because of the difference between the authors and us in the data cleaning phase.
# 
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/13_fig5.png?raw=true"  />
# </center>    
# 

# **In order to identify the changes in the transition between websites we need to create the corresponding process maps. we will create the process map for the 1st and the 15th sessions.**

# ### Creating Process Model representing first sessions of customers

# In[145]:


# susbeting the first sessions records from "clicks_logged_in_SelectedColumns" dataset
first_sessions_of_customers = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['SessionID'].isin(Sessions['1 Sessions'])]


# In[146]:


# counting the repetitions of each activity for the 1st Sessions
activity_counts_first_sessions_of_customers = pd.DataFrame(first_sessions_of_customers['PAGE_NAME'].value_counts())

# calculating the relative frequency for the 1st Sessions
activity_counts_first_sessions_of_customers['Relative Frequency(%)'] = round(activity_counts_first_sessions_of_customers['PAGE_NAME']/len(first_sessions_of_customers)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_first_sessions_of_customers.reset_index(level=0, inplace=True)
activity_counts_first_sessions_of_customers=activity_counts_first_sessions_of_customers.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_first_sessions_of_customers['Absolute Frequency']=activity_counts_first_sessions_of_customers['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than 0.8 percent Relative Frequency
activity_counts_first_sessions_of_customers[activity_counts_first_sessions_of_customers['Relative Frequency(%)'] >= 0.8]


# In[147]:


# Storing the most frequent activities' names of first sessions into a list
most_frequent_activites_first_sessions = activity_counts_first_sessions_of_customers[activity_counts_first_sessions_of_customers['Relative Frequency(%)'] >= 0.8]['Activity'].tolist()
most_frequent_activites_first_sessions


# In[148]:


# copying required columns into new data frames and renaming the columns
first_sessions_of_customers_log = first_sessions_of_customers[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
first_sessions_of_customers_log=first_sessions_of_customers_log.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
first_sessions_of_customers_log.head()


# In[149]:


# keeping only the records of most frequent actvities
first_sessions_of_customers_log = first_sessions_of_customers_log[first_sessions_of_customers_log['activity'].isin(most_frequent_activites_first_sessions)]


# In[150]:


# creating Event Log
event_log_first_sessions = pm4py.format_dataframe(
    first_sessions_of_customers_log,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[151]:


xes_exporter.apply(event_log_first_sessions, 'event_log_first_sessions.xes')


# In[152]:


log_first_sessions = xes_importer.apply('event_log_first_sessions.xes')


# In[153]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_first_sessions, dependency_threshold=0.997, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='log_first_sessions-heuristics_net.png') 
pm4py.view_heuristics_net(heu_net)


# As yous see, the above process map is similar to the following process map, which the authors of the article created using **DISCO** software.  
#     
# <br>    
#     
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/14_fig6.png?raw=true"  />
# </center>    
# 
# 
# 
# 

# ### Creating Process Model representing fifteenth sessions of customers

# In[154]:


# susbeting the fifteenth sessions records from "clicks_logged_in_SelectedColumns" dataset
fifteenth_sessions_of_customers = clicks_logged_in_SelectedColumns[clicks_logged_in_SelectedColumns['SessionID'].isin(Sessions['15 Sessions'])]


# In[155]:


# counting the repetitions of each activity for the 15th Sessions
activity_counts_fifteenth_sessions_of_customers = pd.DataFrame(fifteenth_sessions_of_customers['PAGE_NAME'].value_counts())

# calculating the relative frequency for the 15th Sessions
activity_counts_fifteenth_sessions_of_customers['Relative Frequency(%)'] = round(activity_counts_fifteenth_sessions_of_customers['PAGE_NAME']/len(fifteenth_sessions_of_customers)*100,2)

# resting the index of dataframe and renaming the columns
activity_counts_fifteenth_sessions_of_customers.reset_index(level=0, inplace=True)
activity_counts_fifteenth_sessions_of_customers=activity_counts_fifteenth_sessions_of_customers.rename(columns={'PAGE_NAME': 'Absolute Frequency','index': 'Activity' })

# we don't want decimals to be diplayed
activity_counts_fifteenth_sessions_of_customers['Absolute Frequency']=activity_counts_fifteenth_sessions_of_customers['Absolute Frequency'].apply('{:,.0f}'.format)

# printing the data for activities with more than 0.8 percent Relative Frequency
activity_counts_fifteenth_sessions_of_customers[activity_counts_fifteenth_sessions_of_customers['Relative Frequency(%)'] >= 1.2]


# In[156]:


# Storing the most frequent activities' names of first sessions into a list
most_frequent_activites_fifteenth_sessions = activity_counts_fifteenth_sessions_of_customers[activity_counts_fifteenth_sessions_of_customers['Relative Frequency(%)'] >= 1.2]['Activity'].tolist()
most_frequent_activites_fifteenth_sessions


# In[157]:


# copying required columns into new data frames and renaming the columns
fifteenth_sessions_of_customers_log = fifteenth_sessions_of_customers[['SessionID', 'PAGE_NAME', 'TIMESTAMP']].copy()
fifteenth_sessions_of_customers_log=fifteenth_sessions_of_customers_log.rename(columns={'PAGE_NAME': 'activity','SessionID': 'case_id','TIMESTAMP': 'timestamp' })
fifteenth_sessions_of_customers_log.head()


# In[158]:


# keeping only the records of most frequent actvities
fifteenth_sessions_of_customers_log = fifteenth_sessions_of_customers_log[fifteenth_sessions_of_customers_log['activity'].isin(most_frequent_activites_fifteenth_sessions)]


# In[159]:


# creating Event Log
event_log_fifteenth_sessions = pm4py.format_dataframe(
    fifteenth_sessions_of_customers_log,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[160]:


xes_exporter.apply(event_log_fifteenth_sessions, 'event_log_fifteenth_sessions.xes')


# In[161]:


log_fifteenth_sessions = xes_importer.apply('event_log_fifteenth_sessions.xes')


# In[162]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_fifteenth_sessions, dependency_threshold=0.997, 
    and_threshold=0.999, 
    loop_two_threshold=0.999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='log_fifteenth_sessions-heuristics_net.png') 
pm4py.view_heuristics_net(heu_net)


# As yous see, the above process map is similar to the following process map, which the authors of the article created using **DISCO** software.  The number within the process activities is the absolute frequency of visited websites.  A significant drop in the number of clicks can also be directly observed here compared to the process map of the first sessions we created before.  These process diagrams allow us to observe the process paths evolved.  A narrow analysis of process models reveals that the sequence of transitions did not change significantly with a few exceptions.  E.g., aanvragen-ww was one of the most visited websites in the earlier sessions, which almost disappeared towards the last sessions (the number of clicks dropped dramatically).
# 
# **So our outcome is quite similar to the result of the article.**
#     
# <br>    
#     
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/15_fig7.png?raw=true"  />
# </center>    
# 

# <div id="Transition_expensive_channels">
#     <h2>Challenge 4: Transition to More Expensive Channels</h2>
# </div>
# 
# 

# Within this section, we give answers to some questions: Are customers more likely to use these channels again after they have used them for the first time? How is the process map when we consider the communication channels? After visiting which web pages customers will use the communication channels? 

# To find the answer the question we need to read the four following datasets analyze them separately and then merge them:
# 
# * [BPI Challenge 2016: Clicks Logged In](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Clicks_Logged_In/12674816/1)
# 
# * [BPI Challenge 2016: Questions](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Questions/12687320/1)
# 
# * [BPI Challenge 2016: Werkmap Messages](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Werkmap_Messages/12714569/1)
# 
# * [BPI Challenge 2016: Complaints](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Complaints/12717647/1)

# In[163]:


# reading the csv files of the datasets
clicks_logged_in = pd.read_csv('BPI2016_Clicks_Logged_In.csv', sep = ';', encoding = 'latin')
phone_calls = pd.read_csv('BPI2016_Questions.csv', sep = ';', encoding = 'latin')
workflow_messages = pd.read_csv('BPI2016_Werkmap_Messages.csv', sep = ';', encoding = 'latin')
complaints = pd.read_csv('BPI2016_Complaints.csv', sep = ';', encoding = 'latin')


# ###  Are customers more likely to use these channels again after they have used them for the first time?

# In[164]:


# Creating an empty dictionary to store the total count and unique count of customers using different contact channel
Communication_Channels_Comparison = {'Communication Channel': ['Complaints', 'Werkmap','Questions(phone Call)'],
                     'Total Use': [len(complaints), len(workflow_messages), len(phone_calls)],
                     'Unique Customer Use': [complaints['CustomerID'].nunique(), workflow_messages['CustomerID'].nunique(), phone_calls['CustomerID'].nunique()],
                    }
                                                                              
# convert the dictionary into dataframe and add the sum of each column to the end of dataframe
Communication_Channels_Comparison = pd.DataFrame(data=Communication_Channels_Comparison)

# sorting 
Communication_Channels_Comparison = Communication_Channels_Comparison.sort_values('Total Use', ascending=False)
Communication_Channels_Comparison.reset_index(drop=True, inplace=True)


Communication_Channels_Comparison = Communication_Channels_Comparison.append(Communication_Channels_Comparison[['Total Use']].sum(),ignore_index=True)
Communication_Channels_Comparison.iloc[3,0] = 'Total Comunications(sum of the three above)'


Communication_Channels_Comparison['Relative Use (%)']= round(Communication_Channels_Comparison['Total Use']/Communication_Channels_Comparison.iloc[3,1]*100,2)

Communication_Channels_Comparison ['Averages (Total/Unique)'] = round(Communication_Channels_Comparison ['Total Use']/Communication_Channels_Comparison ['Unique Customer Use'],2)




# we don't want decimals to be diplayed

Communication_Channels_Comparison['Total Use']=Communication_Channels_Comparison['Total Use'].apply('{:,.0f}'.format)
Communication_Channels_Comparison['Unique Customer Use']=Communication_Channels_Comparison['Unique Customer Use'].apply('{:,.0f}'.format)

Communication_Channels_Comparison.iloc[3,4]= '-'
Communication_Channels_Comparison.iloc[3,3]= '-'
Communication_Channels_Comparison.iloc[3,2]= '-'

Communication_Channels_Comparison


# In order to answer this question we have to identify the relevant exploratory statistics about the customers, who used the expensive channels such as sending werkmap message, complaints or contacting the call center. As depicted in the above tabled the total number of contacts via complaints, werkmap and questions are 289, 66,058 and 123,403 respectively. These numbers suggest that calling the customer center is the mostly preferred communication channel by almost doubling the number of werkmap messages.
# 
# Filling complaints is rarely used by customers as means of communication by corresponding to 0.15% of total contacts.
# A further analysis reveals that some customers have used the different channels multiple times as only 226, 16,653 and 21,533 unique customers sent complaints, werkmap messages and questions respectively.
# 
# Considering the averages, we can argue that the customers tend to contact call centers to ask questions (5.73 times) and send werkmap messages (3.97 times) relatively more after they have used them for the first time. This number is 1.28 for sending complaints which implies that with a few exceptions, the customers don’t tend to fill the complaints after their first time.
# 
# We created the following charts with Microsoft **PowerBI** and as you see, it totally matches the results of the above table, which we developed with python codes separately:
# 
# 
# <br>    
#     
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/16-BI-ContactChannels.png?raw=true"  />
# </center>   
# 
# <br>
# 
# And here is the same charts in the article, as you can see it  is identical to our results:
# 
# <br>    
#     
# <center>
#     <img src="https://github.com/sinaaghaee/ProcessMiningProject-CustomerBehaviorAnalysis/blob/main/Images/17-fig9.png?raw=true"  />
# </center>   
# 
# 

# Now we are going to merge four following datasets and explore thme together:
# 
# * [BPI Challenge 2016: Clicks Logged In](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Clicks_Logged_In/12674816/1)
# 
# * [BPI Challenge 2016: Questions](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Questions/12687320/1)
# 
# * [BPI Challenge 2016: Werkmap Messages](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Werkmap_Messages/12714569/1)
# 
# * [BPI Challenge 2016: Complaints](https://data.4tu.nl/articles/dataset/BPI_Challenge_2016_Complaints/12717647/1)

# In[165]:


clicks_logged_in.head(2)


# In[166]:


phone_calls.head(2)


# In[167]:


workflow_messages.head(2)


# In[168]:


complaints.head(2)


# ### Simplifying and Merging the Four Datasets
# 
# To merge the four datasets based on CustomerId first, we need to simplify our data in a way that we have similar columns for each one:

# ### Complaints Dataset

# Since the complaints dataset doesn't comprise time and only have dates, to make this data set consistent with the other three datasets, we will generate random times, and we will add it to our data:

# In[169]:


#defining a function to generate random datetime

def random_datetimes_or_dates(start, end, out_format='datetime', n=10): 

    (divide_by, unit) = (10**9, 's') if out_format=='datetime' else (24*60*60*10**9, 'D')

    start_u = start.value//divide_by
    end_u = end.value//divide_by

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit) 


# In[170]:


# genertating 289 (number of complaints dataset records) random datetimes
start = pd.to_datetime('2000-01-01')
end = pd.to_datetime('2005-01-01')
random_datetime =random_datetimes_or_dates(start, end, out_format='datetime', n=289)
random_datetime = pd.DataFrame(random_datetime)
random_datetime.head(2)


# In[171]:


# since complaints dataset has dates itself, we only need the random times so we divide random datetimes to date and time columns

random_datetime['Time'] = pd.to_datetime(random_datetime[0]).dt.time
random_datetime.head(2)


# In[172]:


# Here we combine actual dates from complaints dataset with random times we genereted to create a complete datetime column 

complaints_datetime = pd.DataFrame(data={'date': complaints.ContactDate, 'time' : random_datetime['Time']})
complaints_datetime['date'] = pd.to_datetime(complaints_datetime['date']).dt.date
complaints_datetime['datetime'] = complaints_datetime.apply(lambda r : pd.datetime.combine(r['date'],r['time']),1)
complaints_datetime.head()


# In[173]:


## Simplify complaints

complaints_simp = pd.DataFrame(data={'customerID': complaints.CustomerID, "activity" : "complaint", "date" : complaints_datetime['datetime']})
complaints_simp.head()


# ### Phone Calls Dataset

# In[174]:


## Simplify phone_calls

phone_calls_simp = pd.DataFrame(data={'customerID': phone_calls.CustomerID, "activity" : "phone_call", "date" : phone_calls.ContactDate +' '+ phone_calls.ContactTimeStart})
phone_calls_simp['date'] = pd.to_datetime(phone_calls_simp['date'])
phone_calls_simp.head()


# ### Workflow Dataset

# In[175]:


## Simplify workflow_messages
workflow_messages_simp = pd.DataFrame(data={'customerID': workflow_messages.CustomerID, "activity" : "workflow_message", "date" : workflow_messages.EventDateTime})
workflow_messages_simp['date'] = pd.to_datetime(workflow_messages_simp['date'])
workflow_messages_simp.head()


# ### clicks_logged_in

# In[176]:


most_frequent_activites_list


# In[177]:


## Simplify clicks_logged_in
clicks_logged_in_simp = clicks_logged_in[clicks_logged_in['PAGE_NAME'].isin(most_frequent_activites_list)]
clicks_logged_in_simp = pd.DataFrame(data={'customerID': clicks_logged_in_simp.CustomerID, "activity" : clicks_logged_in_simp.PAGE_NAME, "date" : clicks_logged_in_simp.TIMESTAMP})
clicks_logged_in_simp['date'] = pd.to_datetime(clicks_logged_in_simp['date'])
clicks_logged_in_simp


# In[178]:


## Merging the four simplified datasets

simple_customer_journey = pd.concat([workflow_messages_simp,phone_calls_simp ,complaints_simp,clicks_logged_in_simp])
simple_customer_journey.sort_values(by='date', ascending=True)


# In[179]:


log_csv = simple_customer_journey.copy()


# In[180]:


log_csv.dtypes


# In[181]:


# Shape
log_csv.shape


# In[182]:


# Check NA-values
log_csv.isna().sum()


# In[183]:


# Info
log_csv.info()


# In[184]:


# Sample
log_csv.sample(frac=.25)


# In[185]:


log_csv = log_csv.rename(columns={'customerID': 'case_id','date': 'timestamp' })


# In[186]:


# Unique values
pd.DataFrame(
    {
        'variable': log_csv.columns, 
        'unique values': [log_csv[col].nunique() for col in log_csv.columns],
        'fraction': [round(log_csv[col].nunique() / log_csv.shape[0], 2) for col in log_csv.columns], 
    }
).set_index('variable')


# In[187]:


activity_counts=pd.DataFrame(log_csv['activity'].value_counts())
activity_counts


# In[188]:


# Unique values: timestamp
log_csv.timestamp.nunique()


# In[189]:


# Multiple occurences: timestamp
log_csv.timestamp.value_counts()[log_csv.timestamp.value_counts() > 1]


# In[190]:


def activity_duration(
    event_log: pd.DataFrame, 
    case_var: str, 
    event_var: str, 
    timestamp_var: str, 
    duration: str ='h'
) -> pd.DataFrame:
    
    """
    Returns a dataframe with activity durations (i.e. 'arc' performance decorators).
    By default duration is set to hours (h).
    Use 's' for seconds, 'D' for days and 'W' for weeks.
    """
    
    data = dict()
    groups = log_csv.groupby(case_var)
    for group in groups:
        arc = group[1].sort_values(timestamp_var)        .rename(columns = {event_var:'event_from', timestamp_var:'time_from'})
        arc['event_to'] = arc['event_from'].shift(-1)
        arc['time_to'] = arc['time_from'].shift(-1)
        arc.dropna(inplace = True)
        duration_var = f'duration ({duration})'
        arc[duration_var] = (arc['time_to'] - arc['time_from']) / np.timedelta64(1, duration)
        data[group[0]] = arc[[case_var, 'event_from', 'event_to', 'time_from', 'time_to', duration_var]]
    return pd.concat(data.values()).set_index(case_var)


# In[191]:


log_csv_Durations = activity_duration(log_csv, 'case_id', 'activity', 'timestamp') 
log_csv_Durations


# In[192]:


def dfg_frequency_matrix(
    event_log: pd.DataFrame, 
    case_var: str, 
    event_var: str, 
    timestamp_var: str
) -> pd.DataFrame:
    
    """
    Return a directly-follows graph frequency matrix based on the traces in the event log.
    Row events (i.e. the index) are events 'from' and column events are the events 'to'.
    """
    
    # event log
    log = event_log[[case_var, event_var, timestamp_var]]
    
    # initiate matrix
    events = log[event_var].unique()
    matrix = pd.DataFrame(columns=events, index=events).fillna(0)
    
    # groupby case_var
    groups = log.groupby(case_var)
    
    # loop through case groups
    for group in groups:
        event = group[1].sort_values(timestamp_var)        .drop([case_var, timestamp_var], axis = 1)        .rename(columns = {event_var:'event_from'})
        event['event_to'] = event['event_from'].shift(-1)
        event.dropna(inplace = True)
        
        # loop through traces
        for trace in event.itertuples(index = False):
            matrix.at[trace.event_from, trace.event_to] += 1
                
    return matrix.replace(0, np.nan)


# In[193]:


# Directly-Follows Graph frequency heatmap
dfg_freq_matrix = dfg_frequency_matrix(log_csv, 'case_id', 'activity', 'timestamp')
sns.set(rc={'figure.figsize':(40, 20)})
sns.heatmap(dfg_freq_matrix, annot=True, fmt='.0f', cmap='Reds', square=True)
plt.show()


# In[194]:


def dfg_performance_matrix(
    event_log: pd.DataFrame, 
    case_var: str, 
    event_var: str, 
    timestamp_var: str, 
    duration: str ='h'
) -> pd.DataFrame:
    
    """
    Return a directly-follows graph duration matrix based on the traces in the event log.
    Row events (i.e. the index) are events 'from' and column events are the events 'to'.
    By default duration is set to hours (h). Use 's' for seconds, 'D' for days and 'W' for weeks.
    """
    
    # event log
    log = event_log[[case_var, event_var, timestamp_var]]
    
    # initiate matrix
    events = log[event_var].unique()
    matrix = pd.DataFrame(columns=events, index=events)
    
    # groupby case_var
    groups = log.groupby(case_var)
    
    # loop through case groups
    for group in groups:
        event = group[1].sort_values(timestamp_var)        .rename(columns = {event_var:'event_from', timestamp_var:'time_begin'})
        event['event_to'] = event['event_from'].shift(-1)
        event['time_end'] = event['time_begin'].shift(-1)
        event['duration'] = (event['time_end'] - event['time_begin']) / np.timedelta64(1, duration)
        event.dropna(inplace = True)
        
        # loop through traces
        for row in event.itertuples(index = False):
            matrix.at[row.event_from, row.event_to] =             np.nansum([matrix.at[row.event_from, row.event_to], row.duration])
    
    return matrix.astype(float)


# In[195]:


# Directly-Follows Graph total duration heapmap (in hours)
dfg_perf_matrix = dfg_performance_matrix(log_csv, 'case_id', 'activity', 'timestamp', duration='h')
sns.set(rc={'figure.figsize':(40, 20)})
sns.heatmap(dfg_perf_matrix, annot=True, fmt='.0f', cmap='BuPu', square=True)
plt.show()


# In[196]:


def dfg_frequency_table(dfg_frequency_matrix: pd.DataFrame) -> pd.DataFrame:
    
    """
    Returns an directly-follows graph frequency table
    """
    
    arcs = list()
    
    for row in dfg_frequency_matrix.index:
        for col in dfg_frequency_matrix.columns:
            if not np.isnan(dfg_frequency_matrix.at[row, col]):
                arcs.append((row, col, dfg_frequency_matrix.at[row, col]))
    
    arc_freq = pd.DataFrame(arcs, columns = ['event_from', 'event_to', 'frequency'])    .set_index(['event_from', 'event_to'])
    
    return arc_freq


# In[197]:


# Directly-Follows Graph frequency and performance (in hours) table
df = pd.merge(
    dfg_frequency_table(
        dfg_frequency_matrix(log_csv, 'case_id', 'activity', 'timestamp'))\
    .reset_index(),
    activity_duration(log_csv, 'case_id', 'activity', 'timestamp')\
    .loc[:, ['event_from', 'event_to', 'duration (h)']]\
    .groupby(by = ['event_from', 'event_to']).mean('duration (h)')\
    .reset_index(),

    left_on = ['event_from', 'event_to'],
    right_on = ['event_from', 'event_to']
).rename(columns = {'duration (h)':'average_duration'})

df['average_duration'] = df['average_duration'].apply(lambda x: np.round(x, decimals=2))
df['frequency'] = df['frequency'].astype(int)
df['total_duration'] = df['frequency'] * df['average_duration']

df.sort_values(['total_duration', 'average_duration', 'frequency'], ascending=False)


# In[198]:


log_csv


# In[199]:


# creating Event Log
event_log_merged = pm4py.format_dataframe(
    log_csv,
    case_id = 'case_id',
    activity_key = 'activity',
    timestamp_key = 'timestamp', 
    timest_format = '%Y-%m-%d %H:%M:%S%z'
)


# In[200]:


xes_exporter.apply(event_log_merged, 'event_log_merged.xes')


# In[201]:


log_merged = xes_importer.apply('event_log_merged.xes')


# In[202]:


# Simplified Interface
heu_net = pm4py.discover_heuristics_net(
    log_merged, dependency_threshold=0.999, 
    and_threshold=0.9999, 
    loop_two_threshold=0.9999
)
pm4py.save_vis_heuristics_net(heu_net, file_path='log_merged-heuristics_net.png') 
pm4py.view_heuristics_net(heu_net)

