import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Cleaning Energy data for Office Jolie PrimClass_Jaylin
data = pd.read_csv('E:\Thesis\Dataset\Working Directory\E2015.csv')
data = data[['Year','timestamp','PrimClass_Javier']]
# Renaming columns
data.columns = ['Year','timestamp','Energy consumption']

# Cleaning Weather data.
df = pd.read_csv('E:\Thesis\Dataset\Working Directory\w1.csv')
df.head()
df.describe()
data1 = df.iloc[0:16016,0:18]
data1[['Hour','Minutes']] = data1.Time.str.split(":", expand= True)
data1[['Date','Time']] = data1.Timestamp.str.split(" ", expand= True)
data1 = data1[data1.Minutes!= '20']
data1 = data1.reset_index()
data1.drop(columns={'index','DateUTC<br />','Timestamp.1','Minutes'}, axis=1, inplace = True)
data = pd.concat([data,data1], sort = True, axis=1)


#Counting number of NA in dataframe
data.isnull().sum()
data.notnull().sum()

# Removing the NA values from dataframe
data.drop(columns={'Events','TimeBST','TimeGMT','Precipitationmm','Gust SpeedKm/h','Time'}, axis=1, inplace=True)
data.rename(columns = {'Dew PointC':'Dew_PointC','Sea Level PressurehPa':'Sea_Level_PressurehPa','Wind Direction':'Wind_Direction','Wind SpeedKm/h':'Wind_Speed_in_KMperHour','Energy consumption':'Energy_consumption'}, inplace=True)
data.isnull().sum()


# Data Cleaning 
#Coding Wind Speed into numeric values.
data['Wind_Speed_in_KMperHour'].replace('Calm','0',inplace=True)
data.Wind_Speed_in_KMperHour = pd.to_numeric(data.Wind_Speed_in_KMperHour,errors='coerce')
data['Hour']=data.Hour.astype('float64')
data.dtypes
     

# Encoding for catagorical data

data.Conditions.unique()
data['Conditions'].replace('Drizzle','Rain',inplace=True)
data['Conditions'].replace('Fog','Rain',inplace=True)
data['Conditions'].replace('Light Rain','Rain',inplace=True)
data['Conditions'].replace('Heavy Rain','Rain',inplace=True)
data['Conditions'].replace('Drizzle','Rain',inplace=True)
data['Conditions'].replace('Light Drizzle','Rain',inplace=True)
data['Conditions'].replace('Fog','Rain',inplace=True)
data['Conditions'].replace('Rain Showers','Rain',inplace=True)
data['Conditions'].replace('Light Snow','Rain',inplace=True)
data['Conditions'].replace('Thunderstorms and Rain','Rain',inplace=True)
data['Conditions'].replace('Heavy Thunderstorms with Small Hail','Rain',inplace=True)
data['Conditions'].replace('Haze','Rain',inplace=True)
data['Conditions'].replace('Mist','Rain',inplace=True)
data['Conditions'].replace('Heavy Drizzle','Rain',inplace=True)
data['Conditions'].replace('Heavy Rain Showers','Rain',inplace=True)
data['Conditions'].replace('Light Rain Showers','Rain',inplace=True)
data['Conditions'].replace('Shallow Fog','Rain',inplace=True)
data['Conditions'].replace('Thunderstorm','Rain',inplace=True)
data['Conditions'].replace('Patches of Fog','Rain',inplace=True)

data['Wind_Direction'] = data['Wind_Direction'].replace(['SW', 'SSW','WSW'], 'SW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['SSE', 'ESE'], 'SE')
data['Wind_Direction'] = data['Wind_Direction'].replace(['WSW'], 'SW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['Variable'], 'Calm')
data['Wind_Direction'] = data['Wind_Direction'].replace(['NNW','WNW'], 'NW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['ENE','NNE'], 'Calm')
data.Conditions.unique()
data.Wind_Direction.unique()

# Removing dummy variables 
data = data[data.VisibilityKm != -9999]
data = data.reset_index(drop = True)
data = data.drop(columns={'Year','timestamp','Timestamp','Date'})


# Plotting the classes of dependent variable
plt.figure(figsize=(8, 6))
sns.countplot('Conditions', data=data)
plt.title('Balanced Classes')
plt.show()


# Plotting the classes of dependent variable
plt.figure(figsize=(8, 6))
sns.countplot('Wind_Direction', data=data)
plt.title('Balanced Classes')
plt.show()

plotdata = data

# Label encoding of dependent variables

from sklearn.preprocessing import LabelEncoder
bin_cols = data.nunique()[data.nunique() <= 9].keys().tolist()
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])


bin_cols = data.nunique()[data.nunique() <= 9].keys().tolist()
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])
data.Wind_Direction.unique()


# Exploratory Data Analysis.

# Co-Relation Plot 1        
        
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), 
        y=y.map(y_to_num), 
        s=size * size_scale, 
        marker='s' 
    )
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
columns = ['Energy_consumption', 'Dew_PointC','TemperatureC', 'Humidity', 'Sea_Level_PressurehPa', 'VisibilityKm','Wind_Speed_in_KMperHour','WindDirDegrees','Hour'] 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
) 

# Co-Relation Plot 2

f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)



# Box plots
# Run for each variable separately

sns.boxplot(data=data.VisibilityKm)
sns.boxplot(data=data.TemperatureC)
sns.boxplot(data=data.Dew_PointC)
sns.boxplot(data=data.Humidity)
sns.boxplot(data=data.Sea_Level_PressurehPa)
sns.boxplot(data=data.WindDirDegrees)
sns.boxplot(data=data.Energy_consumption)
sns.boxplot(data=data.Wind_Speed_in_KMperHour)



# Variable Distribution plots
# Run for each variable separately

plot = plotdata.iloc[:,1]
sns.distplot(plot);
plot = plotdata.iloc[:,2]
sns.distplot(plot, hist=False, rug=True);
plot = plotdata.iloc[:,3]
sns.kdeplot(plot, shade=True);
plot = plotdata.iloc[:,4]
sns.distplot(plot, kde=False, fit=stats.gamma);
plot = plotdata.iloc[:,5]
sns.distplot(plot, kde=False, fit=stats.gamma);
plot = plotdata.iloc[:,6]
sns.kdeplot(plot, shade=True);
plot = plotdata.iloc[:,7]
sns.kdeplot(plot, shade=True);
plot = plotdata.iloc[:,8]
sns.kdeplot(plot, shade=True);
plot = plotdata.iloc[:,9]
sns.kdeplot(plot, shade=True);
plot = plotdata.iloc[:,10]
sns.kdeplot(plot, shade=True);






