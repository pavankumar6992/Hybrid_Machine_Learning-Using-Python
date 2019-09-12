import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Cleaning Energy data for Office Jaylin with weather 2
data = pd.read_csv('E:\\Thesis\dataset\Working Directory\E2015.csv')
data = data[['Year','timestamp','Office_Jett']]
# Renaming columns
data.columns = ['Year','timestamp','Energy consumption']

# Cleaning Weather data.
df = pd.read_csv('E:\Thesis\dataset\Working Directory\w2.csv')
data = pd.concat([data,df], sort = True, axis=1)
#Counting number of NA in dataframe
data.isnull().sum()
data.notnull().sum()

# Removing the NA values from dataframe
data.drop(columns={'Events','Timestamp','TimeEDT','TimeEST','Precipitationmm','Gust SpeedKm/h','Time','Year','timestamp','Date'}, axis=1, inplace=True)
data.rename(columns = {'Dew PointC':'Dew_PointC','Sea Level PressurehPa':'Sea_Level_PressurehPa','Wind Direction':'Wind_Direction','Wind SpeedKm/h':'Wind_Speed_in_KMperHour','Energy consumption':'Energy_consumption'}, inplace=True)
data.isnull().sum()

data = data[data.Dew_PointC != -9999]
data = data[data.Humidity != -9999]
data = data[data.Sea_Level_PressurehPa != -9999]
data = data[data.TemperatureC != -9999]
data = data[data.VisibilityKm != -9999]
data = data[data.Wind_Speed_in_KMperHour != -9999]
data = data.reset_index(drop = True)


# data Cleaning 
#Coding Wind Speed into numeric values.
data['Wind_Speed_in_KMperHour'].replace('Calm','0',inplace=True)
data.Wind_Speed_in_KMperHour = pd.to_numeric(data.Wind_Speed_in_KMperHour,errors='coerce')
data['Hour']=data.Hour.astype('float64')
data.dtypes
     

#lable encoding for catagorical data

data.Conditions.unique()
data['Conditions'].replace('Light Freezing Rain','Rain',inplace=True)
data['Conditions'].replace('Light Rain','Rain',inplace=True)
data['Conditions'].replace('Small Hail Showers','Rain',inplace=True)
data['Conditions'].replace('Small Hail','Rain',inplace=True)
data['Conditions'].replace('Drizzle','Rain',inplace=True)
data['Conditions'].replace('Light Rain Showers','Rain',inplace=True)
data['Conditions'].replace('Light Thunderstorms and Rain','Rain',inplace=True)
data['Conditions'].replace('Thunderstorm','Rain',inplace=True)
data['Conditions'].replace('Thunderstorms and Rain','Rain',inplace=True)
data['Conditions'].replace('Heavy Thunderstorms and Rain','Rain',inplace=True)
data['Conditions'].replace('Heavy Rain','Rain',inplace=True)

data['Conditions'].replace('Mostly Cloudy','Cloudy',inplace=True)
data['Conditions'].replace('Scattered Clouds','Cloudy',inplace=True)
data['Conditions'].replace('Partly Cloudy','Cloudy',inplace=True)

data['Conditions'].replace('Light Snow','Snow',inplace=True)
data['Conditions'].replace('Heavy Snow','Snow',inplace=True)
data['Conditions'].replace('Blowing Snow','Snow',inplace=True)
data['Conditions'].replace('Ice Crystals','Snow',inplace=True)
data['Conditions'].replace('Light Snow Showers','Snow',inplace=True)

data['Conditions'].replace('Light Freezing Fog','Fog',inplace=True)
data['Conditions'].replace('Mist','Fog',inplace=True)
data['Conditions'].replace('Haze','Fog',inplace=True)
data['Conditions'].replace('Patches of Fog','Fog',inplace=True)

data['Wind_Direction'] = data['Wind_Direction'].replace(['SW', 'SSW','WSW'], 'SW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['SSE', 'ESE'], 'SE')
data['Wind_Direction'] = data['Wind_Direction'].replace(['WSW'], 'SW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['Variable'], 'Calm')
data['Wind_Direction'] = data['Wind_Direction'].replace(['NNW','WNW'], 'NW')
data['Wind_Direction'] = data['Wind_Direction'].replace(['ENE','NNE'], 'Calm')
data.Conditions.unique()
data.Wind_Direction.unique()


# label encoding dependent categorical variables

bin_cols = data.nunique()[data.nunique() <= 9].keys().tolist()
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])



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




#data.Wind_Direction.unique()


# Exploratory data Analysis.

# Co-Relation Plot 1
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
        
        
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

# Co-Relation plot 2

f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# Run for each variable separately
# Box Plots

sns.boxplot(data=data.VisibilityKm)
sns.boxplot(data=data.TemperatureC)
sns.boxplot(data=data.Dew_PointC)
sns.boxplot(data=data.Humidity)
sns.boxplot(data=data.Sea_Level_PressurehPa)
sns.boxplot(data=data.WindDirDegrees)
sns.boxplot(data=data.Energy_consumption)
sns.boxplot(data=data.Wind_Speed_in_KMperHour)


plotdata = data

# Run for each variable separately
# Variable distrbution plot


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

