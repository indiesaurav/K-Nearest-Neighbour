# The Lord is my shepherd
# Jesus is my Saviour!
# Jesus is Great!
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#loading the data set
mpm = pd.read_csv('Mobile_data.csv')
mpm.shape #2000, 15
mpm.info()
mpm.isnull().sum() #Data has no missing values

#Target variable - price_range
'The value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).'

mpm.price_range.describe()
'''
count    2000.000000
mean        1.500000
std         1.118314
min         0.000000
25%         0.750000
50%         1.500000
75%         2.250000
max         3.000000'''
mpm.price_range.value_counts()
'''
3    500
2    500
1    500
0    500'''

#Count/Bar Plot
sns.countplot(x='price_range', data=mpm)
plt.title('Counts of Mobile Price Range')

#Box Plot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'blue')
mpm.price_range.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Mobile Price Range')

#battery_power
'Total energy a battery can store in one time measured in mAh'

mpm.battery_power.describe()
'''
count    2000.000000
mean     1238.518500
std       439.418206
min       501.000000
25%       851.750000
50%      1226.000000
75%      1615.250000
max      1998.000000'''

#Histogram
plt.hist(mpm.battery_power, bins='auto', facecolor='blue' )
plt.xlabel('battery_power')
plt.ylabel('counts')
plt.title('Histogram of battery_power')

#Plot Battery Power vs Price_range; NOT MEANINGFUL!
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('battery_power')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('battery_power')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('battery_power')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('battery_power')['price_range'].count())
plt.xlabel('Battery Power', size=16)
plt.ylabel('Counts', size=16)
plt.title('Battery Power vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.battery_power.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of battery_power')

#clock_speed
'The speed at which microprocessor executes instructions'

mpm.clock_speed.describe()
'''
count    2000.000000
mean        1.522250
std         0.816004
min         0.500000
25%         0.700000
50%         1.500000
75%         2.200000
max         3.000000'''

#Histogram; better plot barplot also!
plt.hist(mpm.battery_power, bins='auto', facecolor='blue' )
plt.xlabel('battery_power')
plt.ylabel('counts')
plt.title('Histogram of battery_power')

#Barplot/ Countplot
sns.countplot(x='clock_speed', data=mpm)
plt.xlabel('clock_speed')
plt.ylabel('counts')
plt.title('Barplot of clock_speed')
plt.xticks(rotation=90)
plt.show()

#Plot Clock Speed vs Price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('clock_speed')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('clock_speed')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('clock_speed')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('clock_speed')['price_range'].count())
plt.xlabel('Clock Speed', size=16)
plt.ylabel('Counts', size=16)
plt.title('Clock Speed vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.clock_speed.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of clock_speed')

#fc - Front Camera megapixels
mpm.fc.describe()
'''
count    2000.000000
mean        4.309500
std         4.341444
min         0.000000
25%         1.000000
50%         3.000000
75%         7.000000
max        19.000000'''

#Value counts
mpm.fc.value_counts() #20 Different values

#Barplot/ Countplot
sns.countplot(x='fc', data=mpm)
plt.xlabel('Front Camera megapixels')
plt.ylabel('counts')
plt.title('Barplot of Front Camera megapixels')

#Plot Front Camera vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('fc')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('fc')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('fc')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('fc')['price_range'].count())
plt.xlabel('Front Camera', size=16)
plt.ylabel('Counts', size=16)
plt.title('Front Camera vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.fc.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Front Camera megapixels') #Outliers

#Getting the Iqr, up_lim & low_lim
iqr = mpm.fc.describe()['75%'] - mpm.fc.describe()['25%'] #6.0
up_lim = mpm.fc.describe()['75%']+1.5*iqr #16
len(mpm.fc[mpm.fc > up_lim]) #18

for i in np.arange(16,19,1):
    outliers = len(mpm.fc[mpm.fc > i])
    print('At a limit of :', i, 'There are', outliers, 'outliers')
'''
At a limit of : 16 There are 18 outliers
At a limit of : 17 There are 12 outliers
At a limit of : 18 There are 1 outliers'''

#int_memory - Internal Memory in Gigabytes
mpm.int_memory.describe()
'''
count    2000.000000
mean       32.046500
std        18.145715
min         2.000000
25%        16.000000
50%        32.000000
75%        48.000000
max        64.000000'''

#Value counts
mpm.int_memory.value_counts() #63 different values

#Barplot/ Countplot
fig = plt.gcf() #gcf - Get the current figure
fig.set_size_inches(12, 8)
sns.countplot(x='int_memory', data=mpm)
plt.xlabel('Internal Memory in Gigabytes')
plt.ylabel('counts')
plt.title('Barplot of Internal Memory in Gigabytes')
plt.xticks(rotation=90)
plt.show()

#Plot Internal Memory vs price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('int_memory')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('int_memory')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('int_memory')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('int_memory')['price_range'].count())
plt.xlabel('Internal Memory', size=16)
plt.ylabel('Counts', size=16)
plt.title('Internal Memory vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.int_memory.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Internal Memory in Gigabytes') #No Outliers

#m_dep - Mobile Depth in cm
mpm.m_dep.describe()
'''
count    2000.000000
mean        0.501750
std         0.288416
min         0.100000
25%         0.200000
50%         0.500000
75%         0.800000
max         1.000000'''

#Value counts
mpm.m_dep.value_counts() #10 different values

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='m_dep', data=mpm)
plt.xlabel('Mobile Depth in cm', size=16)
plt.ylabel('counts', size=16)
plt.title('Barplot of Mobile Depth in cm', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Mobile Depth vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('m_dep')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('m_dep')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('m_dep')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('m_dep')['price_range'].count())
plt.xlabel('Mobile Depth in cm', size=16)
plt.ylabel('Counts', size=16)
plt.title('Mobile Depth in cm vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.m_dep.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Mobile Depth in cm') #No outliers

#mobile_wt - Weight of the mobile phone
mpm.mobile_wt.describe()
'''
count    2000.000000
mean      140.249000
std        35.399655
min        80.000000
25%       109.000000
50%       141.000000
75%       170.000000
max       200.000000'''

#Value counts
mpm.mobile_wt.value_counts() #121 counts

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(18, 8)
sns.countplot(x='mobile_wt', data=mpm)
plt.xlabel('Weight of the mobile phone', size=16)
plt.ylabel('counts', size=16)
plt.title('Barplot of Weight of the mobile phone', size=20)
plt.xticks(rotation=90)
plt.show()

#Histogram
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.hist(mpm.mobile_wt)
plt.xlabel('Weight of the mobile phone', size=16)
plt.ylabel('counts', size=16)
plt.title('Histogram of Weight of the mobile phone', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Mobile Weight vs price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('mobile_wt')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('mobile_wt')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('mobile_wt')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('mobile_wt')['price_range'].count())
plt.xlabel('Mobile Weight', size=16)
plt.ylabel('Counts', size=16)
plt.title('Mobile Weight vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.mobile_wt.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Weight of the mobile phone') #No outliers

#n_cores - Number of cores of a processor
mpm.n_cores.describe()
'''
count    2000.000000
mean        4.520500
std         2.287837
min         1.000000
25%         3.000000
50%         4.000000
75%         7.000000
max         8.000000'''

#Value counts
mpm.n_cores.value_counts() 
'''
4    274
7    259
8    256
2    247
5    246
3    246
1    242
6    230'''

#Barplot/ Countplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='n_cores', data=mpm)
plt.xlabel('Number of cores of a processor', size=16)
plt.ylabel('counts', size=16)
plt.title('Barplot of Number of cores of a processor', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Number of cores of a processor  vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('n_cores')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('n_cores')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('n_cores')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('n_cores')['price_range'].count())
plt.xlabel('Number of cores of a processor', size=16)
plt.ylabel('Counts', size=16)
plt.title('Number of cores of a processor vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.n_cores.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Number of cores of a processor') #No Outliers

#pc - Primary Camera megapixels
mpm.pc.describe()
'''
count    2000.000000
mean        9.916500
std         6.064315
min         0.000000
25%         5.000000
50%        10.000000
75%        15.000000
max        20.000000'''

#Value counts
mpm.pc.value_counts() #21 different values

#Barplot
fig = plt.gcf()
fig.set_size_inches(12, 8)
sns.countplot(x='pc', data=mpm)
plt.xlabel('Primary Camera megapixels', size=16)
plt.ylabel('counts', size=16)
plt.title('Barplot of Primary Camera megapixels', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Primary Camera megapixels vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('pc')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('pc')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('pc')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('pc')['price_range'].count())
plt.xlabel('Primary Camera megapixels', size=16)
plt.ylabel('Counts', size=16)
plt.title('Primary Camera megapixels vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.pc.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Primary Camera megapixels') #No outliers

#px_height - Pixel Resolution Height
mpm.px_height.describe()
'''
count    2000.000000
mean      645.108000
std       443.780811
min         0.000000
25%       282.750000
50%       564.000000
75%       947.250000
max      1960.000000'''

#Value counts
mpm.px_height.value_counts() #1137 different values

#Histogram
plt.hist(mpm.px_height, bins='auto')
plt.xlabel('Pixel Resolution Height', size=16)
plt.ylabel('counts', size=16)
plt.title('Histogram of Pixel Resolution Height', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Pixel Resolution Height vs price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('px_height')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('px_height')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('px_height')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('px_height')['price_range'].count())
plt.xlabel('Pixel Resolution Height', size=16)
plt.ylabel('Counts', size=16)
plt.title('Pixel Resolution Height vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.px_height.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Pixel Resolution Height') #Outliers

#Getting the Iqr, up_lim & low_lim
iqr = mpm.px_height.describe()['75%'] - mpm.px_height.describe()['25%'] #664.5
up_lim = mpm.px_height.describe()['75%']+1.5*iqr #1944.0
len(mpm.px_height[mpm.px_height > up_lim]) #2 outliers

#px_width - Pixel Resolution Width
mpm.px_width.describe()
'''
count    2000.000000
mean     1251.515500
std       432.199447
min       500.000000
25%       874.750000
50%      1247.000000
75%      1633.000000
max      1998.000000'''

#Value counts
mpm.px_width.value_counts() #1109 Different values

#Histogram
plt.hist(mpm.px_width, bins='auto')
plt.xlabel('Pixel Resolution Width', size=16)
plt.ylabel('counts', size=16)
plt.title('Histogram of Pixel Resolution Width', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Pixel Resolution Width vs price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('px_width')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('px_width')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('px_width')['price_range'].count(), color='red')
plt.plot(mpm[mpm.price_range==3].groupby('px_width')['price_range'].count())
plt.xlabel('Pixel Resolution Width', size=16)
plt.ylabel('Counts', size=16)
plt.title('Pixel Resolution Width vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.px_width.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Boxplot of Pixel Resolution Width') #No outliers

#Scatter plot - Pixel Height vs Width
sns.scatterplot(x='px_height', y='px_width', data=mpm, hue='price_range')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('Scatter plot b/w Pixel Height & Width with Price')

#Correlation
np.corrcoef(mpm.px_height, mpm.px_width) #.51

#ram - Random Access Memory in MegaBytes
mpm.ram.describe()
'''
count    2000.000000
mean     2124.213000
std      1084.732044
min       256.000000
25%      1207.500000
50%      2146.500000
75%      3064.500000
max      3998.000000'''

#Value counts
mpm.ram.value_counts() #1562 different values 

#Histogram
plt.hist(mpm.ram, bins='auto')
plt.xlabel('Random Access Memory in MegaBytes', size=16)
plt.ylabel('counts', size=16)
plt.title('Random Access Memory in MegaBytes', size=20)
plt.xticks(rotation=90)
plt.show()

#Plot Random Access Memory vs price_range
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.plot(mpm[mpm.price_range==0].groupby('ram')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('ram')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('ram')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('ram')['price_range'].count())
plt.xlabel('Random Access Memory', size=16)
plt.ylabel('Counts', size=16)
plt.title('Random Access Memory vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.ram.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Random Access Memory in MegaBytes') #No outliers

#Scatterplot - Ram & Internal Memory with price_range
sns.scatterplot(x='ram', y='int_memory', data=mpm, hue='price_range')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('Scatter plot b/w Ram & Int_Memory with Price')

np.corrcoef(mpm.ram, mpm.int_memory) #.03 - Interestingly there is no correlation

#Scatter plot - Ram & n_cores (No of cores of processor) with price_range
sns.scatterplot(x='ram', y='n_cores', data=mpm, hue='price_range')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('Scatter plot b/w Ram & No of cores of processor with Price')

np.corrcoef(mpm.ram, mpm.n_cores) #0.004 No correlation

#sc_h - Screen Height of mobile in cm
mpm.sc_h.describe()
'''
count    2000.000000
mean       12.306500
std         4.213245
min         5.000000
25%         9.000000
50%        12.000000
75%        16.000000
max        19.000000'''

#Value counts
mpm.sc_h.value_counts() #14 different values

#Barplot/ Countplot
sns.countplot(x='sc_h', data=mpm)
plt.xlabel('Screen Height of mobile in cm', size=16)
plt.ylabel('counts', size=16)
plt.title('Screen Height of mobile in cm', size=20)
plt.show()

#Plot Screen Height of mobile in cm vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('sc_h')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('sc_h')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('sc_h')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('sc_h')['price_range'].count())
plt.xlabel('Screen Height of mobile in cm', size=16)
plt.ylabel('Counts', size=16)
plt.title('Screen Height of mobile in cm vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.sc_h.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Screen Height of mobile in cm')

#sc_w - Screen Width of mobile in cm
mpm.sc_w.describe()
'''
count    2000.000000
mean        5.767000
std         4.356398
min         0.000000
25%         2.000000
50%         5.000000
75%         9.000000
max        18.000000'''

#Value counts
mpm.sc_w.value_counts() #19 different values

#Barplot/ Countplot
sns.countplot(x='sc_w', data=mpm)
plt.xlabel('Screen Width of mobile in cm', size=16)
plt.ylabel('counts', size=16)
plt.title('Screen Width of mobile in cm', size=20)
plt.show()

#Plot Screen Width of mobile in cm vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('sc_w')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('sc_w')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('sc_w')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('sc_w')['price_range'].count())
plt.xlabel('Screen Width of mobile in cm', size=16)
plt.ylabel('Counts', size=16)
plt.title('Screen Width of mobile in cm vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.sc_w.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Screen Width of mobile in cm') #Outliers

#Scatter Plot b/w Screen Height & width with price
sns.scatterplot(x='sc_h', y='sc_w', data=mpm, hue='price_range')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('Scatter Plot b/w Screen Height & width with price')

np.corrcoef(mpm.sc_h, mpm.sc_w) #.50

#talk_time - The longest time that a single battery charge will last
mpm.talk_time.describe()
'''
count    2000.000000
mean       11.011000
std         5.463955
min         2.000000
25%         6.000000
50%        11.000000
75%        16.000000
max        20.000000'''

#Value counts
mpm.talk_time.value_counts() #19 different values

#Barplot/ Coutplot
sns.countplot(x='talk_time', data=mpm)
plt.xlabel('Time that a battery charge will last', size=16)
plt.ylabel('counts', size=16)
plt.title('Time that a battery charge will last', size=20)
plt.show()

#Plot Time that a battery charge will last in cm vs price_range
plt.plot(mpm[mpm.price_range==0].groupby('talk_time')['price_range'].count())
plt.plot(mpm[mpm.price_range==1].groupby('talk_time')['price_range'].count())
plt.plot(mpm[mpm.price_range==2].groupby('talk_time')['price_range'].count())
plt.plot(mpm[mpm.price_range==3].groupby('talk_time')['price_range'].count())
plt.xlabel('Time that a battery charge will last', size=16)
plt.ylabel('Counts', size=16)
plt.title('Time that a battery charge will last vs Price_range', size=20)
plt.legend(labels = ('Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'),
           bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
mpm.talk_time.plot.box(color=props2, patch_artist = True, vert = False)
plt.title('Time that a battery charge will last') #No outliers

#Correlation with respect to Price Range; THOUGH NOT APPLICABLE RV=catg
mpm.corr()['price_range'].sort_values(ascending=False)
'''
price_range      1.000000
ram              0.917046 #Ram will be key predictor
battery_power    0.200723
px_width         0.165818
px_height        0.148858
int_memory       0.044435
sc_w             0.038711
pc               0.033599
sc_h             0.022986
fc               0.021998
talk_time        0.021859
n_cores          0.004399
m_dep            0.000853
clock_speed     -0.006606
mobile_wt       -0.030302'''

#______________________________________________________________________________
#Assigning Predictors & Target Variable
mpm.info()
x = mpm.iloc[:,:-1] #14 Variables
x.info()
y = mpm.iloc[:,-1]
y

#Splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state = 25,test_size=0.25)

len(x_train) #1500
len(x_test) #500
len(y_train) #1500
len(y_test) #500

#Building Model @ n_neighbors = 13
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13)
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test)

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8

#ROC Curve is primarily for binary class not for multiclass

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0    8  122    4  134
3          0    0    6  112  118
All      124  131  129  116  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.91      0.97      0.94       123
           2       0.95      0.91      0.93       134
           3       0.97      0.95      0.96       118

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500'''

#Building Model @ n_neighbors = 20
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 20)
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #93.2

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #93.2

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0   11  117    6  134
3          0    0    9  109  118
All      124  134  127  115  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.89      0.97      0.93       123
           2       0.92      0.87      0.90       134
           3       0.95      0.92      0.94       118

    accuracy                           0.93       500
   macro avg       0.93      0.93      0.93       500
weighted avg       0.93      0.93      0.93       500'''

#Building Model @ n_neighbors = 10
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #93

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #93

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  120    0    0  123
2          0   13  114    7  134
3          0    0    8  110  118
All      124  137  122  117  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.88      0.98      0.92       123
           2       0.93      0.85      0.89       134
           3       0.94      0.93      0.94       118

    accuracy                           0.93       500
   macro avg       0.93      0.93      0.93       500
weighted avg       0.93      0.93      0.93       500'''

#Building Model @ n_neighbors = 13 with algorithm = 'ball_tree'
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, algorithm= 'ball_tree')
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #94.8

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0    8  122    4  134
3          0    0    6  112  118
All      124  131  129  116  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.91      0.97      0.94       123
           2       0.95      0.91      0.93       134
           3       0.97      0.95      0.96       118

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500'''

#Building Model @ n_neighbors = 13 with algorithm = 'kd_tree'
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, algorithm= 'kd_tree')
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #94.8

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0    8  122    4  134
3          0    0    6  112  118
All      124  131  129  116  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.91      0.97      0.94       123
           2       0.95      0.91      0.93       134
           3       0.97      0.95      0.96       118

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500'''
#__________________________________________________________________brute
#Building Model @ n_neighbors = 13 with algorithm = 'brute'
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13, algorithm= 'brute')
print(knn) 
mpm_knn = knn.fit(x_train, y_train) 
print(mpm_knn)

#Applying on Test data for prediction
y_pred = mpm_knn.predict(x_test)
print(y_pred)

#Prediction Score
mpm_knn.score(x_test, y_test) #94.8

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0    8  122    4  134
3          0    0    6  112  118
All      124  131  129  116  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.91      0.97      0.94       123
           2       0.95      0.91      0.93       134
           3       0.97      0.95      0.96       118

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500'''

#_________________________Building Model with Neighborhood Components Analysis(NCA)
#NCA is a distance metric algorithm
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=13)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
mpm_nca_pipe = nca_pipe.fit(x_train, y_train)
print(mpm_nca_pipe)

#Applying on Test data for prediction
y_pred = mpm_nca_pipe.predict(x_test)
print(y_pred)

#Prediction Score
mpm_nca_pipe.score(x_test, y_test) #94.8

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #94.8

#Classification Matrix
pd.crosstab(y_test, y_pred, margins = True, rownames=['Actual'], colnames=['Predict']) 
'''
Predict    0    1    2    3  All
Actual                          
0        121    4    0    0  125
1          3  119    1    0  123
2          0    8  122    4  134
3          0    0    6  112  118
All      124  131  129  116  500'''

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       125
           1       0.91      0.97      0.94       123
           2       0.95      0.91      0.93       134
           3       0.97      0.95      0.96       118

    accuracy                           0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500'''