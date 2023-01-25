#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import datetime
import random
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#importing dataset
df = pd.read_csv('/kaggle/input/d/ayush12negi/mental/Student Mental health.csv')

df.head()
# Renaming columns so it's easier to call them later on when coding. 
df.rename(columns = {'Choose your gender':'Gender',
                     'What is your course?':'Course',
                     'Your current year of Study': 'Year',
                     'What is your CGPA?': 'CGPA',
                     'Marital status': 'Married',
                     'Do you have Depression?': 'Depression',
                     'Do you have Anxiety?': 'Anxiety',
                     'Do you have Panic attack?': 'Panic Attack',
                     'Did you seek any specialist for a treatment?': 'Seeking Treatment'},inplace = True)

# Dropping the timestamp table as I don't see it as important
df = df.drop(columns = "Timestamp")
df.info()
df.shape
df.dtypes
df.nunique()
df.isnull().sum()
df = df.dropna(how='any',axis=0) 
df.isnull().sum()
df['Course'].unique()
df['Course'] = df['Course'].str.lower()

# Removing unnecessary spaces from the Course and CGPA column
df['Course'] = df['Course'].str.strip()
df['CGPA'] = df['CGPA'].str.strip()

df['Course'].unique()

# I replaced each of the courses into similar courses/categories as well as cleaning up some of the spelling
df['Course'].replace({'pendidikan islam': 'education',
                      'laws': 'law',
                      'engine': 'engineering',
                      'engin': 'engineering',
                      'diploma nursing': 'nursing',
                      'bit': 'it',
                      'kirkhs': 'irkhs',
                      'usuluddin': 'irkhs',
                      'fiqh fatwa': 'irkhs',
                      'fiqh': 'irkhs',
                      'human resources': 'human sciences',
                      'econs': 'economics',
                      'kenms': 'economics',
                      'enm': 'economics',
                      'kop': 'pharmacy',
                      'koe': 'education',
                      'benl': 'education',
                      'islamic education': 'education',
                      'mathemathics': 'mathematics',
                      'diploma tesl': 'education',
                      'mhsc': 'human sciences',
                      'taasl': 'education',
                      'ala': 'human sciences',
                      'bcs': 'computer science',
                      'malcom': 'communication'}, inplace=True)

df['Course'].unique()
# Capitalizing each word in their courses as well as replacing the courses with all uppercase. 
# Most likely there is also a method to change the 3 specific courses into uppercase. 
# Since there were a few, I decided to just replace them.
df['Course'] = df['Course'].str.title()
df['Course'].replace({'It': 'IT',
                      'Irkhs': 'IRKHS',
                      'Cts': 'CTS'
                          }, inplace=True)
df['Course'].unique()

df['CGPA'].replace({'0 - 1.99': 1,
                    '2.00 - 2.49': 2,
                    '2.50 - 2.99': 3,
                    '3.00 - 3.49': 4,
                    '3.50 - 4.00': 5}, inplace=True)
​
# I also capitalized the Year column to group all entries that had different spellings.
df ['Year'] = df['Year'].str.title()

df.info()
df

# Created a new column title "Mental Health Issues" and marking a student with either Yes or No for MH Issues
df.loc[(df["Depression"]=="Yes") | (df["Anxiety"]=="Yes") | (df["Panic Attack"]=="Yes"),'Mental Health Issues']= 'Yes'
df.loc[(df["Depression"]=="No") & (df["Anxiety"]=="No") & (df["Panic Attack"]=="No"),'Mental Health Issues']= 'No'


# Creating a dataframe with students who have mental health issues or not
mental_health_yes = df[df["Mental Health Issues"] == 'Yes']
mental_health_no = df[df["Mental Health Issues"] == 'No']

# Showing the count of Students grouped via Mental Health Issues
df.groupby(['Mental Health Issues']).count()

df.rename(columns = {'Choose your gender': 'gender'}, inplace = True)

plt.figure(figsize=(10,10))
plt.hist(df['Age'],color='r')
plt.title("Age distribution");

plt.figure(figsize=(12,6))
plt.title("gender distribution")
g = plt.pie(df.gender.value_counts(), explode=(0.025,0.025), labels=df.gender.value_counts().index, colors=['skyblue','navajowhite'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()

# Setting default seaborn theme
sns.set()

# Plotting the number of students who have MH Issues vs students with no MH Issues
plt.pie(df['Mental Health Issues'].value_counts(), labels = ['Yes', 'No'], autopct = '%.f')

plt.title('Number of Students with Mental Health Issues')

plt.show()

mental_health_no_course = mental_health_no.groupby("Course")["Mental Health Issues"].count()
mental_health_yes_course = mental_health_yes.groupby("Course")["Mental Health Issues"].count()

# Sorting from highest to lowest 
mental_health_no_course = mental_health_no_course.sort_values(ascending = False)
mental_health_yes_course = mental_health_yes_course.sort_values(ascending = False)

# Creating a new dataframe for this plot
new_df = pd.concat([mental_health_yes_course, mental_health_no_course], axis=1)

# Renaming the column names
new_df.columns.values[0] = "Has Mental Health Issues"
new_df.columns.values[1] = "No Mental Health Issues"

# Changing the NULL values into 0 and then creating a new column named Total.
new_df.fillna(0, inplace = True)
new_df['Total'] = new_df['Has Mental Health Issues'] + new_df['No Mental Health Issues']

# Sorting
new_df = new_df.sort_values(by=['Total','Has Mental Health Issues', 'No Mental Health Issues'], ascending=False)

new_df = new_df.reset_index(level=0)

new_df.head()

new_df = new_df.sort_values(by=['Total','Has Mental Health Issues', 'No Mental Health Issues'], ascending=True)

# Plotting via stacked horizontal bars
ax = new_df.plot(x="Course", y="Has Mental Health Issues", kind="barh", color = 'orange')

new_df.plot(x="Course", y="No Mental Health Issues", kind="barh", ax=ax, color = 'steelblue')
plt.title('Count of Students with MH Issues')
#graphs for my model
plt.figure(figsize=(10,10))
sns.countplot(df['Your current year of Study'],hue=df['gender'])
plt.title("Students studyig in particular year");

plt.figure(figsize=(10,10))
sns.countplot(df['Do you have Anxiety?'],hue=df['Do you have Depression?'])
plt.title("Students studyig in particular year");
plt.show()

plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Do you have Anxiety?", hue="gender", data=df)
plt.title("Anxiety by Gender")
plt.show()

plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Do you have Depression?", hue="gender", data=df)
plt.title("Depression by Gender")
plt.show()

plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Do you have Anxiety?", hue="Your current year of Study", data=df)
plt.title("Anxiety by study year")
plt.show()

plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Do you have Depression?", hue="Your current year of Study", data=df)
plt.title("Depression by study year")
plt.show()

plt.figure(figsize=(10,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Do you have Panic attack?", hue="What is your CGPA?", data=df)
plt.title("Panic attack by CGPA")
plt.show()
