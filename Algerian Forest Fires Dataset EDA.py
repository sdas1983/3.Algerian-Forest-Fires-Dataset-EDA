# Algerian Forest Fires Dataset EDA

import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

# Loading the dataset
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\Algerian_forest_fires_dataset_UPDATE.csv", skiprows=1)

# Assigning the Region
# 2 Regions - Bejaia and Sidi-Bel Abbes
df['Region'] = 'Bejaia'

for i in range(len(df)):
    if i >= 122:
        df['Region'][i] = 'Sidi-Bel Abbes'

df = df.dropna().reset_index(drop=True)

# Removing invalid rows
df = df[df['day'] != 'day']

# Checking for missing values
print("\n### Checking for Missing Values ###")
print(df.isnull().sum())

# Correcting the Column names - Removing Spaces
df.columns = df.columns.str.strip()

# Dataset Information
print("\n### Dataset Information ###")
print(df.info())

# Dataset Description
print("\n### Dataset Description ###")
print(df.describe())

# Unique values in 'Classes' column
print("\n### Unique values in 'Classes' column ###")
print(df['Classes'].unique())

# Cleaning the 'Classes' column
df['Classes'] = df['Classes'].str.strip()

# Converting the required columns to appropriate data types
convert_to_int = ['day', 'month', 'year', 'RH', 'Ws']
convert_to_float = ['Temperature', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
df[convert_to_int] = df[convert_to_int].astype(int)
df[convert_to_float] = df[convert_to_float].astype(float)

# Dataset Information after type conversion
print("\n### Dataset Information after Type Conversion ###")
print(df.info())

# Dataset Description after type conversion
print("\n### Dataset Description after Type Conversion ###")
print(df.describe())

# Boxplot for Continuous Variables
print("\n### Boxplot for Continuous Variables ###")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']])
plt.title("Boxplot of Continuous Variables")
plt.show()

# Class distribution
print("\n### Class Distribution ###")
ax = sns.countplot(data=df, x='Classes')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title("Class Distribution")
plt.show()

# Class distribution using Plotly
print("\n### Class Distribution (Plotly) ###")
fig = px.bar(df['Classes'].value_counts().reset_index(), x='index', y='Classes', text=df['Classes'].value_counts())
fig.update_traces(textposition='outside')
fig.show()

# Forest Fire Analysis Month-wise
print("\n### Forest Fire Analysis Month-wise ###")
ax = sns.countplot(data=df, x='month', hue='Classes')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title("Forest Fire Analysis by Month")
plt.show()

# Month-wise Forest Fire Analysis using Plotly
print("\n### Month-wise Forest Fire Analysis (Plotly) ###")
df1 = df.groupby(['month', 'Classes'])['month'].size().reset_index(name='count')
fig = px.bar(df1, x='month', y='count', color='Classes', barmode='group', text='count')
fig.update_traces(textposition='outside')
fig.show()

# Forest Fire Analysis Month-wise by Region
print("\n### Forest Fire Analysis Month-wise by Region ###")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
for i, region in enumerate(df['Region'].unique()):
    sns.countplot(data=df[df['Region'] == region], x='month', hue='Classes', ax=axes[i])
    axes[i].set_title(f"Forest Fire Analysis in {region}")
    for p in axes[i].patches:
        height = p.get_height()
        axes[i].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                         textcoords='offset points')
axes[1].legend()
plt.tight_layout()
plt.show()

# Month-wise Forest Fire Analysis by Region using Plotly
print("\n### Month-wise Forest Fire Analysis by Region (Plotly) ###")
df2 = df.groupby(['Region', 'month', 'Classes'])['Region'].size().reset_index(name='count')
fig = px.bar(df2, x='month', y='count', color='Classes', facet_col='Region', text='count', barmode='group')
fig.update_traces(textposition='outside')
fig.show()

# Correlation Heatmap
print("\n### Correlation Heatmap ###")
df_corr = df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']]
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Encoding the 'Region' column
print("\n### Encoding 'Region' Column ###")
df['Region'] = df['Region'].map({'Bejaia': 0, 'Sidi-Bel Abbes': 1})
print(df['Region'].unique())

# Preparing for the model building
X = df.drop(['Classes'], axis=1)
y = df['Classes']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # This is untouched data, so we only do transform here

# Training the Logistic Regression Model
regression = LogisticRegression()
regression.fit(X_train, y_train)

# Saving the model
import pickle
pickle.dump(regression, open('Algerian_Fire_Dataset_Logistic_model.pickle', 'wb'))

# Model Evaluation
print("\n### Model Evaluation ###")
train_score = regression.score(X_train, y_train)
test_score = regression.score(X_test, y_test)
print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")

# Confusion Matrix
print("\n### Confusion Matrix ###")
ConfusionMatrixDisplay.from_estimator(regression, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n### Classification Report ###")
y_pred = regression.predict(X_test)
print(classification_report(y_test, y_pred))

# ROC Curve
print("\n### ROC Curve ###")
RocCurveDisplay.from_estimator(regression, X_test, y_test)
plt.title("ROC Curve")
plt.show()
