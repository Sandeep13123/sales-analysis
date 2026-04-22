
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
df = pd.read_csv(r"C:\Users\adija\Downloads\10000 Sales Records.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

df = df.dropna()
df['Profit Margin'] = df['Total Profit'] / df['Total Revenue']

print(df.describe())
print(df.info())

plt.figure()
sns.barplot(x='Region', y='Total Revenue', data=df)
plt.xticks(rotation=45)
plt.title("Sales by Region")
plt.show()

plt.figure()
sns.histplot(df['Total Profit'], bins=30, kde=True)
plt.title("Profit Distribution")
plt.show()

plt.figure()
monthly_sales = df.groupby('Month')['Total Revenue'].sum()
monthly_sales.plot(kind='line')
plt.title("Monthly Sales Trend")
plt.show()


plt.figure()
corr = df[['Total Revenue', 'Total Cost', 'Total Profit']].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

plt.figure()
sns.boxplot(x=df['Total Profit'])
plt.title("Outlier Detection (Profit)")
plt.show()


plt.figure()
df.groupby('Region')['Total Revenue'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title("Revenue Share by Region")
plt.ylabel('')
plt.show()

plt.figure()
sns.scatterplot(x='Total Cost', y='Total Revenue', data=df)
plt.title("Cost vs Revenue")
plt.show()


plt.figure()
sns.countplot(y='Item Type', data=df)
plt.title("Count of Orders by Item Type")
plt.show()


sns.pairplot(df[['Total Revenue', 'Total Cost', 'Total Profit']])
plt.show()




t_stat, p_val = stats.ttest_1samp(df['Total Profit'], 20000)
print("T-test:", t_stat, p_val)


stat, p = stats.shapiro(df['Total Profit'].sample(500))
print("Shapiro Test:", stat, p)

contingency = pd.crosstab(df['Region'], df['Item Type'])
chi2, p, dof, exp = stats.chi2_contingency(contingency)
print("Chi-square:", chi2, p)

X = df[['Units Sold', 'Total Cost']]
y = df['Total Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Linear Regression Score:", lr_model.score(X_test, y_test))

df['Profit_Class'] = df['Total Profit'].apply(lambda x: 1 if x > 20000 else 0)

X = df[['Units Sold', 'Total Cost']]
y = df['Profit_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

print("Logistic Regression Accuracy:", log_model.score(X_test, y_test))
