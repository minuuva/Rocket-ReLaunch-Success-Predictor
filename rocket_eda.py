# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/cmw/Downloads/mission_launches.csv'
data = pd.read_csv(file_path)

data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
data.head()

# %%
# Check for data types and missing values
print("Data Types and Missing Values:\n", data.info())

# Summary statistics of numerical features
print("\nSummary Statistics:\n", data.describe())

# Check for missing values in each column
print("\nMissing Values in Each Column:\n", data.isnull().sum())

# %%
# Plot the distribution of mission prices
plt.figure(figsize=(12, 6))
sns.histplot(data['Price'].dropna(), bins=30, kde=True)
plt.title('Distribution of Mission Prices')
plt.xlabel('Price (in millions)')
plt.xticks(rotation=90)
plt.show()

# Plot distribution of mission outcomes
plt.figure(figsize=(6, 4))
sns.countplot(x='Mission_Status', data=data)
plt.title('Distribution of Mission Outcomes')
plt.xlabel('Mission Status')
plt.show()

# %%
# Analyze the impact of the organization on mission outcomes
plt.figure(figsize=(10, 6))
sns.countplot(x='Organisation', hue='Mission_Status', data=data)
plt.title('Mission Status by Organization')
plt.xticks(rotation=90)
plt.show()

# %%
# Check for class imbalance in the target variable
mission_status_counts = data['Mission_Status'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=mission_status_counts.index, y=mission_status_counts.values)
plt.title('Class Imbalance in Mission Outcomes')
plt.xlabel('Mission Status')
plt.ylabel('Count')
plt.show()

# %%
if 'Price' in data.columns:
    data['Price'] = data['Price'].replace({',': ''}, regex=True).astype(float)

# Convert 'Mission_Status' to a binary variable: 1 for Success, 0 for Failure
data['Mission_Success'] = data['Mission_Status'].apply(lambda x: 1 if x == 'Success' else 0)

# Correlation between Price and Mission Success
price_success_corr = data[['Price', 'Mission_Success']].corr().iloc[0, 1]
print(f"Correlation between Price and Mission Success: {price_success_corr:.2f}")

# Visualize the relationship between Price and Mission Success
plt.figure(figsize=(8, 6))
sns.boxplot(x='Mission_Success', y='Price', data=data)
plt.title('Price vs Mission Success')
plt.xlabel('Mission Success (1=Success, 0=Failure)')
plt.ylabel('Price (in millions)')
plt.show()


# %%
# Convert the Date column to datetime format
data['Launch_Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Extract year, month, and day of the week from the launch date
data['Launch_Year'] = data['Launch_Date'].dt.year
data['Launch_Month'] = data['Launch_Date'].dt.month
data['Launch_DayOfWeek'] = data['Launch_Date'].dt.day_name()

# Plot mission success rates by year
plt.figure(figsize=(10, 6))
sns.countplot(x='Launch_Year', hue='Mission_Status', data=data)
plt.title('Mission Outcomes by Launch Year')
plt.xticks(rotation=90)
plt.show()

# Plot mission success rates by month
plt.figure(figsize=(8, 6))
sns.countplot(x='Launch_Month', hue='Mission_Status', data=data)
plt.title('Mission Outcomes by Launch Month')
plt.show()

# Plot mission success rates by day of the week
plt.figure(figsize=(8, 6))
sns.countplot(x='Launch_DayOfWeek', hue='Mission_Status', data=data)
plt.title('Mission Outcomes by Day of the Week')
plt.xticks(rotation=45)
plt.show()

# %%
# Calculate success rate for each organization
organization_success_rate = data.groupby('Organisation')['Mission_Success'].mean().sort_values(ascending=False)

# Plot success rates by organization
plt.figure(figsize=(12, 6))
organization_success_rate.plot(kind='bar', color='skyblue')
plt.title('Success Rate by Organization')
plt.ylabel('Success Rate')
plt.xlabel('Organization')
plt.xticks(rotation=90)
plt.show()

# %%
# Calculate success rate by rocket status
rocket_status_success_rate = data.groupby('Rocket_Status')['Mission_Success'].mean().sort_values(ascending=False)

# Plot success rates by rocket status
plt.figure(figsize=(8, 6))
rocket_status_success_rate.plot(kind='bar', color='coral')
plt.title('Success Rate by Rocket Status')
plt.ylabel('Success Rate')
plt.xlabel('Rocket Status')
plt.xticks(rotation=45)
plt.show()


