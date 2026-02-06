# Sample African Dataset Generator
# This script generates sample African health/economic data with missing values
# for PCA assignment purposes

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate sample African health and economic data
data = {
    'Country': np.random.choice([
        'Nigeria', 'South Africa', 'Kenya', 'Ethiopia', 'Egypt', 
        'Ghana', 'Tanzania', 'Uganda', 'Morocco', 'Angola',
        'Sudan', 'Algeria', 'Mozambique', 'Zimbabwe', 'Rwanda'
    ], n_samples),
    
    'Region': np.random.choice(['East', 'West', 'North', 'South', 'Central'], n_samples),
    
    'GDP_per_capita': np.random.uniform(500, 15000, n_samples),
    'Life_Expectancy': np.random.uniform(50, 75, n_samples),
    'Infant_Mortality_Rate': np.random.uniform(20, 100, n_samples),
    'Healthcare_Expenditure_Percent': np.random.uniform(2, 12, n_samples),
    'Education_Expenditure_Percent': np.random.uniform(2, 8, n_samples),
    'Unemployment_Rate': np.random.uniform(5, 35, n_samples),
    'Population_Millions': np.random.uniform(5, 200, n_samples),
    'Urban_Population_Percent': np.random.uniform(20, 85, n_samples),
    'Internet_Users_Percent': np.random.uniform(10, 70, n_samples),
    'Literacy_Rate': np.random.uniform(40, 95, n_samples),
    'Access_to_Clean_Water_Percent': np.random.uniform(30, 95, n_samples),
    'Malaria_Incidence_per_1000': np.random.uniform(0, 400, n_samples),
    'HIV_Prevalence_Percent': np.random.uniform(0.5, 20, n_samples),
    'Tuberculosis_Incidence_per_100k': np.random.uniform(50, 500, n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce missing values randomly (10-20% missing in numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.15), replace=False)
    df.loc[missing_indices, col] = np.nan

# Introduce some missing values in categorical columns too
categorical_cols = ['Region']
for col in categorical_cols:
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, col] = np.nan

# Save to CSV
df.to_csv('african_health_economic_data.csv', index=False)

print("Sample dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
