import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Read Dataset
df = pd.read_csv('dataset/Formula1_Pitstop_Data_1950-2024_all_rounds.csv', na_values='-')

# List all data
pd.set_option('display.max_seq_items', None)

# Print the statistical description of the dataset
print("=" * 50)
print("Before Preprocessing")
print("=" * 50 + "\n")

print("=" * 50)
print("[df.describe()]\n")
print(df.describe())
print("=" * 50 + "\n")

print("=" * 50)
print("[df.info()]\n")
print(df.info())
print("=" * 50 + "\n")

# Number of null data per column
print("=" * 50)
print("[Before cleaning the dirty data]\n")
print(df.isnull().sum())
print("=" * 50 + "\n")

## Data Restructuring - Table Decomposition
# Filter data from 2011 onwards
new_df = df[df['Season'] >= 2011].copy()

## Data Value Changes - Cleaning dirty data - Missing Data
# Remove rows with missing values in 'AvgPitStopTime'
new_df = new_df.dropna(subset=['AvgPitStopTime'])

## Data Value Changes - Cleaning dirty data - Wrong Data
# Define wrong-to-correct name mappings
changenames = {
    'Circuit': {
        'NÃ¼rburgring': 'Nurburgring',
        'AutÃ³dromo JosÃ© Carlos Pace': 'Autodromo Jose Carlos Pace',
        'AutÃ³dromo Internacional do Algarve': 'Autodromo Internacional do Algarve',
        'AutÃ³dromo Hermanos RodrÃ­guez': 'Autodromo Hermanos Rodriguez'
    },
    'Driver': {
        'SÃ©bastien Buemi': 'Sebastien Buemi',
        'JÃ©rÃ´me d\'Ambrosio': 'Jerome Dambrosio',
        'Sergio PÃ©rez': 'Sergio Perez',
        'Kimi RÃ¤ikkÃ¶nen': 'Kimi Raikkonen',
        'Jean-Ã‰ric Vergne': 'Jean-Eric Vergne',
        'Nico HÃ¼lkenberg': 'Nico Hulkenberg',
        'Esteban GutiÃ©rrez': 'Esteban Gutierrez',
        'AndrÃ© Lotterer': 'Andre Lotterer'
    }
}

# Apply corrections to specified columns
for column, mapping in changenames.items():
    new_df[column] = new_df[column].replace(mapping)

## Data Value Changes - Cleaning dirty data - Outliers
# Calculate outlier threshold using the IQR method
Q1 = new_df['AvgPitStopTime'].quantile(0.25)
Q3 = new_df['AvgPitStopTime'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR

print(f"IQR-based outlier threshold: {outlier_threshold:.2f} seconds")

# Identify outliers
outliers = new_df[new_df['AvgPitStopTime'] > outlier_threshold]
print(f"Number of outliers detected: {len(outliers)}")

# Create a new DataFrame excluding outliers
new_df = new_df[new_df['AvgPitStopTime'] <= outlier_threshold].copy()
print(f"DataFrame size after outlier removal: {new_df.shape}")

## Data Value Changes - Text preprocessing - Noise removal
## Feature Engineering - Feature Creation - Deriving Features from Existing Features

# ast: turns a string into a Python object
import ast

def clean_pitstops(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Parse the stringified list into a real Python list[dict]
    df["PitStops"] = df["PitStops"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    # 2. Derive numerical features
    df["FirstPitLap"] = df["PitStops"].apply(
        lambda lst: min(d["Lap"] for d in lst) if lst else None
    )

    df["LastPitLap"] = df["PitStops"].apply(
        lambda lst: max(d["Lap"] for d in lst) if lst else None
    )

    df["FastestStop"] = df["PitStops"].apply(
        lambda lst: min(d["StopTime"] for d in lst) if lst else None
    )

    df["SlowestStop"] = df["PitStops"].apply(
        lambda lst: max(d["StopTime"] for d in lst) if lst else None
    )

    return df

new_df = clean_pitstops(new_df)

## Feature Engineering - Feature Creation - Deriving Features from Existing Features
# Create a column to see if it's in the Top10
new_df['Top10'] = (new_df['Position'] <= 10).astype(int)

# Save the cleaned and filtered original data
new_df.to_csv('dataset/Formula1_Pitstop_Data_2011-2024_all_rounds.csv', index=False)

# Display the first 50 rows of the dataset
print("=" * 50)
print("After Preprocessing")
print("=" * 50 + "\n")
new_df.head(50)

# Print the statistical description of the dataset after preprocessing
print("=" * 50)
print("[df.describe()]\n")
print(new_df.describe())
print("=" * 50 + "\n")

print("=" * 50)
print("[df.info()]\n")
print(new_df.info())
print("=" * 50 + "\n")

# Number of null data per column
print("=" * 50)
print("[After cleaning the dirty data]\n")
print(new_df.isnull().sum())
print("=" * 50 + "\n")
