import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Load dataset
df = pd.read_csv('dataset/Formula1_Pitstop_Data_2011-2024_all_rounds.csv')

# Drop columns that are not necessary for analysis
drop_cols = ['Round', 'Position', 'Driver', 'PitStops']
df = df.drop(columns=drop_cols)

# Encode categorical variables
label_encoders = {}
for col in ['Circuit', 'Constructor']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardization (StandardScaler)
std_cols = ['FastestStop', 'SlowestStop']
std_scaler = StandardScaler()
df[std_cols] = std_scaler.fit_transform(df[std_cols])

# Normalization (MinMaxScaler)
mm_cols = ['Laps', 'TotalPitStops', 'FirstPitLap', 'LastPitLap']
mm_scaler = MinMaxScaler()
df[mm_cols] = mm_scaler.fit_transform(df[mm_cols])

# Robust scaling
rb_cols = ['AvgPitStopTime']
rb_scaler = RobustScaler()
df[rb_cols] = rb_scaler.fit_transform(df[rb_cols])

# Save the preprocessed dataset
df.to_csv('dataset/Pitstop_Data_Preprocessed.csv', index=False)


## Feature Engineering - Feature Selection

# Utility function - select numeric columns only
def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=["int64", "float64"])

# Feature importance plot using ExtraTreesRegressor
def feature_importance(df: pd.DataFrame, target: str, title: str = "") -> None:
    # Prepare features (X) and target (y)
    X = df.drop(columns=[target])
    y = df[target]
    n = 10 # Number of top features to display

    # Train the Extra Trees Regressor model
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X, y)

    # Get feature importances and plot the top n features
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    plt.title(f"{title} - Feature Importance (Top {n})")
    feat_imp.nlargest(n).plot(kind="barh")
    plt.gca().invert_yaxis()  # Ensure the most important feature appears at the top

# Correlation heatmap
def corr_heatmap(df: pd.DataFrame, title: str = "") -> None:
    # Calculate the correlation matrix for numeric columns
    corrmat = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    plt.title(f"{title} - Correlation Matrix")
    sns.heatmap(
        corrmat,
        annot=True,
        cmap="RdYlGn",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": .8},
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

# Main function to generate plots
def get_plots(csv_path: str,
              target_col: str = "Top10",
              show_corr: bool = True) -> None:
    df = pd.read_csv(csv_path)

    # The dataset contains only numeric columns including encoded categorical features
    num_df = _numeric_df(df)

    # Feature importance
    feature_importance(num_df, target=target_col, title="F1 Pit-Stop")

    # Correlation heatmap
    if show_corr:
        corr_heatmap(num_df, title="F1 Pit-Stop")

    plt.show()

# Example execution
get_plots(
    csv_path="dataset/Pitstop_Data_Preprocessed.csv",
    target_col="Top10",
    show_corr=True,
)
