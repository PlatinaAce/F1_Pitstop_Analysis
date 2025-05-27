import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Declare global variables for data and model state
df_preprocessed = None
df_original = None
circuit_id_to_name = {}
min_pitstops = None
max_pitstops = None
model = None
X_train = None
X_test = None
y_train = None
y_test = None


def load_data():
    global df_preprocessed, df_original, circuit_id_to_name, min_pitstops, max_pitstops

    df_preprocessed = pd.read_csv('dataset/Pitstop_Data_Preprocessed.csv')
    df_original = pd.read_csv('dataset/Formula1_Pitstop_Data_2011-2024_all_rounds.csv')

    # Create mapping from circuit ID to circuit name (By Visualization)
    unique_circuits = df_original['Circuit'].unique()
    circuit_encoder = LabelEncoder()
    circuit_encoder.fit(unique_circuits)
    circuit_id_to_name = {i: name for i, name in enumerate(circuit_encoder.classes_)}

    # Store original pitstop range
    min_pitstops = df_original['TotalPitStops'].min()
    max_pitstops = df_original['TotalPitStops'].max()

    # Add a column for original pitstop count
    df_preprocessed['OriginalPitStops'] = df_preprocessed['TotalPitStops'].apply(
        lambda x: int(round(x * (max_pitstops - min_pitstops) + min_pitstops))
    )

def basic_eda():
    global df_preprocessed, circuit_id_to_name

    print("\n" + "=" * 50)
    print("BASIC EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print("Dataset Info:")
    print(f"Shape: {df_preprocessed.shape}")
    print(f"Features: {list(df_preprocessed.columns)}")
    print(f"Target distribution:")
    print(df_preprocessed['Top10'].value_counts())

    # Top10 entry rate by circuit
    circuit_top10_rates = df_preprocessed.groupby('Circuit')['Top10'].mean().sort_values(ascending=False)
    print(f"\nTop 10 circuits with highest Top10 entry rates:")
    for i, (circuit_id, rate) in enumerate(circuit_top10_rates.head(10).items()):
        circuit_name = circuit_id_to_name.get(circuit_id, f"Circuit ID {circuit_id}")
        print(f"{i + 1:2d}. {circuit_name[:30]:<30} {rate:.3f}")


def visualize_pitstop_analysis():
    ## Visualize the relationship between the number of pitstops and Top10 entry outcomes
    global df_preprocessed

    # Pitstop count vs Top10 entry frequency
    plt.figure(figsize=(12, 6))
    sns.countplot(x='OriginalPitStops', hue='Top10', data=df_preprocessed)
    plt.title('Frequency of Top10 Entry by Number of Pitstops', fontsize=14)
    plt.xlabel('Number of Pitstops', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(['Outside Top10', 'Top10'])
    plt.tight_layout()
    plt.show()

    # Success rate by pitstop count
    pitstop_success = df_preprocessed.groupby('OriginalPitStops')['Top10'].agg(['count', 'sum'])
    pitstop_success['rate'] = pitstop_success['sum'] / pitstop_success['count']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(pitstop_success.index, pitstop_success['rate'], color='skyblue', alpha=0.7)
    plt.xlabel('Number of Pitstops', fontsize=12)
    plt.ylabel('Top10 Success Rate', fontsize=12)
    plt.title('Top10 Success Rate by Pitstop Count', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(pitstop_success['rate']):
        plt.text(pitstop_success.index[i], v + 0.01, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def circuit_analysis():
    ## Detailed analysis per circuit
    global df_preprocessed, circuit_id_to_name

    # Heatmap of pitstop distribution by circuit
    plt.figure(figsize=(15, 10))

    # Create pivot table
    circuit_pitstop_heatmap = pd.pivot_table(
        df_preprocessed,
        values='Top10',
        index='Circuit',
        columns='OriginalPitStops',
        aggfunc='mean'
    )

    # Select top 10 circuits
    top_circuits = df_preprocessed['Circuit'].value_counts().nlargest(10).index
    circuit_pitstop_heatmap = circuit_pitstop_heatmap.loc[
        circuit_pitstop_heatmap.index.isin(top_circuits)
    ]

    # Convert circuit IDs to names
    circuit_names = [circuit_id_to_name.get(idx, f"Circuit ID {idx}")
                     for idx in circuit_pitstop_heatmap.index]
    circuit_names = [name[:25] + "..." if len(name) > 25 else name for name in circuit_names]
    circuit_pitstop_heatmap.index = circuit_names

    sns.heatmap(circuit_pitstop_heatmap, cmap='YlGnBu', annot=True, fmt='.2f',
                cbar_kws={'label': 'Top10 Entry Probability'}, linewidths=0.5)
    plt.title('Heatmap: Top10 Entry Probability by Circuit and Pitstop Count', fontsize=15)
    plt.xlabel('Number of Pitstops', fontsize=12)
    plt.ylabel('Circuit', fontsize=12)
    plt.tight_layout()
    plt.show()


def prepare_features():
    ## Prepare features and target variables for model training
    global df_preprocessed, X_train, X_test, y_train, y_test

    print("\n" + "=" * 50)
    print("FEATURE PREPARATION")
    print("=" * 50)

    # Define feature columns
    feature_cols = [
        "Circuit",
        "Laps",
        "TotalPitStops",
        "AvgPitStopTime",
        "FastestStop",
        "Constructor",
        "Season",
        "SlowestStop",
        "FirstPitLap"
        "LastPitLap"
    ]

    # Ensure all features exist
    available_features = [col for col in feature_cols if col in df_preprocessed.columns]
    print(f"Available features: {available_features}")

    X = df_preprocessed[available_features]
    y = df_preprocessed['Top10']

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Feature columns used: {list(X_train.columns)}")


def train_model():
    ## Train the model and perform cross-validation
    global X_train, y_train, model

    print("\n" + "=" * 50)
    print("MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 50)

    # Initialize Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 5-fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Multiple scoring metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=metric)
        cv_results[metric] = scores
        print(f"CV {metric.capitalize()}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Final model training
    print(f"\nTraining final model...")
    model.fit(X_train, y_train)
    print(f"Model training completed!")

    return cv_results


def evaluate_model():
    ## Evaluate model performance
    global model, X_train, X_test, y_test

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, digits=3))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.3f}")

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], color='teal', alpha=0.7)
    plt.title('Feature Importances', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(len(importances)),
               [X_train.columns[i] for i in indices],
               rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def analyze_optimal_pitstops():
    ## Analyze optimal number of pitstops per circuit
    global df_preprocessed, circuit_id_to_name

    print("\n" + "=" * 50)
    print("OPTIMAL PITSTOP ANALYSIS BY CIRCUIT")
    print("=" * 50)

    optimal_pitstops = {}

    for circuit_id in df_preprocessed['Circuit'].unique():
        circuit_data = df_preprocessed[df_preprocessed['Circuit'] == circuit_id]

        if len(circuit_data) < 20:  # Skip circuits with insufficient data
            continue

        pitstop_success = circuit_data.groupby('OriginalPitStops')['Top10'].agg(['count', 'sum'])
        pitstop_success['rate'] = pitstop_success['sum'] / pitstop_success['count']

        # Only consider pitstop counts with at least 5 data points
        valid_pitstops = pitstop_success[pitstop_success['count'] >= 5]

        if not valid_pitstops.empty:
            best_pitstop = valid_pitstops['rate'].idxmax()
            circuit_name = circuit_id_to_name.get(circuit_id, f"Circuit ID {circuit_id}")
            optimal_pitstops[circuit_name] = {
                'best_pitstop': best_pitstop,
                'success_rate': valid_pitstops.loc[best_pitstop, 'rate'],
                'sample_count': valid_pitstops.loc[best_pitstop, 'count']
            }

    # Convert to DataFrame and display
    optimal_df = pd.DataFrame.from_dict(optimal_pitstops, orient='index')
    optimal_df = optimal_df.sort_values(by='success_rate', ascending=False)

    print("Top 10 Circuits - Optimal Pitstop Strategy:")
    print(optimal_df.head(10).to_string())

    # Visualization
    plt.figure(figsize=(16, 8))
    top_optimal = optimal_df.head(10)

    # Shorten circuit names for visualization
    display_names = [name[:25] + "..." if len(name) > 25 else name for name in top_optimal.index]

    bars = plt.bar(range(len(top_optimal)), top_optimal['best_pitstop'], color='skyblue', alpha=0.7)
    plt.xlabel('Circuit', fontsize=12)
    plt.ylabel('Optimal Number of Pitstops', fontsize=12)
    plt.title('Optimal Pitstop Strategy for Top Circuits', fontsize=15)
    plt.xticks(range(len(top_optimal)), display_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add success rate labels
    for i, (idx, row) in enumerate(top_optimal.iterrows()):
        plt.text(i, row['best_pitstop'] + 0.1,
                 f"{row['success_rate']:.2f}\n({row['sample_count']} samples)",
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    return optimal_df


def main():
    ## Main function to run the analysis

    # Step 1: Load data
    load_data()

    # Step 2: Basic EDA
    basic_eda()

    # Step 3: Pitstop visualizations
    visualize_pitstop_analysis()

    # Step 4: Circuit analysis
    circuit_analysis()

    # Step 5: Prepare features
    prepare_features()

    # Step 6: Train model
    cv_results = train_model()

    # Step 7: Evaluate model
    eval_results = evaluate_model()

    # Step 8: Optimal pitstop analysis
    optimal_df = analyze_optimal_pitstops()

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)

    results = {
        'model': model,
        'cv_results': cv_results,
        'eval_results': eval_results,
        'optimal_pitstops': optimal_df,
        'circuit_mapping': circuit_id_to_name
    }

    print(f"Key Results Summary:")
    print(f"- Model ROC-AUC: {results['eval_results']['roc_auc']:.3f}")
    print(f"- Total circuits analyzed: {len(results['optimal_pitstops'])}")
    print(f"- Best performing circuit: {results['optimal_pitstops'].index[0]} "
          f"(Success rate: {results['optimal_pitstops'].iloc[0]['success_rate']:.3f})")

    return results

if __name__ == "__main__":
    results = main()
