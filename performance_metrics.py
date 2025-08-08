import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score

# Read CSV
df = pd.read_csv("results_with_accuracy.csv")

# Calculate error metrics
df["error"] = df["true_count"] - df["predicted_count"]
df["absolute_error"] = df["error"].abs()

# Overall Accuracy
total_correct = sum(df["true_count"] == df["predicted_count"])

# Mean Absolute Error (MAE)
mae = mean_absolute_error(df["true_count"], df["predicted_count"])

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(df["true_count"], df["predicted_count"]))

# Precision, Recall, F1-score (treating counts as classification for exact match)
y_true_exact = (df["true_count"] == df["predicted_count"]).astype(int)
y_pred_exact = (df["predicted_count"] == df["true_count"]).astype(int)

precision = precision_score(y_true_exact, y_pred_exact, zero_division=0)
recall = recall_score(y_true_exact, y_pred_exact, zero_division=0)
f1 = f1_score(y_true_exact, y_pred_exact, zero_division=0)

# Save results
df.to_csv("output/results_with_metrics(1).csv", index=False)

# Print
print(df)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print("\nResults saved to output/results_with_metrics.csv")
