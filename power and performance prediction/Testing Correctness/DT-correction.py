import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv(r"D:\BITS_Acad\2-2\SOP\rtl_power_modeling_dataset2.csv")

# 1. Absolute Data Isolation (The 80/20 Physical Split)
# We capture the exact index of the 80% sample to ensure strict separation
file_1_index = df.sample(frac=0.8, random_state=42).index
file_2_index = df.drop(file_1_index).index

# 2. Pre-processing BEFORE splitting (Fixing the Encoding Loophole)
# We drop the leaky target columns first
drop_cols = ['static_power_mW', 'dynamic_power_mW', 'total_check', 'cycle_id', 'total_power_mW']
x_full = df.drop(columns=drop_cols)
y_full = df['total_power_mW']

# Encode the ENTIRE feature set once. This guarantees that if a rare instruction 
# only exists in the 80% split, the 20% split will still have a column for it (filled with 0s),
# preventing the model from crashing during Phase 2.
x_full_encoded = pd.get_dummies(x_full, columns=['benchmark', 'block', 'instruction'])

# Now recreate the physical split with the properly encoded, perfectly aligned data
x = x_full_encoded.loc[file_1_index]
y = y_full.loc[file_1_index]

x1_encoded = x_full_encoded.loc[file_2_index]
y_holdout = y_full.loc[file_2_index]

# --- PHASE 1: Internal Training and Validation ---
print("--- PHASE 1: Internal Training (80% Data) ---")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Using Random Forest Regressor (n_jobs=-1 uses all CPU cores for faster training)
model = RandomForestRegressor(n_estimators=100, random_state=10, n_jobs=-1)
model.fit(x_train, y_train)

ypred = model.predict(x_test)
error_phase_1 = mean_absolute_error(ypred, y_test)
r2_phase_1 = r2_score(y_test, ypred)

print(f"Internal R2 score: {r2_phase_1:.4f}")
print(f"Internal Mean Absolute Error: {error_phase_1:.4f} mW\n")

# --- PHASE 2: Holdout Verification (20% Brand New Data) ---
# Fixing the print bug: We are now explicitly printing the error calculated for the holdout set.
print("--- PHASE 2: Strict Holdout Verification (20% Data) ---")
ypred_holdout = model.predict(x1_encoded)
error_phase_2 = mean_absolute_error(ypred_holdout, y_holdout)
r2_phase_2 = r2_score(y_holdout, ypred_holdout)

print(f"Holdout R2 score: {r2_phase_2:.4f}")
print(f"Holdout Mean Absolute Error: {error_phase_2:.4f} mW\n")

# --- FEATURE IMPORTANCE ---
print("--- FEATURE IMPORTANCE (Top 10) ---")
feature_importance = pd.DataFrame({
    'Feature': x1_encoded.columns,
    'Importance': model.feature_importances_ * 100
}).sort_values(by='Importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))