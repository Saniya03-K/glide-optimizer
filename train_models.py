import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

print("=" * 60)
print("training models + shap (both direct and residual)")
print("=" * 60)

# load data
df = pd.read_csv('data/synthetic_flight_data.csv')
print(f"\nloaded {len(df)} rows")

features = [
    'altitude_ft', 'weight_lbs', 'wind_kts', 'alpha_deg',
    'wing_area_sqft', 'wing_span_ft', 'aspect_ratio', 'max_gross_weight_lbs'
]
target = 'dist_ground_nm'

# baseline formula for residual learning (prof's suggestion)
df['baseline'] = (df['altitude_ft'] * 0.012) + (df['alpha_deg'] * 1.5) - (df['wind_kts'] * 0.6)
df['residual'] = df[target] - df['baseline']

print(f"\nmean residual: {df['residual'].mean():.2f}")
print(f"std residual: {df['residual'].std():.2f}")

X = df[features]

# split once for both models
X_train, X_test, y_train_direct, y_test_direct = train_test_split(
    X, df[target], test_size=0.2, random_state=42
)
_, _, y_train_res, y_test_res = train_test_split(
    X, df['residual'], test_size=0.2, random_state=42
)

print(f"\ntrain: {len(X_train)}, test: {len(X_test)}")

# ============================================================
# PART 1: DIRECT MODEL (predicts distance directly)
# ============================================================
print("\n" + "-" * 60)
print("PART 1: direct model (no formula)")
print("-" * 60)

# linear regression
print("\nlinear regression...")
t0 = time.time()
lr_direct = LinearRegression()
lr_direct.fit(X_train, y_train_direct)
lr_time = time.time() - t0

pred_lr_d = lr_direct.predict(X_test)
lr_rmse_d = np.sqrt(mean_squared_error(y_test_direct, pred_lr_d))
lr_mae_d = mean_absolute_error(y_test_direct, pred_lr_d)
lr_r2_d = r2_score(y_test_direct, pred_lr_d)

print(f"  time: {lr_time:.2f}s, rmse: {lr_rmse_d:.2f}, mae: {lr_mae_d:.2f}, r2: {lr_r2_d:.4f}")

# random forest
print("\nrandom forest...")
t0 = time.time()
rf_direct = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_direct.fit(X_train, y_train_direct)
rf_time = time.time() - t0

pred_rf_d = rf_direct.predict(X_test)
rf_rmse_d = np.sqrt(mean_squared_error(y_test_direct, pred_rf_d))
rf_mae_d = mean_absolute_error(y_test_direct, pred_rf_d)
rf_r2_d = r2_score(y_test_direct, pred_rf_d)

print(f"  time: {rf_time:.2f}s, rmse: {rf_rmse_d:.2f}, mae: {rf_mae_d:.2f}, r2: {rf_r2_d:.4f}")

# ============================================================
# PART 2: RESIDUAL MODEL (predicts the correction)
# ============================================================
print("\n" + "-" * 60)
print("PART 2: residual model (formula + ML correction)")
print("-" * 60)

# linear regression
print("\nlinear regression...")
t0 = time.time()
lr_res = LinearRegression()
lr_res.fit(X_train, y_train_res)
lr_time_r = time.time() - t0

res_pred_lr = lr_res.predict(X_test)
base_test = df.loc[y_test_direct.index, 'baseline']
final_lr = base_test + res_pred_lr

lr_rmse_r = np.sqrt(mean_squared_error(y_test_direct, final_lr))
lr_mae_r = mean_absolute_error(y_test_direct, final_lr)
lr_r2_r = r2_score(y_test_direct, final_lr)

print(f"  time: {lr_time_r:.2f}s, rmse: {lr_rmse_r:.2f}, mae: {lr_mae_r:.2f}, r2: {lr_r2_r:.4f}")

# random forest
print("\nrandom forest...")
t0 = time.time()
rf_res = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_res.fit(X_train, y_train_res)
rf_time_r = time.time() - t0

res_pred_rf = rf_res.predict(X_test)
final_rf = base_test + res_pred_rf

rf_rmse_r = np.sqrt(mean_squared_error(y_test_direct, final_rf))
rf_mae_r = mean_absolute_error(y_test_direct, final_rf)
rf_r2_r = r2_score(y_test_direct, final_rf)

print(f"  time: {rf_time_r:.2f}s, rmse: {rf_rmse_r:.2f}, mae: {rf_mae_r:.2f}, r2: {rf_r2_r:.4f}")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("comparison:")
print(f"  direct RF:   r2 = {rf_r2_d:.4f}")
print(f"  residual RF: r2 = {rf_r2_r:.4f}")
print("=" * 60)

# feature importance
print("\ndirect RF feature importance:")
imp_d = rf_direct.feature_importances_
order = np.argsort(imp_d)[::-1]
for i in order:
    print(f"  {features[i]}: {imp_d[i]*100:.1f}%")

print("\nresidual RF feature importance:")
imp_r = rf_res.feature_importances_
order = np.argsort(imp_r)[::-1]
for i in order:
    print(f"  {features[i]}: {imp_r[i]*100:.1f}%")

# save metrics
results = pd.DataFrame({
    'Model': [
        'Direct Linear', 'Direct RF',
        'Residual Linear', 'Residual RF'
    ],
    'Test RMSE': [lr_rmse_d, rf_rmse_d, lr_rmse_r, rf_rmse_r],
    'Test MAE': [lr_mae_d, rf_mae_d, lr_mae_r, rf_mae_r],
    'Test R2': [lr_r2_d, rf_r2_d, lr_r2_r, rf_r2_r]
})
results.to_csv('data/ml_results.csv', index=False)

fi = pd.DataFrame({
    'Feature': features,
    'Direct_Importance': rf_direct.feature_importances_,
    'Residual_Importance': rf_res.feature_importances_
}).sort_values('Direct_Importance', ascending=False)
fi.to_csv('data/feature_importance.csv', index=False)

# ============================================================
# SHAP FOR BOTH MODELS
# ============================================================
print("\n" + "-" * 60)
print("running shap on both models...")
print("-" * 60)

# use a sample so it's fast
sample_size = 500
X_sample = X_test.sample(n=sample_size, random_state=42)

# --- direct model SHAP ---
print("\ndirect model shap...")
exp_d = shap.TreeExplainer(rf_direct)
shap_d = exp_d.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_d, X_sample, show=False)
plt.tight_layout()
plt.savefig('data/shap_summary_direct.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved shap_summary_direct.png")

plt.figure()
shap.dependence_plot('wind_kts', shap_d, X_sample, show=False)
plt.tight_layout()
plt.savefig('data/shap_wind_direct.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved shap_wind_direct.png")

# --- residual model SHAP ---
print("\nresidual model shap...")
exp_r = shap.TreeExplainer(rf_res)
shap_r = exp_r.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_r, X_sample, show=False)
plt.tight_layout()
plt.savefig('data/shap_summary_residual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved shap_summary_residual.png")

plt.figure()
shap.dependence_plot('wind_kts', shap_r, X_sample, show=False)
plt.tight_layout()
plt.savefig('data/shap_wind_residual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved shap_wind_residual.png")

print("\ndone.")
print("=" * 60)