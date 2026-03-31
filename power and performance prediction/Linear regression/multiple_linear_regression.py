# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r"D:\BITS_Acad\2-2\SOP\rtl_power_modeling_dataset2.csv")
print(df.head())

# %% dropping static, dynamic, check power, cycle id
x = df.drop(columns=['static_power_mW', 'dynamic_power_mW', 'total_check', 'cycle_id', 'total_power_mW'])
y=df['total_power_mW']
print(x)

# %%
# using one hot encoding for string based columns
# x_encoded stores boolean values for each type of benchmark..
x_encoded = pd.get_dummies(x, columns=['benchmark', 'block', 'instruction'])
print(x_encoded)

# %% splitting and training
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=4)
print(x_test)
model = LinearRegression()
model.fit(x_train, y_train)

# %%
ypred=model.predict(x_test)
error_in_pred = mean_absolute_error(ypred, y_test)
print("mean error: ")
print(error_in_pred)

# %%
result_db = pd.DataFrame({"total_power_mW": y_test, "predicted_total_power_mW": ypred})
result_db["abs_error_mW"] = (result_db["total_power_mW"] - result_db["predicted_total_power_mW"]).abs()
print(result_db)

r2 = r2_score(y_test, ypred)
print("R2 score:", f"{r2 : .4f}")
print("mean absolute eror:",  f"{result_db["abs_error_mW"].mean():.4f}")


# %% Observing the coefficients assigned by the model
coeff_table = pd.DataFrame({
    'Feature': x_encoded.columns,
    'Weight_in_mW': model.coef_
})

# Adding the baseline intercept
intercept_df = pd.DataFrame({'Feature': ['Baseline (Intercept)'], 'Weight_in_mW': [model.intercept_]})
coeff_table = pd.concat([intercept_df, coeff_table], ignore_index=True)
# Sort by absolute impact to identify the high impact tests
coeff_table['importance'] = coeff_table['Weight_in_mW'].abs()
coeff_table = coeff_table.sort_values(by='importance', ascending=False)

coeff_table = coeff_table.drop('importance', axis=1)
print(coeff_table.to_string(index=False))


# %%
coeff_table.to_csv('coefficients_of_linear_regression.csv', index=False)
result_db.to_csv('linear_regression_results.csv', index=False)

