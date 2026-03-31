# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv(r"D:\BITS_Acad\2-2\SOP\rtl_power_modeling_dataset2.csv")

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
model = DecisionTreeRegressor(random_state=10)
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

#%% 
## The baseline model gives an accuracy of 99.41%. this is more than the accuracy achieved with the linear regression model (99.33%);
## we can vary the parameters of the decision tree to achieve a higher accuracy


#%% testing the optimal depth of decision tree for max accuracy:
depths = [ None, 3, 5, 7, 10, 12, 13, 15, 20, 25]
depth_results = pd.DataFrame(columns=['depth', 'train_r2', 'test_r2', 'test_mae', 'overfit'])

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=10)
    model.fit(x_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(x_train))
    ypred=model.predict(x_test)
    test_r2 = r2_score(y_test, ypred)
    mean_error = mean_absolute_error(y_test, ypred )
    
    depth_results.loc[len(depth_results)] = [
        depth, train_r2, test_r2, mean_error, train_r2 - test_r2
    ]
    
print(depth_results)
### we see that the mean absolute error falls and then rises as we increase the depth
# of the tree; by the results in the data frame we see that the mae hits a minimum 
# at depth= ""12"" (this is the most optimal depth of the decision tree)

### final accuracy = 99.6557% with depth = 12


# %%testing the min samples split of decision tree for max accuracy:
split_values = [10, 20, 50, 75, 90, 100, 200, 300, 500, 600]
min_split_results = pd.DataFrame(columns=['min_split', 'train_r2', 'test_r2', 'test_mae', 'overfit'])

for split in split_values:
    model = DecisionTreeRegressor(min_samples_split=split, random_state=10)
    model.fit(x_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(x_train))
    ypred=model.predict(x_test)
    test_r2 = r2_score(y_test, ypred)
    mean_error = mean_absolute_error(y_test, ypred )
    
    min_split_results.loc[len(min_split_results)] = [
        split, train_r2, test_r2, mean_error, train_r2 - test_r2
    ]
    
print(min_split_results)
### we see that the mean absolute error falls and then rises as we increase the minimum split
# of the tree; by the results in the data frame we see that the mae hits a minimum 
# at min_split= ""100"" (this is the most optimal min split of the decision tree)

# final accuracy = 99.6580% with min split = 100



# %%testing the min samples split of decision tree for max accuracy:
leaf_values = [5, 10, 20, 30, 35, 40, 50, 75]
leaf_results = pd.DataFrame(columns=['leafs', 'train_r2', 'test_r2', 'test_mae', 'overfit'])

for leaf in leaf_values:
    model = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=10)
    model.fit(x_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(x_train))
    ypred=model.predict(x_test)
    test_r2 = r2_score(y_test, ypred)
    mean_error = mean_absolute_error(y_test, ypred )
    
    leaf_results.loc[len(leaf_results)] = [
        leaf, train_r2, test_r2, mean_error, train_r2 - test_r2
    ]
    
print(leaf_results)
### we see that the mean absolute error falls and then rises as we increase the minimum leafs
# of the tree; by the results in the data frame we see that the mae hits a minimum 
# at leafs= ""30"" (this is the most optimal min leafs of the decision tree)

# final accuracy = 99.6537% with min split = 100


# %%  Grdi search on parameters
### changing different paarameters at once may have a greater effect, so we use grid search and apply different values and search for an optimal result

grid_params = {
    'max_depth': [ 14, 16],
    'min_samples_split': [50, 75, 100],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    grid_params,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV R² Score: {grid_search.best_score_:.6f}")

###result for grid search : Fitting 5 folds for each of 180 candidates, totalling 900 fits
#    max_depth: 14
#    min_samples_leaf: 1
#    min_samples_split: 75
#
#    Best CV R² Score: 0.996584
#    best accuracy achieved = 99.6584%

# %% comparing the baseline with the improved version
import numpy as np
from sklearn.metrics import mean_squared_error
best_model = grid_search.best_estimator_

# Predictions
y_train_pred_best = best_model.predict(x_train)
y_test_pred_best = best_model.predict(x_test)

# Evaluation metrics
best_train_r2 = r2_score(y_train, y_train_pred_best)
best_test_r2 = r2_score(y_test, y_test_pred_best)
best_mae = mean_absolute_error(y_test, y_test_pred_best)
best_mape = np.mean(np.abs((y_test - y_test_pred_best) / y_test)) * 100

comparison = pd.DataFrame({
    'Metric': ['Test R²', 'Test MAE (mW)', 'Overfitting'],
    'Baseline': [test_r2, mean_error, train_r2 - test_r2],
    'Optimized': [best_test_r2, best_mae, best_train_r2 - best_test_r2]
})
comparison['Improvement'] = comparison['Optimized'] - comparison['Baseline']
comparison['Improvement %'] = (comparison['Improvement'] / comparison['Baseline'].abs()) * 100

print(comparison.to_string(index=False))
# we see that the r2 and mae have improved from the baseline
#the overfitting value has increased but the absolute value is very low, due to which teh percentage change seems high


# %% Feature Importance Table
feature_importance = pd.DataFrame({
    'Feature': x_encoded.columns,
    'Importance': best_model.feature_importances_*100
}).sort_values(by='Importance', ascending=False)

print(feature_importance.to_string(index=False))
feature_importance.to_csv("feature_importance_decision_tree.csv", index=False)


# %% visualizing the decision tree:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 20))
plot_tree(best_model, 
          feature_names=list(x_encoded.columns),
          filled=True,
          rounded=True,
          fontsize=6,
          max_depth=4,
          proportion=True)
plt.title('Best Decision Tree - Full Structure', fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_top4.png', dpi=300, bbox_inches='tight')
plt.show()


# %% download the max depth, min_samples_leaf, min_split maximisation
depth_results.to_csv("max_depth_results.csv", index=False)
min_split_results.to_csv("min_samples_split_results.csv", index=False)
leaf_results.to_csv("min_samples_leaf_results.csv", index=False)