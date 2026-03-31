#%%                                                        Random Forest Regressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#%% Pre processing
df= pd.read_csv(r"D:\BITS_Acad\2-2\SOP\rtl_power_modeling_dataset2.csv")
X= df.drop(columns=['static_power_mW', 'dynamic_power_mW', 'total_check', 'cycle_id', 'total_power_mW'])
y= df['total_power_mW']

X_encoded = pd.get_dummies(X, columns=['benchmark', 'block', 'instruction'])

X_train, X_test, y_train, y_test = train_test_split( X_encoded, y, test_size=0.2, random_state=40)
n_features_total = X_encoded.shape[1]
print(n_features_total)


# %%  Baseline model using most common n_estimators and max_features
sqrt_features = int(np.sqrt(n_features_total))
log2_features = int(np.log2(n_features_total))
feature_options = ['sqrt', 'log2', n_features_total] 
#iterate throught the array and see which of the maximum features yields best results
feature_results = []

for max_feat in feature_options:
    rf = RandomForestRegressor(
        n_estimators = 100,        #number of trees in the random forest
        max_features = max_feat,
        bootstrap = True,
        oob_score = True,       #internal validation in the random forest on the out of bag data 
        random_state = 40,
        n_jobs = -1,         # use all the cores of the cpu to process it
    )
    rf.fit(X_train, y_train)
    ypred = rf.predict(X_test)
    test_r2 = r2_score(y_test, ypred)
    oob_r2 = rf.oob_score_
    mae = mean_absolute_error(y_test, ypred)
    feature_results.append({
        'max_features' : max_feat,
        'n_features_used' : (sqrt_features if max_feat == 'sqrt'
                            else log2_features if max_feat == 'log2'
                            else n_features_total),
        'OOB R2' : round(oob_r2,6),
        'Test R2' : round(test_r2,6),
        'Test MAE (mW)' : round(mae,6),
    })
    print(f"[{max_feat}]  OOB R2={oob_r2:.6f}  "f"Test R2={test_r2:.6f}  MAE={mae:.6f} mW")
#[sqrt]  OOB R2=0.996374  Test R2=0.996486  MAE=0.046806 mW
#[log2]  OOB R2=0.996230  Test R2=0.996357  MAE=0.047532 mW
#[27]  OOB R2=0.996665  Test R2=0.996686  MAE=0.045665 mW


# %% optimisation using a permutation of of n_estimators and max_features
sqrt_features = int(np.sqrt(n_features_total))
log2_features = int(np.log2(n_features_total))
max_features_candidates =[
    log2_features,
    sqrt_features,
    n_features_total,
    10,
    15,
    20
]
print(max_features_candidates)
n_estimators_candidates = [50, 100, 150, 200, 300]
print(n_estimators_candidates)


# %% use grid search to find the best model
grid = GridSearchCV(
    estimator  = RandomForestRegressor(
                    bootstrap = True,
                    oob_score = True,
                    random_state = 40,
                ),
    param_grid = {
        'n_estimators' : n_estimators_candidates,
        'max_features' : max_features_candidates,
    },
    cv = 3,
    scoring = 'r2',
    n_jobs = -1,
    verbose = 2,
    refit = True,           # refit best model on full training set
    return_train_score = True,
)
grid.fit(X_train, y_train)
cv_results = pd.DataFrame(grid.cv_results_)
print(cv_results)


# %% results:
cv_results.to_csv('model_results.csv', index=False)
#some folds are not completed due to cv=3
#on an average the final R2 is 99.677% for many of the max_features around 15-20 independent of the number of trees int he forest (n_estimators)
#the regular desicion tree with optimaisation of the hyperparameters performed similarly with an R2 = 99.6584%;


# %% Another way to get the best result is to control the parametrs of the decision tree to use it in the random forest
#using the optimised parameters of the decision tree we found
#    max_depth: 14
#    min_samples_leaf: 1
#    min_samples_split: 75

rf_final = RandomForestRegressor(
    n_estimators = 300,   # found from the above grid search
    max_features = 15,    # found from the above grid search
    max_depth = 14,               
    min_samples_split = 75,                  
    min_samples_leaf = 1,                
    bootstrap = True,
    oob_score = True,
    random_state = 30
)

rf_final.fit(X_train, y_train)
ypred = rf_final.predict(X_test)
test_r2 = r2_score(y_test, ypred)
oob_r2 = rf_final.oob_score_
mae = mean_absolute_error(y_test, ypred)
print(f" OOB R2={oob_r2:.6f}  "f"Test R2={test_r2:.6f}  MAE={mae:.6f} mW")

# OOB R2=0.996990  Test R2=0.996991  MAE=0.043558 mW
# the best possible R2 score we get is 99.6991% from the optimisations

#%% feature importance:
feature_importance = pd.DataFrame({
    'Feature' : X_encoded.columns,
    'Importance' : rf_final.feature_importances_ * 100,
}).sort_values('Importance', ascending=False).reset_index(drop=True)
print(feature_importance)
feature_importance.to_csv('feature_importance_random_forest.csv', index=False)
#the ranking of the feature importance largely remains the same but we see considerably
#more diversification and increase in the importance of hamming distance

# %% 1st tree for visualisation:
first_tree = rf_final.estimators_[0]
plt.figure(figsize=(40, 20))
plot_tree(first_tree, 
          feature_names=list(X_encoded.columns),
          filled=True,
          fontsize=6,
          max_depth=4,
          proportion=True)
plt.title('1st decision tree', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig('random_forest_tree_top4.png')
plt.show()


#%%as a last step we can run a grid search on the tree parameters to find the best tree int he random forest
tree_param_grid = {
    'max_depth': [15, 20, 25],          
    'min_samples_split': [50, 60, 75],     
    'min_samples_leaf': [1, 3, 5]            
}
tree_grid = GridSearchCV(
    estimator = RandomForestRegressor(
        n_estimators = 300,
        max_features = 15,
        bootstrap = True,
        oob_score = True,
        random_state = 40,
    ),
    param_grid = tree_param_grid,
    cv = 2,       
    scoring = 'r2',     
    verbose = 2,      
    refit = True   
)
tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_)
best_tree_model = tree_grid.best_estimator_
ypred = best_tree_model.predict(X_test)
test_r2 = r2_score(y_test, ypred)
oob_r2 = best_tree_model.oob_score_
mae = mean_absolute_error(y_test, ypred)

print(f"OOB R2: {oob_r2:.6f}")
print(f"Test R2: {test_r2:.6f}")
print(f"MAE: {mae:.6f} mW")

#{'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 50}
#OOB R2: 0.997031
#Test R2: 0.997027
#MAE: 0.043328 mW
# with optimisation of the tree in the random forest we get 99.7027%(best till now)