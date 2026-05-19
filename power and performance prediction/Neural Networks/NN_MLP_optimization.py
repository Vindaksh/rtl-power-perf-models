#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#%% import the baseline model previously done
from NN_MLP_baseline import (
    X_train, X_test, y_train, y_test,
    preprocessor, baseline_pipe, SEED,
    CAT_COLS, NUM_COLS
)

#%% creating a new pipeline
def validate(pipe, label=""):
    pipe.fit(X_train, y_train)
    yp= pipe.predict(X_test)
    r2= r2_score(y_test, yp)
    rmse= float(np.sqrt(mean_squared_error(y_test, yp)))
    mae= float(mean_absolute_error(y_test, yp))
    if label:
        print(f"  {label}  R²={r2:.4f}  RMSE={rmse:.4f} mW  MAE={mae:.4f} mW")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "yp": yp}

def new_pipe(hidden=(100,), activation="relu", **mlp_kw):
    from NN_MLP_baseline import (HybridCategoricalEncoder, EntityEmbeddingLayers, TrainableEmbeddingLayer, CAT_COLS, NUM_COLS, SEED)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    pre = ColumnTransformer([
        ("hybrid_emb",
         HybridCategoricalEncoder(
            stat_dim=2, learn_dim=4, n_splits=5,
            smoothing=10.0, n_epochs=40, lr=0.05,
            batch_size=256, random_state=SEED,
         ),
        CAT_COLS),
        ("scaler", StandardScaler(), NUM_COLS),
    ])

    return Pipeline([
        ("preprocess", pre),
        ("mlp", MLPRegressor(hidden_layer_sizes=hidden, activation=activation,  max_iter=300, random_state=SEED, **mlp_kw)),
    ])

#%% search for the best available activation function 
#%% activation function testing 
test_acts = ["relu", "tanh", "logistic", "identity"]
act_results  = {}
best_act = None
best_r2 = -1.0

for act in test_acts:
    res = validate(new_pipe(hidden=(200, 100), activation=act), f"{act}")
    act_results[act] = res

    if res["R2"] > best_r2:
        best_r2 = res["R2"]
        best_act = act

print(f"Best activation : {best_act}  (R²={best_r2:.4f})")
#the tanh and relu performed similarly but the RMSE of the tanh is slightly better than that tof relu

#%% using a randomised search on all possible hyper parameters, because a grid search would run all combos and take
#a lot of time

PARAM_DIST = {
    "mlp__hidden_layer_sizes": [(100,), (200,), (200, 100), (200, 100, 50), (300, 200, 100)],
    "mlp__activation" : ["relu", "tanh"],
    "mlp__solver" : ["adam", "sgd"],
    "mlp__alpha" : [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "mlp__batch_size" : [32, 64, 128, "auto"],
    "mlp__learning_rate" : ["constant", "adaptive", "invscaling"],
    "mlp__max_iter" : [300, 500],
    "mlp__early_stopping" : [True, False],
}

random_search = RandomizedSearchCV(
    new_pipe(),
    PARAM_DIST,
    n_iter=40, cv=2, scoring="r2",
    n_jobs=-1, random_state=SEED, verbose=2, refit=True,
)

random_search.fit(X_train, y_train)

yp_rs = random_search.predict(X_test)
r2_rs = r2_score(y_test, yp_rs)
rmse_rs = float(np.sqrt(mean_squared_error(y_test, yp_rs)))
mae_rs = float(mean_absolute_error(y_test, yp_rs))

print(f"\nCV R²={random_search.best_score_:.4f}  Test R²={r2_rs:.4f}  RMSE={rmse_rs:.4f} mW")
print("Best params:")
for k, v in random_search.best_params_.items():
    print(f"{k}: {v}")

#results:
'''Fitting 2 folds for each of 40 candidates, totalling 80 fits

CV R²=0.9971  Test R²=0.9971  RMSE=0.0535 mW
  Best params:
  mlp__solver: adam
  mlp__max_iter: 500
  mlp__learning_rate: invscaling
  mlp__hidden_layer_sizes: (200, 100, 50)
  mlp__early_stopping: True
  mlp__batch_size: 64
  mlp__alpha: 0.0005
  mlp__activation: relu'''

# %%now running a grid search on the approximate parameters we found:
#finding the combinations for the grid search by using the neighbours of the data found through randomised search
def get_neighbours(val, options):
    options = sorted(list(set(options)))
    if val not in options: return [val]
    i = options.index(val)
    subset = [options[i]]
    if i > 0: subset.append(options[i-1])
    if i < len(options)-1: subset.append(options[i+1])
    return list(set(subset))

bp = random_search.best_params_

h_layers = [bp["mlp__hidden_layer_sizes"], (200, 100)]
if h_layers[0] == h_layers[1]:
    h_layers = [h_layers[0]]

PARAM_GRID = {
    "mlp__hidden_layer_sizes":h_layers,
    "mlp__activation": [bp["mlp__activation"]],
    "mlp__solver": [bp["mlp__solver"]],
    "mlp__alpha": get_neighbours(bp["mlp__alpha"], [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    "mlp__batch_size": [bp["mlp__batch_size"]],
    "mlp__learning_rate": [bp["mlp__learning_rate"]],
    "mlp__max_iter": [bp.get("mlp__max_iter", 300)],
    "mlp__early_stopping": [bp["mlp__early_stopping"]],
}

grid_search = GridSearchCV(
    new_pipe(),
    PARAM_GRID,
    cv=3, scoring="r2", n_jobs=-1, verbose=1, refit=True,
)

grid_search.fit(X_train, y_train)

yp_gs= grid_search.predict(X_test)
r2_gs= r2_score(y_test, yp_gs)
rmse_gs= float(np.sqrt(mean_squared_error(y_test, yp_gs)))
mae_gs= float(mean_absolute_error(y_test, yp_gs))

print(f"\nCV R²={grid_search.best_score_:.4f}  Test R²={r2_gs:.4f}  RMSE={rmse_gs:.4f} mW")

print("Best params:")
for k, v in grid_search.best_params_.items():
    print(f"{k}: {v}")

#results of the grid search:
'''Grid combinations : 6
Fitting 3 folds for each of 6 candidates, totalling 18 fits

CV R²=0.9971  Test R²=0.9971  RMSE=0.0535 mW
Best params:
mlp__activation: relu
mlp__alpha: 0.0001
mlp__batch_size: 64
mlp__early_stopping: True
mlp__hidden_layer_sizes: (200, 100, 50)
mlp__learning_rate: invscaling
mlp__max_iter: 500
mlp__solver: adam'''

# %%search using halving search for significantly lower time requirement 
hs = HalvingRandomSearchCV(
    new_pipe(),
    PARAM_DIST,
    n_candidates=10, factor=3, cv=3,
    scoring="r2", n_jobs=-1,
    random_state=SEED, verbose=2,
    refit=True, min_resources="exhaust",
)

hs.fit(X_train, y_train)

yp_hs= hs.predict(X_test)
r2_hs = r2_score(y_test, yp_hs)
rmse_hs = float(np.sqrt(mean_squared_error(y_test, yp_hs)))
mae_hs= float(mean_absolute_error(y_test, yp_hs))

print(f"\nCV R²={hs.best_score_:.4f}  Test R²={r2_hs:.4f}  RMSE={rmse_hs:.4f} mW")
print(f"Halving rounds: {hs.n_iterations_}")
print("Best params:")
for k, v in hs.best_params_.items():
    print(f"{k}: {v}")

#results of the halving procedure:
'''CV R²=0.9971  Test R²=0.9971  RMSE=0.0539 mW
Halving rounds: 4
Best params:
mlp__solver: adam
mlp__max_iter: 300
mlp__learning_rate: invscaling
mlp__hidden_layer_sizes: (200,)
mlp__early_stopping: True
mlp__batch_size: auto
mlp__alpha: 0.0005
mlp__activation: relu'''

#%% analysing the faeture importance
#since MLP is black box there is no direct function to display the feature importance, so we 
#use permutation importnace to find the feature importance 
from sklearn.inspection import permutation_importance

best_model = hs.best_estimator_
result = permutation_importance(
    best_model, X_test, y_test, 
    n_repeats=10, random_state=SEED, scoring="r2", n_jobs=-1
)

feature_names = X_test.columns if isinstance(X_test, pd.DataFrame) else (CAT_COLS + NUM_COLS)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Mean': result.importances_mean,
    'Importance_Std': result.importances_std
})

importance_df = importance_df.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)

for index, row in importance_df.iterrows():
    print(f"{row['Feature']}  {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}")

#results of the importance factors:
#these results show the weightage assigned by the algo by every parameter(i.e the drop in r2 when these parameters are removed)
'''
toggle_count  0.5163 ± 0.0020
block  0.4790 ± 0.0020
data_bus_activity  0.0138 ± 0.0001
hamming_dist  0.0046 ± 0.0000
control_activity  0.0006 ± 0.0000
instruction  0.0000 ± 0.0000
benchmark  -0.0000 ± 0.0000
stall  -0.0000 ± 0.0000'''