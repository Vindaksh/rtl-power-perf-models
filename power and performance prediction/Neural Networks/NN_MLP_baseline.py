#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

SEED = 42
np.random.seed(SEED)

#%% Data loading
df = pd.read_csv(r"D:\BITS_Acad\2-2\SOP\rtl_power_modeling_dataset2.csv")

TARGET = "total_power_mW"
GIVEAWAY = {"static_power_mW", "dynamic_power_mW", "total_check", "cycle_id", TARGET}
FEATURE_COLS = [c for c in df.columns if c not in GIVEAWAY]
CAT_COLS = ["benchmark", "block", "instruction"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

print(f"Train : {X_train.shape}   Test : {X_test.shape}")
print(f"Categorical :{CAT_COLS}")
print(f"Numerical : {NUM_COLS}")

#%% ENTITY EMBEDDING ENCODER 
class EntityEmbeddingLayers(BaseEstimator, TransformerMixin):

    def __init__(self, n_splits: int = 5, embed_dim: int = 4, smoothing: float = 10.0, random_state: int = 42):
        self.n_splits = n_splits
        self.embed_dim = embed_dim
        self.smoothing = smoothing
        self.random_state = random_state

    def _make_stats(self, X_df: pd.DataFrame, y_arr: np.ndarray, col: str):
        gm = float(np.mean(y_arr))
        gs = float(np.std(y_arr)) or 1.0
        cd = X_df[col].astype(str)
        stats = {}

        for cat, grp in cd.groupby(cd):
            idx = grp.index
            n = len(idx)
            vals = y_arr[idx]
            mu = float(np.mean(vals))
            sd = float(np.std(vals)) if n > 1 else 0.0
            sm = (n * mu + self.smoothing * gm) / (n + self.smoothing)
            z = (sm - gm) / gs

            stats[cat] = [sm, sd, np.log1p(n), z][: self.embed_dim]
        stats["__UNKNOWN__"] = [gm, gs, 0.0, 0.0][: self.embed_dim]
        return stats
    
    # Swap every text word with its matching 4-number list
    # Stack all the translated columns together into a single matrix
    def _lookup(self, X_df: pd.DataFrame, stats_map: dict) -> np.ndarray:
        parts = []
        for col in self.col_names_:
            cd = X_df[col].astype(str)
            sts = stats_map[col]
            emb = cd.map(lambda v: sts.get(v, sts["__UNKNOWN__"]))
            parts.append(np.vstack(emb.values))
        return np.hstack(parts)
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float) if y is not None else np.zeros(len(X))
        self.col_names_ = X.columns.tolist()
        self.cat_stats_ = {col: self._make_stats(X, y_arr, col) for col in self.col_names_}
        return self

    def fit_transform(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float) if y is not None else np.zeros(len(X))
        self.col_names_ = X.columns.tolist()

        result = np.zeros((len(X), len(self.col_names_) * self.embed_dim))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for _, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_arr[tr_idx]
            X_val = X.iloc[val_idx].reset_index(drop=True)

            fold_stats = {col: self._make_stats(X_tr, y_tr, col) for col in self.col_names_}
            result[val_idx] = self._lookup(X_val, fold_stats)

        self.cat_stats_ = {col: self._make_stats(X, y_arr, col) for col in self.col_names_}
        return result
    
    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        return self._lookup(X, self.cat_stats_)


#%% TRAINABLE EMBEDDING LAYER
class TrainableEmbeddingLayer(BaseEstimator, TransformerMixin):

#few layers can be used to include the hidden relationships between the categories which the stats
#in the previous layer dint account for 

    def __init__(self, embed_dim: int = 4, n_epochs: int = 40, lr: float = 0.05, batch_size: int = 256, random_state: int = 42):
        self.embed_dim = embed_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float) if y is not None else np.zeros(len(X))

        self.col_names_ = X.columns.tolist()
        self.vocab_ = {}
        self.embeddings_ = {}

        for col in self.col_names_:
            cats = X[col].astype(str).values
            unique = sorted(set(cats))
            vocab = {c: i for i, c in enumerate(unique)}
            vocab["__UNKNOWN__"] = len(unique)
            self.vocab_[col] = vocab

            # Create a table filled with random weights to act as starting point and guess the power consumption.
            n_cats = len(unique) + 1
            E = rng.normal(0.0, 0.1, (n_cats, self.embed_dim))  
            w = rng.normal(0.0, 0.1, (self.embed_dim,))         
            b = float(np.mean(y_arr))

            idx = np.array([vocab.get(c, vocab["__UNKNOWN__"]) for c in cats])
            n = len(y_arr)

            for _ in range(self.n_epochs):
                perm = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    bi = perm[start: start + self.batch_size]
                    emb = E[idx[bi]]   

                    # Make a guess for the power using the current random numbers.           
                    pred = emb @ w + b                   
                    err = pred - y_arr[bi]               
                    s = 1.0 / len(bi)

                    # Use calculus to find which direction to nudge our numbers to be less wrong.
                    d_w = (emb.T @ err) * s
                    d_b = err.mean()
                    d_emb = np.outer(err * s, w)          

                    #update the weights and the table
                    w -= self.lr * d_w
                    b -= self.lr * d_b
                    np.add.at(E, idx[bi], -self.lr * d_emb)

            self.embeddings_[col] = E

        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        parts = []
        for col in self.col_names_:
            cats = X[col].astype(str).values
            vocab = self.vocab_[col]
            unk = vocab["__UNKNOWN__"]
            idx = np.array([vocab.get(c, unk) for c in cats])
            parts.append(self.embeddings_[col][idx])
        return np.hstack(parts)
    
#%% HYBRID ENCODER COMBINING THE ENTITY EMBEDDING LAYER AND TRAINABLE EMBEDDING LAYER
class HybridCategoricalEncoder(BaseEstimator, TransformerMixin):

#For every categorical column : the 4 numbers from the entity embedding layer are combined
# with the other 4 from the trainable layer to create a 8 number vector for prediction 

    def __init__(self, stat_dim: int = 4, learn_dim: int = 4,
                n_splits: int = 5,smoothing: float = 10.0,
                n_epochs: int = 40, lr: float = 0.05,
                batch_size: int = 256, random_state: int = 42):
        self.stat_dim = stat_dim
        self.learn_dim = learn_dim
        self.n_splits = n_splits
        self.smoothing= smoothing
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _make_encoders(self):
        return (
            EntityEmbeddingLayers(
                n_splits=self.n_splits, embed_dim=self.stat_dim,
                smoothing=self.smoothing, random_state=self.random_state,
            ),
            TrainableEmbeddingLayer(
                embed_dim=self.learn_dim, n_epochs=self.n_epochs,
                lr=self.lr, batch_size=self.batch_size,
                random_state=self.random_state,
            ),
        )
    
    def fit(self, X, y=None):
        self.oof_enc_, self.learn_enc_ = self._make_encoders()
        self.oof_enc_.fit(X, y)
        self.learn_enc_.fit(X, y)
        return self
    
    def fit_transform(self, X, y=None):
        self.oof_enc_, self.learn_enc_ = self._make_encoders()
        stat_feat = self.oof_enc_.fit_transform(X, y)  
        learn_feat = self.learn_enc_.fit(X, y).transform(X)   
        return np.hstack([stat_feat, learn_feat])
    
    def transform(self, X):
        return np.hstack([
            self.oof_enc_.transform(X),
            self.learn_enc_.transform(X),
        ])
    
# %% PREPROCESSING PIPELINE
preprocessor = ColumnTransformer([
    ("hybrid_emb", HybridCategoricalEncoder(
        stat_dim=4, learn_dim=4, n_splits=5,
        smoothing=10.0, n_epochs=40, lr=0.05,
        batch_size=256, random_state=SEED,
    ), CAT_COLS),
    ("scaler", StandardScaler(), NUM_COLS),
])


#%% BASELINE MLP
baseline_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("mlp", MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=300, random_state=SEED)),
])

baseline_pipe.fit(X_train, y_train)
yp = baseline_pipe.predict(X_test)
r2 = r2_score(y_test, yp)
rmse= float(np.sqrt(mean_squared_error(y_test, yp)))
mae = float(mean_absolute_error(y_test, yp))
print(f"R²={r2:.4f}  RMSE={rmse:.4f} mW  MAE={mae:.4f} mW")