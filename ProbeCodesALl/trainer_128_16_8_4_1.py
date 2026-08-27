import os
import pandas as pd
import jax
import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import trange
from probe_128_16_8_4_1 import MLP128_16_8_4_1 as MLPProbe128_16_8_4_1
from probe_128_16_8_4_1 import MLP64_16_8_4_1 as MLPProbe64_16_8_4_1    
from sklearn.preprocessing import RobustScaler

def csvETL(csv_path, sample_size = 14500, seed=0):
    df = pd.read_csv(csv_path)
    df = df.sample(frac = 1, random_state=seed).reset_index(drop=True)
    df = df.iloc[:sample_size]
    X = df[[f'X{i}' for i in range(1, 129)]].values.astype(np.float32)
    #X = df[[f'X{i}' for i in range(1, 65)]].values.astype(np.float32)
    y = df['safety_label'].values.astype(np.float32)
    return X, y

def print_feature_stats(X, name=""):
    print(f"\n{name} Feature Statistics:")
    print(f" Min: {X.min(axis=0).min():.4f}")
    print(f" Max: {X.max(axis=0).max():.4f}")
    print(f" Mean: {X.mean(axis=0).mean():.4f}")
    print(f" Std: {X.std(axis=0).mean():.4f}")

def split_data(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])

def create_datasets(unsafe_csv, safe_csv, sample_size=14500, seed=0):
    X_unsafe, y_unsafe = csvETL(unsafe_csv, sample_size, seed)
    X_safe, y_safe = csvETL(safe_csv, sample_size, seed)

    train_u, val_u, test_u = split_data(X_unsafe, y_unsafe)
    train_s, val_s, test_s = split_data(X_safe, y_safe)

    #NORMALIZATION BLOCK HM
    '''safe_scalar = RobustScaler()
    unsafe_scalar = RobustScaler()
    X_safe = safe_scalar.fit_transform(X_safe)
    X_unsafe = unsafe_scalar.fit_transform(X_unsafe)
    print_feature_stats(X_safe, "Safe After Scaling")
    print_feature_stats(X_unsafe, "Unsafe After Scaling")'''

    X_train = np.concatenate([train_u[0], train_s[0]], axis=0)
    y_train = np.concatenate([train_u[1], train_s[1]], axis=0)
    X_val = np.concatenate([val_u[0], val_s[0]], axis=0)
    y_val = np.concatenate([val_u[1], val_s[1]], axis=0)
    X_test = np.concatenate([test_u[0], test_s[0]], axis=0)
    y_test = np.concatenate([test_u[1], test_s[1]], axis=0)
    def shuffle(X, y):
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]
    return shuffle(X_train, y_train), shuffle(X_val, y_val), shuffle(X_test, y_test)

def loss_function(params, apply_fn, batch):
    X, y = batch
    logits = apply_fn(params, X)
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss

@jax.jit
def train_step(state, batch):
    grads = jax.grad(loss_function)(state.params, state.apply_fn, batch)
    return state.apply_gradients(grads=grads)

def eval_step(params, apply_fn, batch):
    X, y = batch
    logits = apply_fn(params, X)
    return logits

def main():
    unsafe_csv = 'unsafe_critic_ValueAttn.csv'
    safe_csv = 'safe_critic_ValueAttn_balanced.csv'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_datasets(unsafe_csv, safe_csv, sample_size=14500, seed=42)
    print_feature_stats(X_train, "Train")
    print_feature_stats(X_val, "Val")
    print_feature_stats(X_test, "Test")

    model = MLPProbe128_16_8_4_1(features=128)
    #odel = MLPProbe64_16_8_4_1(features=64)
    rng = jax.random.PRNGKey(0)
    #params = model.init(rng, jnp.ones((1, 128)))
    params = model.init(rng, jnp.ones((1, 128)))
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    num_epochs = 500
    batch_size = 32
    num_batches = len(X_train) // batch_size

    for epoch in range(num_epochs):
        perm = np.random.permutation(len(X_train))
        for i in trange(num_batches, desc=f"Epoch {epoch+1}"):
            idx = perm[i*batch_size:(i+1)*batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            state = train_step(state, (batch_X, batch_y))

        val_logits = eval_step(state.params, state.apply_fn, (X_val, y_val))
        val_preds = (val_logits > 0.5).astype(np.float32)
        val_acc = accuracy_score(y_val, val_preds)
        print(f'Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}')

    test_logits = eval_step(state.params, state.apply_fn, (X_test, y_test))
    test_preds = (test_logits > 0.5).astype(np.float32)
    test_acc = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)

    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')

if __name__ == "__main__":
    main()
