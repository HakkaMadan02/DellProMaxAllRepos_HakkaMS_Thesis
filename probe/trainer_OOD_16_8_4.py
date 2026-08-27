import os
import pandas as pd
import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax.training import train_state
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from probe_256_16_8_4_1 import MLP256_16_8_4_1 as MLPProbe256_16_8_4_1


def csvETL(csv_path, sample_size=None, seed=0):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if sample_size is not None:
        df = df.iloc[:sample_size]
    X = df[[f'X{i}' for i in range(1, 257)]].values.astype(np.float32)
    y = df['safety_label'].values.astype(np.float32)
    return X, y


def split_data(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        (X[:train_end], y[:train_end]),
        (X[train_end:val_end], y[train_end:val_end]),
        (X[val_end:], y[val_end:])
    )


def shuffle_dataset(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def create_datasets(
    unsafe_csv,
    safe_csv,
    ood_unsafe_csv=None,
    ood_safe_csv=None,
    sample_size=14500,
    seed=0
):
    X_unsafe, y_unsafe = csvETL(unsafe_csv, sample_size, seed)
    X_safe, y_safe = csvETL(safe_csv, sample_size, seed)

    train_u, val_u, test_u = split_data(X_unsafe, y_unsafe)
    train_s, val_s, test_s = split_data(X_safe, y_safe)

    X_train = np.concatenate([train_u[0], train_s[0]], axis=0)
    y_train = np.concatenate([train_u[1], train_s[1]], axis=0)

    X_val = np.concatenate([val_u[0], val_s[0]], axis=0)
    y_val = np.concatenate([val_u[1], val_s[1]], axis=0)

    X_test = np.concatenate([test_u[0], test_s[0]], axis=0)
    y_test = np.concatenate([test_u[1], test_s[1]], axis=0)

    X_train, y_train = shuffle_dataset(X_train, y_train, seed=seed)
    X_val, y_val = shuffle_dataset(X_val, y_val, seed=seed + 1)
    X_test, y_test = shuffle_dataset(X_test, y_test, seed=seed + 2)

    ood_dataset = None

    if ood_unsafe_csv is not None and ood_safe_csv is not None:
        test_size_total = len(y_test)

        ood_unsafe_count = test_size_total // 2
        ood_safe_count = test_size_total - ood_unsafe_count

        X_ood_unsafe, y_ood_unsafe = csvETL(ood_unsafe_csv, sample_size=None, seed=seed + 3)
        X_ood_safe, y_ood_safe = csvETL(ood_safe_csv, sample_size=None, seed=seed + 4)

        if len(X_ood_unsafe) < ood_unsafe_count:
            raise ValueError(
                f"Not enough OOD unsafe samples. Needed {ood_unsafe_count}, found {len(X_ood_unsafe)}."
            )

        if len(X_ood_safe) < ood_safe_count:
            raise ValueError(
                f"Not enough OOD safe samples. Needed {ood_safe_count}, found {len(X_ood_safe)}."
            )

        X_ood = np.concatenate(
            [X_ood_unsafe[:ood_unsafe_count], X_ood_safe[:ood_safe_count]],
            axis=0
        )
        y_ood = np.concatenate(
            [y_ood_unsafe[:ood_unsafe_count], y_ood_safe[:ood_safe_count]],
            axis=0
        )

        X_ood, y_ood = shuffle_dataset(X_ood, y_ood, seed=seed + 5)
        ood_dataset = (X_ood, y_ood)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), ood_dataset


def loss_function(params, apply_fn, batch):
    X, y = batch
    logits = apply_fn(params, X).squeeze()
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss


@jax.jit
def train_step(state, batch):
    grads = jax.grad(loss_function)(state.params, state.apply_fn, batch)
    return state.apply_gradients(grads=grads)


def eval_step(params, apply_fn, X):
    logits = apply_fn(params, X).squeeze()
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.float32)
    return np.array(preds)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    unsafe_csv = '/home/kunalgarg/Desktop/Probed_GCBFPlus/Probed_GCBFplus/GNN_MLP_1_2_3_Datas/OriginalDatas_PreTrained/unsafe_agents_log_MLP4.csv'
    safe_csv = '/home/kunalgarg/Desktop/Probed_GCBFPlus/Probed_GCBFplus/GNN_MLP_1_2_3_Datas/OriginalDatas_PreTrained/safe_agents_log_MLP4.csv'

    ood_unsafe_csv = '/home/kunalgarg/Desktop/Probed_GCBFPlus/Probed_GCBFplus/GNN_MLP_1_2_3_Datas/OOD/unsafe_agents_log_MLP4_OOD.csv'
    ood_safe_csv = '/home/kunalgarg/Desktop/Probed_GCBFPlus/Probed_GCBFplus/GNN_MLP_1_2_3_Datas/OOD/safe_agents_log_MLP4_OOD.csv'

    (X_train, y_train), (X_val, y_val), (X_test, y_test), ood_dataset = create_datasets(
        unsafe_csv=unsafe_csv,
        safe_csv=safe_csv,
        ood_unsafe_csv=ood_unsafe_csv,
        ood_safe_csv=ood_safe_csv,
        sample_size=14500,
        seed=42
    )

    if ood_dataset is None:
        raise ValueError("OOD dataset was not created. Provide both ood_unsafe_csv and ood_safe_csv.")

    X_ood, y_ood = ood_dataset

    model = MLPProbe256_16_8_4_1(features=256)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 256)))
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_epochs = 500
    batch_size = 32
    num_batches = len(X_train) // batch_size

    for epoch in range(num_epochs):
        perm = np.random.permutation(len(X_train))
        for i in range(num_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            state = train_step(state, (batch_X, batch_y))

    val_preds = eval_step(state.params, state.apply_fn, X_val)
    val_acc = accuracy_score(y_val, val_preds)

    test_preds = eval_step(state.params, state.apply_fn, X_test)
    test_metrics = compute_metrics(y_test, test_preds)

    ood_preds = eval_step(state.params, state.apply_fn, X_ood)
    ood_metrics = compute_metrics(y_ood, ood_preds)

    print(f"Validation Accuracy: {val_acc:.4f}")

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")

    print(f"OOD Test Accuracy: {ood_metrics['accuracy']:.4f}")
    print(f"OOD Test Precision: {ood_metrics['precision']:.4f}")
    print(f"OOD Test Recall: {ood_metrics['recall']:.4f}")
    print(f"OOD Test F1 Score: {ood_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()