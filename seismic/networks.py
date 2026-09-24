"""Small configurable sequence networks; training never uses the test block."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NetworkConfig:
    architecture: str = "lstm"
    units: int = 8
    learning_rate: float = 0.001
    batch_size: int = 8
    patience: int = 8

    def validate(self):
        if self.architecture not in {"lstm", "gru", "cnn"}:
            raise ValueError("Architecture must be lstm, gru or cnn.")
        if not 1 <= self.units <= 128 or not 0 < self.learning_rate <= 1:
            raise ValueError("Use 1–128 units and a learning rate in (0, 1].")
        if self.batch_size < 1 or self.patience < 1:
            raise ValueError("Batch size and patience must be positive.")


@dataclass
class TrainingResult:
    model: object
    history: pd.DataFrame
    config: NetworkConfig


def train_network(split, config: NetworkConfig, epochs: int, seed: int) -> TrainingResult:
    """Fit only train windows; EarlyStopping consults validation windows only."""
    import tensorflow as tf

    config.validate()
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = [tf.keras.Input(shape=split.x.shape[1:])]
    if config.architecture == "cnn":
        layers += [tf.keras.layers.Conv1D(config.units, 3, padding="causal", activation="relu"),
                   tf.keras.layers.GlobalAveragePooling1D()]
    else:
        layer = tf.keras.layers.LSTM if config.architecture == "lstm" else tf.keras.layers.GRU
        layers.append(layer(config.units))
    model = tf.keras.Sequential([*layers, tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss="mse")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1

    def dataset(mask):
        return (tf.data.Dataset.from_tensor_slices((split.x[mask], split.y[mask]))
                .batch(config.batch_size).with_options(options))

    fitted = model.fit(
        dataset(split.train_mask), validation_data=dataset(split.validation_mask),
        epochs=epochs, shuffle=False, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True,
        )],
    )
    history = pd.DataFrame(fitted.history)
    history.index = np.arange(1, len(history) + 1)
    history.index.name = "epoch"
    return TrainingResult(model, history, config)


def config_dict(config: NetworkConfig) -> dict:
    return asdict(config)
