from tensorflow.keras import layers
import tensorflow as tf


class AttentionPool(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        # tell Keras this layer understands masks
        self.supports_masking = True

        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def compute_mask(self, inputs, mask=None):
        # This layer reduces the time dimension to a vector, so it produces no mask.
        return None

    def call(self, inputs, mask=None):
        # inputs: [B, T, F], mask: [B, T] or None
        x = self.dense(inputs)            # [B, T, units]
        scores = self.score(x)            # [B, T, 1]

        if mask is not None:
            # Broadcast mask to scores' shape and make masked positions very negative
            mask = tf.cast(mask, scores.dtype)           # [B, T]
            mask = tf.expand_dims(mask, axis=-1)         # [B, T, 1]
            scores = scores + (mask - 1.0) * 1e9         # masked positions -> ~ -inf

        weights = tf.nn.softmax(scores, axis=1)          # [B, T, 1]
        weighted = tf.reduce_sum(weights * inputs, axis=1)  # [B, F]
        return weighted

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.dense.units if hasattr(self.dense, "units") else None})
        return cfg