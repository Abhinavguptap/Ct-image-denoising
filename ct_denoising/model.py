"""Residual DnCNN model definition."""

from __future__ import annotations

import tensorflow as tf


def build_dncnn(depth: int = 17, filters: int = 64, channels: int = 1) -> tf.keras.Model:
    if depth < 3:
        raise ValueError("DnCNN depth must be at least 3")
    inputs = tf.keras.Input(shape=(None, None, channels), name="noisy_image")
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name="conv_01")(inputs)
    x = tf.keras.layers.ReLU(name="relu_01")(x)
    for index in range(2, depth):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal", name=f"conv_{index:02d}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{index:02d}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index:02d}")(x)
    residual = tf.keras.layers.Conv2D(channels, 3, padding="same", kernel_initializer="he_normal", name=f"conv_{depth:02d}")(x)
    return tf.keras.Model(inputs=inputs, outputs=residual, name="residual_dncnn")
