import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import argparse

def build_model(input_shape=(64,64,3), num_classes=10):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(64,64),
        batch_size=args.batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(64,64),
        batch_size=args.batch_size
    )
    model = build_model(input_shape=(64,64,3), num_classes=args.num_classes)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model.save(args.model)
    print(f"Model saved to {args.model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/ALL_RICE')
    parser.add_argument('--model', type=str, default='models/rice_model_fast.h5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    main(args)