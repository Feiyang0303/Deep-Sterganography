#!/usr/bin/env python3
"""
Deep Steganography — Training Script
Hides a secret image inside a cover image using encoder-decoder networks.

Usage:
    python train.py [--epochs N] [--batch_size B] [--lr LR]
                    [--train_per_class N] [--test_images N]
                    [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, concatenate, GaussianNoise
from keras.models import Model
from keras.preprocessing import image
import keras.ops as K


def parse_args():
    p = argparse.ArgumentParser(description="Train Deep Steganography model")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--train_per_class", type=int, default=10,
                    help="Images per class for training")
    p.add_argument("--test_images", type=int, default=500,
                    help="Total test images")
    p.add_argument("--data_dir", type=str, default="./tiny-imagenet-200")
    p.add_argument("--output_dir", type=str, default="./output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_dataset_small(data_dir, num_images_per_class_train=10,
                       num_images_test=500):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    X_train, X_test = [], []

    for c in os.listdir(train_dir):
        c_dir = os.path.join(train_dir, c, "images")
        if not os.path.isdir(c_dir):
            continue
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name in c_imgs[:num_images_per_class_train]:
            img = image.load_img(os.path.join(c_dir, img_name))
            X_train.append(image.img_to_array(img))
    random.shuffle(X_train)

    test_img_dir = os.path.join(test_dir, "images")
    test_imgs = os.listdir(test_img_dir)
    random.shuffle(test_imgs)
    for img_name in test_imgs[:num_images_test]:
        img = image.load_img(os.path.join(test_img_dir, img_name))
        X_test.append(image.img_to_array(img))

    return np.array(X_train), np.array(X_test)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
BETA = 1.0

def rev_loss(s_true, s_pred):
    return BETA * K.sum(K.square(s_true - s_pred))

def full_loss(y_true, y_pred):
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]
    s_loss = rev_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))
    return s_loss + c_loss


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def make_encoder(input_size):
    input_S = Input(shape=input_size)
    input_C = Input(shape=input_size)

    # Preparation Network
    x3 = Conv2D(50, (3, 3), padding="same", activation="relu", name="conv_prep0_3x3")(input_S)
    x4 = Conv2D(10, (4, 4), padding="same", activation="relu", name="conv_prep0_4x4")(input_S)
    x5 = Conv2D(5,  (5, 5), padding="same", activation="relu", name="conv_prep0_5x5")(input_S)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), padding="same", activation="relu", name="conv_prep1_3x3")(x)
    x4 = Conv2D(10, (4, 4), padding="same", activation="relu", name="conv_prep1_4x4")(x)
    x5 = Conv2D(5,  (5, 5), padding="same", activation="relu", name="conv_prep1_5x5")(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    # Hiding Network
    for i in range(5):
        suffix = str(i)
        x3 = Conv2D(50, (3, 3), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_3x3")(x)
        x4 = Conv2D(10, (4, 4), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_4x4")(x)
        x5 = Conv2D(5,  (5, 5), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_5x5")(x)
        x = concatenate([x3, x4, x5])

    output_Cprime = Conv2D(3, (3, 3), padding="same", activation="relu",
                           name="output_C")(x)

    return Model(inputs=[input_S, input_C], outputs=output_Cprime, name="Encoder")


def make_decoder(input_size):
    reveal_input = Input(shape=input_size)
    input_with_noise = GaussianNoise(0.01, name="output_C_noise")(reveal_input)

    x = input_with_noise
    for i in range(5):
        suffix = str(i)
        inp = x if i > 0 else input_with_noise
        x3 = Conv2D(50, (3, 3), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_3x3")(inp if i == 0 else x)
        x4 = Conv2D(10, (4, 4), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_4x4")(inp if i == 0 else x)
        x5 = Conv2D(5,  (5, 5), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_5x5")(inp if i == 0 else x)
        x = concatenate([x3, x4, x5])

    output_Sprime = Conv2D(3, (3, 3), padding="same", activation="tanh",
                           name="output_S")(x)

    return Model(inputs=reveal_input, outputs=output_Sprime, name="Decoder")


def make_model(input_size):
    input_S = Input(shape=input_size)
    input_C = Input(shape=input_size)

    encoder = make_encoder(input_size)

    decoder = make_decoder(input_size)
    decoder.compile(optimizer="adam", loss=rev_loss)
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer="adam", loss=full_loss)

    return encoder, decoder, autoencoder


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(input_S, input_C, args):
    encoder_model, reveal_model, autoencoder_model = make_model(input_S.shape[1:])

    autoencoder_model.optimizer.learning_rate.assign(args.lr)
    reveal_model.optimizer.learning_rate.assign(args.lr)

    m = input_S.shape[0]
    loss_history = []
    lr = args.lr

    for epoch in range(args.epochs):
        np.random.shuffle(input_S)
        np.random.shuffle(input_C)

        ae_losses, rev_losses = [], []

        for idx in range(0, m, args.batch_size):
            batch_S = input_S[idx:min(idx + args.batch_size, m)]
            batch_C = input_C[idx:min(idx + args.batch_size, m)]

            C_prime = encoder_model.predict([batch_S, batch_C], verbose=0)

            ae_loss = autoencoder_model.train_on_batch(
                x=[batch_S, batch_C],
                y=np.concatenate((batch_S, batch_C), axis=3),
            )
            ae_losses.append(ae_loss)

            rev_l = reveal_model.train_on_batch(x=C_prime, y=batch_S)
            rev_losses.append(rev_l)

        lr = lr_schedule(epoch + 1)
        autoencoder_model.optimizer.learning_rate.assign(lr)
        reveal_model.optimizer.learning_rate.assign(lr)

        mean_ae = np.mean(ae_losses)
        mean_rev = np.mean(rev_losses)
        loss_history.append(mean_ae)

        print(f"Epoch {epoch+1:4d}/{args.epochs} | "
              f"AE Loss: {mean_ae:10.2f} | Rev Loss: {mean_rev:10.2f} | lr: {lr}")

    return loss_history, autoencoder_model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def pixel_errors(input_S, input_C, decoded_S, decoded_C):
    see_S = np.sqrt(np.mean(np.square(255 * (input_S - decoded_S))))
    see_C = np.sqrt(np.mean(np.square(255 * (input_C - decoded_C))))
    return see_S, see_C


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Download dataset if needed
    zip_path = os.path.join(os.path.dirname(args.data_dir), "tiny-imagenet-200.zip")
    if not os.path.isdir(args.data_dir):
        print("Downloading Tiny ImageNet...")
        os.system(f"wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O {zip_path}")
        os.system(f"unzip -q {zip_path} -d {os.path.dirname(args.data_dir) or '.'}")

    print("Loading dataset...")
    X_train_orig, X_test_orig = load_dataset_small(
        args.data_dir, args.train_per_class, args.test_images
    )
    X_train = X_train_orig / 255.0
    X_test = X_test_orig / 255.0
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"X_train shape: {X_train.shape}")

    input_S = X_train[: X_train.shape[0] // 2]
    input_C = X_train[X_train.shape[0] // 2:]

    print("Starting training...")
    loss_history, autoencoder_model = train(input_S, input_C, args)

    # Save weights & model
    weights_path = os.path.join(args.output_dir, "model_weights.weights.h5")
    model_path = os.path.join(args.output_dir, "model.keras")
    autoencoder_model.save_weights(weights_path)
    autoencoder_model.save(model_path)
    print(f"Model saved to {args.output_dir}")

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    print("Loss curve saved.")

    # Evaluate
    decoded = autoencoder_model.predict([input_S, input_C])
    decoded_S, decoded_C = decoded[..., 0:3], decoded[..., 3:6]
    S_error, C_error = pixel_errors(input_S, input_C, decoded_S, decoded_C)
    print(f"S error per pixel [0, 255]: {S_error:.4f}")
    print(f"C error per pixel [0, 255]: {C_error:.4f}")


if __name__ == "__main__":
    main()
