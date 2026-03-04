# Deep Steganography

A deep learning approach to image steganography — hiding a full-color **secret image** inside a **cover image** so that the result is visually indistinguishable from the original cover, while a trained neural network can recover the hidden secret.

Based on the paper: *Hiding Images in Plain Sight: Deep Steganography* (Baluja, 2017).

## Architecture

The system consists of three convolutional neural networks:

1. **Preparation Network** — transforms the secret image into features optimized for hiding
2. **Hiding Network** — embeds the prepared secret into the cover image, producing a container image
3. **Reveal Network** — extracts the secret image from the container

All networks use multi-scale parallel convolutions (3x3, 4x4, 5x5) concatenated together.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

Download and extract [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) into the project root:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

This creates a `tiny-imagenet-200/` directory (~475MB) containing 64x64 RGB images.

### Pre-trained Weights

Pre-trained model weights are included:
- `model_weights_best.hdf5` — best checkpoint weights
- `model.h5` — full saved model

## Usage

Open `deep_steganography.ipynb` and run the cells sequentially. The notebook covers:

1. Loading and preprocessing the Tiny ImageNet dataset
2. Defining the encoder (Prep + Hiding) and decoder (Reveal) networks
3. Training the full model
4. Evaluating reconstruction quality (per-pixel RMSE)
5. Visualizing results: Cover, Secret, Encoded Cover, Decoded Secret, and difference maps

## Results

After training for 1000 epochs:
- Secret image reconstruction error: ~15.8 per pixel (0-255 scale)
- Cover image reconstruction error: ~18.1 per pixel (0-255 scale)

## License

MIT
