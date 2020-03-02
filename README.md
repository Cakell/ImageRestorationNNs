# Image Restoration Neural Networks

A 'ResNet' inspired Neural Network for image restoration (denoising &amp; deblurring images) using the 'Keras' framework.

The implementation of the project consists of three steps:

1. Collect "clean" images, apply simulated random corruptions, and extract small patches.

2. Train a neural network to map corrupted patches to clean patches.

3. Given a corrupted image, use the trained network to restore the complete image by restoring each patch separately,
by applying the “ConvNet Trick” for approximating this process.
