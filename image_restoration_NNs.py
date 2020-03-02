import numpy as np
from . import image_restoration_NNs_utils
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from tensorflow.keras.layers import Input, Conv2D, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


GRAYSCALE = 1
NORMALIZATION_FACTOR = 255
ESTIMATION_OF_IMAGE_MEAN = 0.5
TRAINING_SET_PORTION = 0.8


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation - RGB or grayscale.
    :param filename: string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether the
                            output should be grayscale image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with intensities
             (either grayscale or RGB channel intensities) normalized to the range [0,1].
    """
    image = imread(filename)
    if representation == GRAYSCALE:
        image = rgb2gray(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float64)
        image /= NORMALIZATION_FACTOR
    return image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Generates (using python's generator) pairs of image patches comprising an original (clean and sharp) image patch
    with a corrupted version of the same patch.
    These pairs of patches are being extracted randomly from a random choice of an image file from 'filenames'
    and its respective corruption.
    :param filenames: a list of filenames of clean images.
    :param batch_size: the size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: a function receiving a numpy's array representing of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param crop_size: a tuple (height, width) specifying the crop size of the patches to extract.
    :return: yields random tuples of the form (source_batch, target_batch), where each output variable is an
             array of shape (batch_size, height, width, 1), target_batch is made of clean images, and source_batch
             is their respectively randomly corrupted version according to corrupted_func(im).
    """
    filenames_to_images = {}
    source_batch = np.empty((batch_size, crop_size[0], crop_size[1], 1), dtype=np.float64)
    target_batch = np.empty((batch_size, crop_size[0], crop_size[1], 1), dtype=np.float64)
    while True:
        for i in range(batch_size):

            # First, we pick a random clean image from 'filenames' (and cache it if it wasn't cached already).
            filename = filenames[np.random.random_integers(0, len(filenames) - 1)]
            if filename not in filenames_to_images:
                filenames_to_images[filename] = read_image(filename, GRAYSCALE)
            clean_image = filenames_to_images[filename]

            # We sample a larger random crop of size 3 x crop_size, and apply the corruption function on it.
            corruption_row_slice = np.random.random_integers(0, clean_image.shape[0] - 3 * crop_size[0])
            corruption_column_slice = np.random.random_integers(0, clean_image.shape[1] - 3 * crop_size[1])
            clean_cropped_image = clean_image[corruption_row_slice : corruption_row_slice + crop_size[0],
                                              corruption_column_slice : corruption_column_slice + crop_size[1]]
            corrupted_image = corruption_func(clean_cropped_image)

            # We take a random crop of the requested size from both original larger crop and the corrupted copy.
            row_slice = np.random.random_integers(0, corrupted_image.shape[0] - crop_size[0])
            column_slice = np.random.random_integers(0, corrupted_image.shape[1] - crop_size[1])
            clean_patch = (clean_cropped_image[row_slice : row_slice + crop_size[0],
                                               column_slice : column_slice + crop_size[1]]).copy()
            corrupted_patch = corrupted_image[row_slice : row_slice + crop_size[0],
                                              column_slice : column_slice + crop_size[1]]

            # Finally, we subtract the estimated mean from the two patches, and add them to the output batches.
            clean_patch -= ESTIMATION_OF_IMAGE_MEAN
            corrupted_patch -= ESTIMATION_OF_IMAGE_MEAN
            clean_patch = clean_patch.reshape((crop_size[0], crop_size[1], 1))
            corrupted_patch = corrupted_patch.reshape((crop_size[0], crop_size[1], 1))
            target_batch[i] = clean_patch
            source_batch[i] = corrupted_patch
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    Defines a single residual block of a complete 'ResNet' inspired Neural Network model for image restoration.
    :param input_tensor: a symbolic input tensor.
    :param num_channels: the number of channels for each of the input tensor convolutional layers.
    :return: the symbolic output tensor of the layer configuration of a single residual block.
    """
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    output_tensor = Activation('relu')(output_tensor)
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(output_tensor)
    output_tensor = Add()([input_tensor, output_tensor])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Builds a complete 'ResNet' inspired Neural Network model for image restoration
    (with the given number of residual blocks).
    :param height: the height of the input of the model (whose shape is (height, width, 1)).
    :param width: the width of the input of the model (whose shape is (height, width, 1)).
    :param num_channels: the number of channels for each of the convolutional layers in the model (except the
                         very last convolutional layer, which has a single output channel).
    :param num_res_blocks: the number of the residual blocks used in the 'ResNet' inspired Neural Network model.
    :return: a complete 'ResNet' inspired Neural Network model for image restoration.
    """
    input_tensor = Input(shape=(height, width, 1))
    output_tensor = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    output_tensor = Activation('relu')(output_tensor)
    for i in range(num_res_blocks):
        output_tensor = resblock(output_tensor, num_channels)
    output_tensor = Conv2D(1, (3, 3), padding='same')(output_tensor)
    output_tensor = Add()([input_tensor, output_tensor])
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Given a neural network model for image restoration, trains the model over 80% of the images
    given by 'images' (the other 20% are used as validation set in the training process).
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files.
    :param corruption_func: a function receiving a numpy's array representing of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation setto test on after every epoch.
    """
    slice_index = round(TRAINING_SET_PORTION * len(images))
    training_images, validation_images = images[:slice_index], images[slice_index:]
    crop_size = model.input_shape[1:3]
    training_set = load_dataset(training_images, batch_size, corruption_func, crop_size)
    validation_set = load_dataset(validation_images, batch_size, corruption_func, crop_size)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_set, steps_per_epoch, num_epochs, validation_data=validation_set,
                                  validation_steps=(num_valid_samples / batch_size))


def restore_image(corrupted_image, base_model):
    """
    Restores a corrupted image by using a base model of a neural network trained to restore small image patches.
    :param corrupted_image: a grayscale image of shape (height, width) with values in range [0,1] of type
                            float64, that is affected by a corruption generated from the same corruption
                            function encountered during training.
    :param base_model: a neural network trained to restore small patches.
                       The input and output of the network are images in the [-0.5,0.5] range.
    :return: the restored image of the given corrupted image, after it was restored by an augmented
             version of the given neural network.
    """
    height, width = corrupted_image.shape[0], corrupted_image.shape[1]
    corrupted_image_tensor = Input(shape=(height, width, 1))
    small_patches_model = base_model(corrupted_image_tensor)
    new_model = Model(inputs=corrupted_image_tensor, outputs=small_patches_model)
    shifted_corrupted_image = (corrupted_image.copy() - ESTIMATION_OF_IMAGE_MEAN).reshape((1, height, width, 1))
    restored_image = new_model.predict(shifted_corrupted_image)
    restored_image = (restored_image + ESTIMATION_OF_IMAGE_MEAN).reshape((height, width)).astype(np.float64)
    np.clip(restored_image, 0, 1, out=restored_image)
    return restored_image


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Adds a zero-mean random gaussian noise (with a standard deviation of a randomly sampled sigma,
    distributed uniformly between 'min_sigma' and 'max_sigma') to the given image.
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma,
                      representing the maximal variance of the gaussian distribution.
    :return: the corrupted version of the given image, i.e. its sum with a zero-mean random gaussian noise image.
    """
    sigma_of_standard_deviation = np.random.uniform(min_sigma, max_sigma)
    gaussian_noise_image = np.random.normal(scale=sigma_of_standard_deviation, size=image.shape)
    corrupted_image = (image + gaussian_noise_image).astype(np.float64)
    corrupted_image = (np.around(corrupted_image * 255)) / 255
    np.clip(corrupted_image, 0, 1, out=corrupted_image)
    return corrupted_image


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a neural network to denoise image patches of size 24x24.
    :param num_res_blocks: the number of the residual blocks used in the 'ResNet' inspired Neural Network model.
    :param quick_mode: if True (e.g. during presubmission), the model is being trained quicker (by using smaller
                       parameters during the training of the model).
    :return: a trained image-denoising neural network model.
    """
    images_for_denoising = image_restoration_NNs_utils.images_for_denoising()
    denoising_model = build_nn_model(24, 24, 48, num_res_blocks)
    noise_corruption_func = lambda x: add_gaussian_noise(x, 0, 0.2)
    if quick_mode:
        train_model(denoising_model, images_for_denoising, noise_corruption_func, batch_size=10,
                              steps_per_epoch=3, num_epochs=2, num_valid_samples=30)
    else:
        train_model(denoising_model, images_for_denoising, noise_corruption_func, batch_size=100,
                              steps_per_epoch=100, num_epochs=5, num_valid_samples=1000)
    return denoising_model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulates motion blur on the given image using a square kernel of size 'kernel_size', made of a single
    line crossing its center, and the line has the given 'angle' in radians, measured relative to the positive
    horizontal axis.
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0,pi).
    :return: the given image after it was blurred by convolving it with a kernel as described above.
    """
    motion_blur_kernel = image_restoration_NNs_utils.motion_blur_kernel(kernel_size, angle)
    blurred_image = (convolve(image, motion_blur_kernel)).astype(np.float64)
    blurred_image = (np.around(blurred_image * 255)) / 255
    np.clip(blurred_image, 0, 1, out=blurred_image)
    return blurred_image


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulates a random motion blur on the given image using a square random-sized kernel (chosen randomly from the
    given 'list_of_kernel_sizes'), made of a single line crossing its center, and the line has a random angle in
    the range[0, pi}, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0,1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: the given image after it was randomly blurred by convolving it with a kernel as described above.
    """
    random_angle = np.random.uniform(0, np.pi)
    random_kernel_size = np.random.choice(list_of_kernel_sizes)
    randomly_blurred_image = add_motion_blur(image, random_kernel_size, random_angle)
    return randomly_blurred_image


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a neural network to deblur image patches of size 16x16.
    :param num_res_blocks: the number of the residual blocks used in the 'ResNet' inspired Neural Network model.
    :param quick_mode: if True (e.g. during presubmission), the model is being trained quicker (by using smaller
                       parameters during the training of the model).
    :return: a trained image-deblurring neural network model.
    """
    images_for_deblurring = image_restoration_NNs_utils.images_for_deblurring()
    deblurring_model = build_nn_model(16, 16, 32, num_res_blocks)
    blur_corruption_func = lambda x: random_motion_blur(x, list_of_kernel_sizes=[7])
    if quick_mode:
        train_model(deblurring_model, images_for_deblurring, blur_corruption_func, batch_size=10,
                              steps_per_epoch=3, num_epochs=2, num_valid_samples=30)
    else:
        train_model(deblurring_model, images_for_deblurring, blur_corruption_func, batch_size=100,
                              steps_per_epoch=100, num_epochs=10, num_valid_samples=1000)
    return deblurring_model


