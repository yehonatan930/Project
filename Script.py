# %% [markdown]
# # Image Colorization Using Autoencoders and Resnet
#
# This notebook is an attempt to colorize black-white images using deep learning.

# %% [code]
# importing general modules
import os
import sys
import random
import warnings
import zipfile
import time
from tqdm import tqdm

# for computing ect.
import numpy as np
import cv2

# for displaying
import matplotlib.pyplot as plt

# for color format and image alterations
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb, rgba2rgb

# keras 2.2.4
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
# tensorflow 1.11.0-rc1
import tensorflow as tf

# filters warnings regarding the skimage module, since it tends to sends some messages while operating.
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
seed = 42
random.seed = seed
np.random.seed = seed


class Color:
    # coloring class I copied online.
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def myprint(message, mode, start_newline=True):
    # gets 2 strings, one as a message to print, the other set the color.
    # based on the 'mode' string (w/e/i), decides the color to print
    # gets another bool to decide whether to start a new line at the end of the print.
    string = ""
    if mode == 'w':  # warning is red
        string = '{1}{0}{2}'.format(message, Color.RED + Color.BOLD, Color.END)
    if mode == 'e':  # entering value is purple
        string = '{1}{0}{2}'.format(message, Color.PURPLE + Color.BOLD, Color.END)
    if mode == 'i':  # info is light blue
        string = '{1}{0}{2}'.format(message, Color.CYAN + Color.BOLD, Color.END)

    if start_newline:
        print(string)
    else:
        print(string, end=" ")


# %% [code]
def path_verify(message, can_be):
    # gets a string of a message, and a list of what the path can lead to.
    # prints a message and inputs a path
    # make sure the path is right and meet the standards.
    # return a path once it is clear it is fine. otherwise, prints a warning and tries again.
    while True:  # doesn't stop until good path can be returned
        myprint(message, 'e', start_newline=False)
        myprint("Can be " + " or ".join(can_be) + ":    ", 'e')
        path = input()

        if 'zip' in can_be and path.endswith('.zip'):
            return path
        if 'dir' in can_be and os.path.isdir(path):
            return path
        if 'file' in can_be and os.path.isfile(path):
            return path

        myprint("Path is not " + "/".join(can_be), 'w')


def int_verify(massage, min_=1, max_=1000000):
    # make sure input is an integer in a certain range
    # gets a string message to print and min and max int values
    # return input once it is clear it is an integer in range. otherwise, prints a warning and tries again.
    while True:  # doesn't stop until a good int can be returned
        myprint(massage, 'e')
        inp = input()

        if not inp.isdigit():  # if it's not a number
            myprint("Input is not an integer.", 'w')
        elif int(inp) < min_:  # if too small, mainly for training quantity
            myprint("Enter a larger number.", 'w')
        elif int(inp) > max_:  # if it's too big, mainly for testing quantity
            myprint("Enter a smaller number.", 'w')
        else:
            return int(inp)


def option_verify(message, options):
    # make sure input is one of certain options
    # gets string message to print and a list of options
    # return the input once it is one of the options in the list
    while True:  # doesn't stop until a good input is given
        myprint(message, 'e')
        inp = input()
        if inp in options:
            return inp
        else:
            myprint("Not an option.", 'w')


def draw_progress_bar(n, total, bar_len=50):
    # draws a progress bar on iteration, like tqdm
    # at some point, notebook's interface doesn't seem to completely handle tqdm, hence this function.
    # gets 3 ints: current number in iteration, the total length of the iteration, and the bar length
    sys.stdout.write("\r")  # return the carriage back to start
    progress = ""
    percent = n / total  # current percent of iteration

    for i in range(bar_len):  # create the bar as string
        if i < int(bar_len * percent):
            progress += "="
        else:
            progress += " "
    s = "[{}] {:.2f}%".format(progress, percent * 100)
    if percent == 1:
        s += "\n"
    sys.stdout.write(s)  # printing the bar
    sys.stdout.flush()
    # needed to make the printing work
    # (without it, the characters will be stored in a buffer rather than printing them immediately)


# %% [code]
def extraction(path):
    # gets a path to file, the path had already been tested at path_verify
    # check if file is zip and if so extract it
    # if zip, return the path to extracted folder. otherwise, return the path function got.
    if path.endswith('.zip'):
        myprint("Path leads to zip direcory", 'i')
        extract_path = path_verify("Enter path of directory to extract into.", ['dir'])
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return extract_path
    return path


def scan_dataset(path):
    # get a path to a directory or file
    # if file, return list containing the path
    # if dir, return paths of all files in this directory and it's sub-directories
    paths = []
    if os.path.isfile(path):
        myprint("Loaded file.", 'i')
        return [path]

    for root, dirs, files in os.walk(path):
        for name in files:
            paths.append(os.path.join(root, name))
    myprint("Loaded {} files from dataset.".format(len(paths)), 'i')
    return paths


def remove_bads(unreadables, wrong_colors, min_):
    # gets 2 lists of paths
    # remove those paths from the IMAGES_PATHS list
    # returns a bool value representing whether the dataset can be accepted.
    # dataset can be accepted if it has at least one batch worth of images.
    global IMAGES_PATHS
    myprint("{} wrong format files or unreadable images.".format(len(unreadables)), 'i')
    time.sleep(1)
    if len(unreadables) != 0:
        yn = option_verify('Print paths? [y/n]    ', ['y', 'n'])
        for path in unreadables:
            IMAGES_PATHS.remove(path)
            if yn == 'y':
                myprint(path, 'i')
        myprint("Paths removed.", 'i')

    myprint("\n{} wrong color format excluded images.".format(len(wrong_colors)), 'i')
    if len(wrong_colors) != 0:
        yn = option_verify('Print paths? [y/n]    ', ['y', 'n'])
        for path in wrong_colors:
            IMAGES_PATHS.remove(path)
            if yn == 'y':
                myprint(path, 'i')
        myprint("Paths removed.", 'i')

    myprint("\n{} images are fine.".format(len(IMAGES_PATHS)), 'i')
    if len(IMAGES_PATHS) < min_:
        myprint("You have too little valid images in here, select another dataset.", 'w')
        return False
    else:
        return True


# %% [code]
def normal_check_data():
    # create the global variables IMAGES_PATHS, CHANNELS
    # input the path to dataset and extract if needed
    # uses scan_dataset to make a list of all paths in dataset
    # goes over all the paths and check that the files are readable images and in the right color format
    # color format is chosen by the one of the first readable image, can't be grayscale
    # any file that doesn't meet those standards is removed from IMAGES_PATHS using remove_bads
    global IMAGES_PATHS, CHANNELS
    while True:  # loop 1 - doesn't stop until a good dataset is found
        data_path = path_verify("Enter path to dataset.", ['zip', 'dir'])  # path to dataset
        data_path = extraction(data_path)  # check if file is zip and if so extract it
        myprint('...', 'i')
        IMAGES_PATHS = scan_dataset(data_path)  # get all paths from data_path (dataset)

        myprint('Checking dataset for incorrect files or images... ', 'i')
        time.sleep(1)  # to avoid printing issues

        unreadables = []  # a list for all paths which don't leads to a readable image.
        wrong_colors = []  # a list for all paths which leads to a wrong color format image
        for n, path in tqdm(enumerate(IMAGES_PATHS), total=len(IMAGES_PATHS), ncols=70):
            # loop 2 - goes over all paths in IMAGES_PATHS
            # tqdm generates the progress bar

            # can replace tqdm:
            # draw_progress_bar(n+1, len(IMAGES_PATHS))

            try:  # make sure image is readable
                img = imread(path)  # image as array
            except (OSError, RuntimeError, ValueError):
                unreadables.append(path)
                continue  # to loop 2

            if n == len(unreadables):  # if it is the first readable image
                if len(img.shape) == 2 or img.shape[2] == 1:  # there are 0/1 channels - grayscale, can't learn from it
                    wrong_colors.append(path)
                else:  # there are 3/4 channels - RGB/RGBA color formats respectively
                    CHANNELS = img.shape[2]  # state how many channels the program will handle

            else:  # any other image
                batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, CHANNELS))  # like an empty batch
                try:  # color format check
                    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    batch[0] = img  # here is supposed to be the fail
                except ValueError:  # image has wrong amount of channels
                    wrong_colors.append(path)

        if remove_bads(unreadables, wrong_colors, BATCH_SIZE):
            # if this is true, then remove_bads has assesed this dataset is ok after removing the bad paths,
            # therefore function had finished it's job and can break from the loop
            break  # from loop 1


# %% [code]
def gray_check_data():
    # create the global variables IMAGES_PATHS, CHANNELS, for the only-prediction route
    # input the path to image/s and extract if needed
    # uses scan_dataset to make a list of all paths in dataset
    # goes over all the paths and check that the files are readable images and in the right color format
    # color format is chosen by the one of the first readable image
    # any file that doesn't meet those standrds is removed from IMAGES_PATHS using remove_bads
    global IMAGES_PATHS, CHANNELS
    while True:  # loop 1 - doesn't stop until a good image source is found
        data_path = path_verify("Enter path for black and white image/s.", ['dir', 'file', 'zip'])
        data_path = extraction(data_path)  # check if file is zip and if so extract it
        myprint('...', 'i')
        IMAGES_PATHS = scan_dataset(data_path)  # get all paths from data_path

        myprint('Checking for incorrect files or images... ', 'i')
        time.sleep(1)  # to avoid printing issues

        unreadables = []  # a list for all paths which don't leads to a readable image.
        wrong_colors = []  # a list for all paths which leads to a wrong color format image
        for n, path in tqdm(enumerate(IMAGES_PATHS), total=len(IMAGES_PATHS), ncols=70):
            # loop 2 - goes over all paths in IMAGES_PATHS
            # tqdm generates the progress bar

            # can replace tqdm:
            # draw_progress_bar(n+1, len(IMAGES_PATHS))

            try:  # make sure image is readable
                img = imread(path)  # image as array
            except (OSError, RuntimeError, ValueError):
                unreadables.append(path)
                continue  # to loop 2

            if n == len(unreadables):  # if it is the first readable image
                if len(img.shape) == 2:  # 0 channels - grayscale
                    CHANNELS = 0
                else:  # there are 1/3/4 channels - Grayscale/RGB/RGBA color formats respectively
                    CHANNELS = img.shape[2]
            else:  # any other image
                if CHANNELS == 0:
                    batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH))  # like an empty batch for 2d images
                else:
                    batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, CHANNELS))  # like an empty batch for 3d images

                try:  # color fomrat check
                    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    batch[0] = img  # here is supposed to be the fail
                except ValueError:  # image has wrong amount of channels
                    wrong_colors.append(path)

        if remove_bads(unreadables, wrong_colors, 1):
            # if this is true, then remove_bads has assesed this dataset is ok after removing the bad paths,
            # therefore function had finished it's job and can break from the loop
            break  # from loop 1


# %% [code]
def normal_split_data():
    # inputs global quantities, then divide IMAGES_PATHS to some other global vars
    # inputs EPOCHS
    global TRAIN_PATHS, VALIDATE_PATHS, TEST_PATHS, EPOCHS
    while True:  # doesn't stop until quantities are ok
        train_quantity = int_verify(
            "\nEnter how many images to take for training proccess, "
            "including validation in 2:8 ratio (hundreds~thousands):    ",
            min_=100)
        # split train and validate in 8:2 ratio
        validate_quantity, train_quantity = int(train_quantity * 0.2), int(train_quantity * 0.8)

        test_quantity = int_verify("Enter how many images to take for testing proccess (singles):    ", max_=BATCH_SIZE)

        if train_quantity + test_quantity + validate_quantity > len(IMAGES_PATHS):
            # if user inputed greated numbers than he/she can afford, it will cause issues,
            # so print a warning and goes back to loop
            myprint("You entered higher values than dataset provides ({}), try again.".format(len(IMAGES_PATHS)), 'w')
        else:
            break

    random.shuffle(IMAGES_PATHS)  # shuffling for randomness
    # let's say train_quantity, validate_quantity, test_quantity = 3400, 600, 10
    TRAIN_PATHS = IMAGES_PATHS[:train_quantity]  # first 3400 paths
    VALIDATE_PATHS = IMAGES_PATHS[train_quantity:validate_quantity + train_quantity]  # the next 600 paths
    TEST_PATHS = IMAGES_PATHS[0 - test_quantity:]  # the last 10 paths

    myprint("Paths randomly assigned for training, validating and testing.", 'i')

    EPOCHS = int_verify(
        "\nEnter how many epochs the training process will take (aprox. 1k images per epoch: GPU-1.5m | CPU-0.5h):    ")


# %% [code]
def gray_split_data():
    # inputs global quantity, then cut IMAGES_PATHS to the quantity
    global IMAGES_PATHS
    if len(IMAGES_PATHS) != 1:  # if there is more than one image
        while True:  # doesn't stop until quantity is ok
            quantity = int_verify("\nEnter how many images to take for predicting:    ")
            if quantity > len(IMAGES_PATHS):
                # if user inputed greated numbers than he/she can afford, it will cause issues,
                # so print a warning and goes back to loop
                myprint("You entered higher values than dataset provides ({}), try again.".format(len(IMAGES_PATHS)),
                        'w')
            else:
                break

        random.shuffle(IMAGES_PATHS)  # shuffling for randomness
        IMAGES_PATHS = IMAGES_PATHS[:quantity]  # cuts IMAGES_PATHS to length equal to quantity
        myprint("Paths randomly assigned.", 'i')

    else:  # if there is only one image
        pass


# %% [markdown]
# # Create the Model

# %% [code]
def create_inception():
    # create the global var for embedding, based on the pretrained InceptionResNetV2 module found in keras nodule
    global INCEPTION
    try:
        # make sure inception hadn't been already created, since it takes time.
        # proved useful during testing in notebook
        INCEPTION  # fails if inception doesn't exist yet, hence create it in the excpet
    except NameError:
        while True:  # doesn't stop until InceptionResNetV2 loaded successfully
            path = path_verify(
                "\nEnter path for inception ResNet weights (came with project, in folder with the default dataset).",
                ['file'])
            # '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5' in kaggle
            myprint('...', 'i')
            # pretrained Inception-ResNet-v2 convolutional neural network.
            # further details: https://ai.googleblog.com/2016/08/improving-inception-and-image.html
            INCEPTION = InceptionResNetV2(weights=None, include_top=True)
            try:
                INCEPTION.load_weights(path)  # load further pretrained weights for the model
                break  # at this point inception was loaded completely, so we can break from loop
            except (ValueError, OSError):
                myprint("Not an inception ResNetv2 weights file.", 'w')

        myprint("Loaded inception ResNet model with exterior weights.", 'i')
        INCEPTION.graph = tf.get_default_graph()
        # a tensorflow computation, represented as a dataflow graph.
        # further details: https://www.tensorflow.org/api_docs/python/tf/Graph


# %% [code]
def colorize():
    # create the autoencoder using CNN, inspired greatly from the internet.
    # return the autoencoder without compiling
    # autoencoder is a combination of an encoder (compressed input) and a decoder (decompression output)
    # The model is a combination of an autoencoder and resnet classifier
    # The best an autoencoder by itself is just shade everything in a brownish tone.
    # The model uses an resnet classifier as an embedding to give the
    # neural network an "idea" of what things should be colored.

    # I used the keras functional API, which is a way to create models that is more flexible than the sequential API.
    # the functional API can handle models with non-linear topology, models
    # with shared layers, and models with multiple inputs or outputs.

    embed_input = Input(shape=(1000,))  # instantiate a keras tensor
    # indicates that the expected input will be batches of 1000-dimensional vectors

    # encoder segment

    # instantiate a keras tensor same as L layer of a lab image in (IMG_WIDTH, IMG_HEIGHT) size
    encoder_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1,))
    # spatial convolutions over the images, output values are 128, 256
    # the height and width of the 2D convolution windows (kernels) vary between 2~4
    # using relu (rectified linear unit) as activation function
    # at the end of encoder, image dims are (32, 32, 256) after 3 maxpoolings
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4, 4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

    # fusion segment
    fusion_output = RepeatVector(32 * 32)(embed_input)  # repeat embed_input 1024 times
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    # fuse encoder and embedding to create the final input
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    # decoder segment
    # spatial convolutions over the images, output values are 2~128
    # the height and width of the 2D convolution windows (kernels) vary between 2~4
    # using relu (rectified linear unit) as activation function,
    # except last tanh (Hyperbolic tangent), since AB layers can be minuses, like the results of tanh
    # at the end of decoder, image dims are (256, 256, 2) after 3 upsamplings - same as AB layers of a lab image
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4, 4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2, 2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    # model gets [L layer batch, resnet model prediction based on BW rgb batch] and return AB layers batch


def model_making(optimizer=SGD, only_load=False):
    # create the global model var for training or testing
    # gets one bool to declare if the function will only load a model
    # the function make the model by creating autoencoder using colorize() or by loading autoencoder from existing file
    # if user wants, saving visualization or showing summary of the model
    global MODEL
    while True:  # doesn't stop until model was created or loaded successfully
        if not only_load:  # if it is not certain what to do
            yn = option_verify("Would you like to load an existing model or create a new one? [load/create]    ",
                               ['load', 'create'])
        if only_load or yn == 'load':  # load model
            path = path_verify("Enter path for model.", ['file'])
            try:
                MODEL = load_model(path)
                break  # in this case the load is successfull
            except ValueError:
                myprint("This model is weights only or incompatible.", 'w')
        else:  # create model
            myprint('...', 'i')
            MODEL = colorize()
            # compiling, after some trying, I found that this optimizer makes the best results
            MODEL.compile(optimizer=optimizer(lr=0.01), loss='mean_squared_error', metrics=["accuracy"])
            myprint('Model created and compiled.', 'i')
            break

    yn = option_verify(
        "Would you like to:\n1. save visualiztion of model as an image.\n2. print summary of model.\n3. do nothing.",
        ['1', '2', '3'])
    if yn == '1':
        path = path_verify("Enter path to direrctory to save in.", ['dir'])
        # converts the model to dot format and save to path
        plot_model(MODEL, to_file=os.path.join(path, "model_visualization.png"), show_shapes=True)
    elif yn == '2':
        # prints a string summary of the network
        MODEL.summary()


# %% [markdown]
# # Data Generator Functions

# %% [code]
def create_inception_embedding(grayscaled_rgb):
    # create embedding for model using inception
    # return the embedding - a prediction based on BW rgb batch
    global INCEPTION
    # resize every pic in grayscaled_rgb and rearranging it in
    # a numpy array in the input shape for resnet: (299, 299, 3)
    grayscaled_rgb_resized = np.array([resize(x, (299, 299, 3), mode='constant') for x in grayscaled_rgb])
    # makes sure grayscaled_rgb_resized is all prepared to enter resnet
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with INCEPTION.graph.as_default():
        # with inception's graph as default graph of the tensorflow session
        embed = INCEPTION.predict(grayscaled_rgb_resized)  # prediction occurs

    return embed


def train_gen():
    # generate training data using data_transformer_generator
    num_of_batchs = len(TRAIN_PATHS) // BATCH_SIZE  # number of full batches in training data
    batch_size = BATCH_SIZE
    paths = TRAIN_PATHS
    while True:  # needed in order to work in keras' functions
        yield next(data_transformer_generator(num_of_batchs, batch_size, paths))


def validate_gen():
    # generate validation data using data_transformer_generator
    num_of_batchs = len(VALIDATE_PATHS) // BATCH_SIZE
    batch_size = BATCH_SIZE
    paths = VALIDATE_PATHS
    while True:  # needed in order to work in keras' functions
        yield next(data_transformer_generator(num_of_batchs, batch_size, paths))


def test_gen(is_sample=False):
    # generate testing data using data_transformer_generator
    num_of_batchs = 1  # only one batch
    batch_size = len(TEST_PATHS)  # this batch is on the test images
    paths = TEST_PATHS
    while True:  # needed in order to work in keras' functions
        yield next(data_transformer_generator(num_of_batchs, batch_size, paths, is_sample))


def gray_gen():
    # generate predicting data for grayscale images using data_transformer_generator
    num_of_batchs = 1  # only one batch
    batch_size = len(IMAGES_PATHS)  # this batch is on the test images
    paths = IMAGES_PATHS
    while True:  # needed in order to work in keras' functions
        yield next(data_transformer_generator(num_of_batchs, batch_size, paths, is_sample=True))


def data_transformer_generator(num_of_batchs,
                               batch_size,
                               paths,
                               is_sample=False):
    # gets the number of batchs to do, the batchs' size, the paths to data,
    # and bool stating whether to yield values espacially for sampling
    # yields the values the model needs to learn or value needed for sampling
    # imports the images from paths into batchs, according to CHANNELS
    # standardizes, creates embedding, creates the AB layers and yield everything

    for n in range(num_of_batchs):  # for each batch, create an empty batch according to CHANNELS
        if CHANNELS == 0:  # grayscale
            batch = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        else:  # rgb / rgba / grayscale
            batch = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        start, end = n * batch_size, (n + 1) * batch_size  # starting and ending points in paths
        for i, path in enumerate(paths[start:end]):  # for each image in batch
            img = imread(path)  # image as numpy array
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            batch[i] = img

        if CHANNELS == 4:  # rgba2rgb standardize automatically
            batch = rgba2rgb(batch)  # turn rgba batch to rgb
        else:
            batch = batch.astype('float64') / 255.  # standardize

        if CHANNELS in [0, 1]:  # grayscale images have one channel or zero
            # in here rgbd_grayscale serves 2 purposes: the lab material and the embedding material
            rgbd_grayscale = gray2rgb(batch)  # BW rgb batch - tensor w 3d arrays w values in range 0~1
            lab = rgb2lab(rgbd_grayscale)  # lab batch of BW images
        else:  # image is not grayscale
            # in here rgbd_grayscale serves 1 purpose: the embedding material
            grayscaled_rgb = rgb2gray(batch)  # grayscale batch - tensor w 2d arrays w values in range 0~1
            rgbd_grayscale = gray2rgb(grayscaled_rgb)  # BW rgb batch - tensor w 3d arrays w values in range 0~1
            lab = rgb2lab(batch)  # lab batch is created from original rgb batch

        gray_rgb_embed = create_inception_embedding(rgbd_grayscale)
        gray_lab = lab[:, :, :, 0]  # BW lab batch - the L layer
        # turning batch to tensor. ex: (20, 256, 256) -> (20, 256, 256, 1)
        gray_lab = gray_lab.reshape(gray_lab.shape + (1,))
        color_lab = lab[:, :, :, 1:] / 128  # standardized color lab batch  - the AB layers
        if is_sample:
            # if a sampling function calls the generator, returns the original
            # batch as rgb, the lab batch, and the color lab prediction batch
            yield batch, lab, MODEL.predict([gray_lab, gray_rgb_embed])
        else:  # normal yielding, specially for model input
            yield [gray_lab, gray_rgb_embed], color_lab


# %% [markdown]
# # Train and Evaluate the Model

# %% [code]
def callbacks_making():
    # create the global var of the callbacks for training.
    global CALLBACKS

    # reduce learning rate when a metric has stopped improving.
    # models often benefit from reducing the learning rate by a factor of 2-10 once learning doesn't imporve.
    # so tis callback states that if no improvement is seen for a 5 epochs, the learning rate will multiply by 0.2
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                patience=5,
                                                verbose=1,
                                                factor=0.2,
                                                min_lr=0.00001)

    yn = option_verify('\nSave best model while runnig? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        # decide on a metric to follow
        monitor = option_verify("By which monitor? [loss / acc / val_loss / val_acc]    ",
                                ['loss', 'acc', 'val_loss', 'val_acc'])

        filepath = path_verify("Enter path of directory to save in.", ['dir'])
        filepath = os.path.join(
            filepath,
            "Colorization_Model-epoch_{0}epoch:03d{1}-{monitor}:{0}{monitor}:.4f{1}.h5"
                .format('{', '}', monitor=monitor))

        # examine the metric every epoch, if best so far, save current version of model to filepath
        checkpoint = ModelCheckpoint(filepath,
                                     save_best_only=True,
                                     monitor=monitor,
                                     mode='auto')
        CALLBACKS = [learning_rate_reduction, checkpoint]
    else:
        CALLBACKS = [learning_rate_reduction]


# %% [code]
def training():
    # creates the global var hist
    # hist contains the details of the learning proccess
    # here the learning actually occurs
    global MODEL, HIST
    myprint("\nBeginning training...", 'i')
    # fit_generator learn from genrators, so it doesn't overload RAM,
    # it does computations in many stages, so the cpu/gpu gets crowded fast
    HIST = MODEL.fit_generator(
        train_gen(),  # using the training generator for basic data
        validation_data=validate_gen(),  # using the validate generator for validating data
        epochs=EPOCHS,
        verbose=1,  # progress bar and data on console are normal
        steps_per_epoch=(len(TRAIN_PATHS) // BATCH_SIZE),
        validation_steps=(len(VALIDATE_PATHS) // BATCH_SIZE),
        callbacks=CALLBACKS)


# %% [code]
def evaluating():
    # evaluate the model and print results
    global MODEL
    print('\n')
    # run on test_gen() which contain one batch, optimal for evaluating
    score = MODEL.evaluate_generator(generator=test_gen(),
                                     verbose=1,
                                     steps=len(TEST_PATHS))
    myprint("loss: {:.4f}, accuracy: {:.4f}".format(score[0], score[1]), 'i')


# %% [code]
def model_saving():
    # saves to model after the training if user wants
    # saves model and weights separately, just in case
    yn = option_verify('\nSave model after training? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        dirpath = path_verify("Enter path save in.", ['dir'])
        filepath = os.path.join(dirpath,
                                "Colorization_END_Model_{}__.h5".format(time.strftime('%d.%m_%H:%M')))
        MODEL.save(filepath)
        filepath = os.path.join(dirpath,
                                "Colorization_END_Weights_{}__.h5".format(time.strftime('%d.%m_%H:%M')))
        MODEL.save_weights(filepath)
        myprint("\nSaved model and weights.", 'i')


# %% [code]
def sampling_graph():
    # sampling accuracy graph and loss graph if the user wants
    yn = option_verify('\nShow metrics graph? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        for key in HIST.history.keys():
            if 'acc' in key:
                plt.plot(HIST.history[key], label=key)
        plt.legend()
        plt.show()
        for key in HIST.history.keys():
            if 'loss' in key:
                plt.plot(HIST.history[key], label=key)
        plt.legend()
        plt.show()


# %% [markdown]
# # Sample the Results

# %% [code]
def sampling_images(grays=False):
    # sampling the results as images, if user wants
    yn = option_verify('\nShow results of model colorization? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        yn = option_verify('\nSave results of model colorization? [y/n]    ', ['y', 'n'])
        if yn == 'y':
            path = path_verify("Enter path to direcory.", ['dir'])
        if not grays:  # if sampling for color images
            images, lab, predicted_color = next(test_gen(
                True))  # all of the test images: normal, in lab format, predicted color by the model - all as tensors
            quantity = len(TEST_PATHS)
        else:  # if sampling for BW images
            images, lab, predicted_color = next(gray_gen())
            quantity = len(IMAGES_PATHS)

        # creating the big figure for all of the frames to be in
        fig = plt.figure(figsize=(10, quantity * 3))  # total size in inches: W, H
        c = 1  # for stating images' place in figure
        for i in range(quantity):
            img = images[i]

            if not grays:  # if not BW then color grayscale apart from original image
                grayscale_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
                grayscale_img[:, :, 0] = lab[i][:, :, 0]  # L layer
                ax = fig.add_subplot(quantity, 3, c)  # adding the frame for the image
                ax.set_title("original grayscale")
                imshow(lab2rgb(grayscale_img))  # showing
                ax.axis("off")

            predicted_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
            predicted_img[:, :, 0] = lab[i][:, :, 0]  # L layers
            predicted_img[:, :, 1:] = predicted_color[i] * 128
            ax = fig.add_subplot(quantity, 3, c + 1)  # adding the frame for the image
            ax.set_title("predicted color")
            imshow(lab2rgb(predicted_img))  # showing
            ax.axis("off")
            if yn == 'y':  # saving predicted image
                cv2.imwrite(os.path.join(path, "img_" + str(i) + ".jpg"), lab2rgb(predicted_img))

            ax = fig.add_subplot(quantity, 3, c + 2)  # adding the frame for the image
            ax.set_title("original color")
            imshow(img)  # showing
            ax.axis("off")

            c += 3

        plt.show()


# %% [code]
myprint(
    "Hello stranger, this is my python based deep learning project "
    "designed to learn how to color black and white images.",
    'i')
ACTION = option_verify(
    "What would you like to do?\n1 - model training\n2 - predicting color of BW images with existing model", ['1', '2'])
# global variables needed to
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 64
"""
# these are created and changed during run:
IMAGES_PATHS        # list of all data paths
TRAIN_PATHS         # list of all train images
VALIDATE_PATHS      # list of all validation images
TEST_PATHS          # list of all test images
CHANNELS            # state how many channels the program will handle
EPOCHS              # number of epochs, how many times fit_generator will go over the dataset
MODEL               # the model
INCEPTION           # the inception_resnet_v2 model
CALLBACKS           # the callbacks for fit_generator
HIST                # information about training process for sampling
"""

if ACTION == '1':
    # normal activation route
    myprint("\nDue to this deep learning project's purpose, any grayscale image will be excluded.", 'i')
    myprint("Any non-image format or image with broken data will also be excluded.", 'i')
    myprint(
        "The program works with ONE color format (RGB/RGBA) per run for all the images you choose, based of your first "
        "image. Any other color format will not be accepted.",
        'i')

    normal_check_data()  # getting the paths to data and making sure their ok
    normal_split_data()  # getting quantities and splitting data
    create_inception()  # create the inception model
    model_making(SGD)  # make the model
    callbacks_making()  # make the callbacks for training
    training()  # training process w validation
    evaluating()  # evaluating process
    model_saving()  # maybe saving model
    sampling_graph()  # sampling graph of model metrics at the end
    sampling_images()  # sampling results

else:
    # 'gray' activation route
    myprint("You chose to enter black and white image/s and recieve a color version.", 'i')
    myprint(
        "The program works with ONE color format (Grayscale/RGB/RGBA) per run for all the images you choose, based of "
        "your first image. Any other color format will not be accepted.",
        'i')

    gray_check_data()  # getting the paths to data and making sure their ok
    gray_split_data()  # getting quantities and splitting data
    create_inception()  # create the inception model
    model_making(only_load=True)  # make the model
    sampling_images(True)  # sampling results

# paths for testing program:
# inception:
# '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
# grayscale:
# '../input/fingerprint-dataset-for-fvc2000-db4-b/dataset_FVC2000_DB4_B/dataset/train_data/'
# good:
# '../input/flickr-image-dataset-30k/compressed/dataset-compressed/'
# some broken images:
# '../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/painting/'
# RGBA:
# '../input/flower-color-images/flower_images/flower_images/'
# '../input/flower-color-images/FlowerColorImages.h5'