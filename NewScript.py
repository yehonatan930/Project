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

# for computing ect.
import numpy as np

# for displaying
import matplotlib.pyplot as plt

# for color format and image alterations
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb

# keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D,  Concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model


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


def iskaggle():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook platform
        else:
            return False  # other type
    except NameError:  # not ipython at all
        return False


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
def find_start_dataset(gray=False):
    # get wheather it is the gray route
    # sets the default dataset's path by that
    # or return an empty string in case user doesn't wnat to use the default dataset or if it wasn't found
    yn = option_verify("\nUse default dataset? [y/n]", ['y', 'n'])
    if yn == 'y':
        other_zippath = os.path.join(os.getcwd(), "Dataset.zip")
        if gray:
            kaggle_path = "../input/flickr-image-dataset-30k/BW/"
            other_dirpath = os.path.join(os.getcwd(), "Dataset", "BW")
        else:
            kaggle_path = "../input/flickr-image-dataset-30k/"
            other_dirpath = os.path.join(os.getcwd(), "Dataset")

        if iskaggle():
            data_path = kaggle_path
        elif os.path.isdir(other_dirpath):
            # not in kaggle and already extracted
            data_path = other_dirpath
        else:
            # not in kaggle and not already extracted
            data_path = extraction(other_zippath)
            if data_path == other_zippath:
                # if there wasn't an exctraction - no file
                myprint("Default dataset wan't found.", 'w')
                data_path = ""
    else:
        data_path = ""

    return data_path


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
def normal_check_data(data_path):
    # create the global variables IMAGES_PATHS, CHANNELS
    # input the path to dataset and extract if needed
    # uses scan_dataset to make a list of all paths in dataset
    # goes over all the paths and check that the files are readable images and in the right color format
    # color format is chosen by the one of the first readable image, can't be grayscale
    # any file that doesn't meet those standards is removed from IMAGES_PATHS using remove_bads
    global IMAGES_PATHS, CHANNELS

    myprint("\nDue to this deep learning project's purpose, any grayscale image will be excluded.", 'i')
    myprint("Any non-image format or image with broken data will also be excluded.", 'i')
    myprint(
        "The program works with ONE color format (RGB/RGBA) per run for all the images you choose, based of your first "
        "image. Any other color format will not be accepted.",
        'i')

    while True:  # loop 1 - doesn't stop until a good dataset is found
        if data_path == "":
            # if we won't use default dataset
            data_path = path_verify("Enter path to dataset.", ['zip', 'dir'])  # path to dataset
            data_path = extraction(data_path)  # check if file is zip and if so extract it

        myprint('...', 'i')
        IMAGES_PATHS = scan_dataset(data_path)  # get all paths from data_path (dataset)

        myprint('Checking dataset for incorrect files or images... ', 'i')
        time.sleep(1)  # to avoid printing issues

        unreadables = []  # a list for all paths which don't leads to a readable image.
        wrong_colors = []  # a list for all paths which leads to a wrong color format image
        for n, path in enumerate(IMAGES_PATHS):
            draw_progress_bar(n + 1, len(IMAGES_PATHS))
            # loop 2 - goes over all paths in IMAGES_PATHS
            # tqdm generates the progress bar

            # can replace tqdm:
            # draw_progress_bar(n+1, len(IMAGES_PATHS))

            try:  # make sure image is readable
                img = imread(path)  # image as array
            except (OSError, RuntimeError, ValueError):
                unreadables.append(path)
                continue  # to loop 2

            if n == len(unreadables)+len(wrong_colors):  # if it is the first readable image
                if len(img.shape) == 2 or img.shape[2] == 1:  # there are 0/1 channels - grayscale, can't learn from it
                    wrong_colors.append(path)
                else:  # first good colored - there are 3/4 channels - RGB/RGBA color formats respectively
                    CHANNELS = img.shape[2]  # state how many channels the program will handle
                    batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, CHANNELS))  # like an empty batch

            else:  # any other image
                try:  # color format check
                    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    batch[0] = img  # here is supposed to be the fail
                except ValueError:  # image has wrong amount of channels
                    wrong_colors.append(path)

        if remove_bads(unreadables, wrong_colors, BATCH_SIZE):
            # if this is true, then remove_bads has assesed this dataset is ok after removing the bad paths,
            # therefore function had finished it's job and can break from the loop
            break  # from loop 1
        else:
            data_path = ""  # activating the part in start of loop 1


# %% [code]
def gray_check_data(data_path):
    # create the global variables IMAGES_PATHS, CHANNELS, for the only-prediction route
    # input the path to image/s and extract if needed
    # uses scan_dataset to make a list of all paths in dataset
    # goes over all the paths and check that the files are readable images and in the right color format
    # color format is chosen by the one of the first readable image
    # any file that doesn't meet those standrds is removed from IMAGES_PATHS using remove_bads
    global IMAGES_PATHS, CHANNELS

    myprint("You chose to enter black and white image/s and recieve a color version.", 'i')
    myprint(
        "The program works with ONE color format (Grayscale/RGB/RGBA) per run for all the images you choose, based of "
        "your first image. Any other color format will not be accepted.",
        'i')

    while True:  # loop 1 - doesn't stop until a good image source is found
        if data_path == "":
            # if we won't use default dataset
            data_path = path_verify("Enter path for black and white image/s.", ['zip', 'dir', 'file'])
            data_path = extraction(data_path)  # check if file is zip and if so extract it

        myprint('...', 'i')
        IMAGES_PATHS = scan_dataset(data_path)  # get all paths from data_path

        myprint('Checking for incorrect files or images... ', 'i')
        time.sleep(1)  # to avoid printing issues

        unreadables = []  # a list for all paths which don't leads to a readable image.
        wrong_colors = []  # a list for all paths which leads to a wrong color format image
        for n, path in enumerate(IMAGES_PATHS):
            draw_progress_bar(n + 1, len(IMAGES_PATHS))
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
                    batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH))  # like an empty batch for 2d images
                else:  # there are 1/3/4 channels - Grayscale/RGB/RGBA color formats respectively
                    CHANNELS = img.shape[2]
                if CHANNELS == 0:
                    batch = np.zeros((2, IMG_HEIGHT, IMG_WIDTH, CHANNELS))  # like an empty batch for 3d images                    
            
            else:  # any other image
                try:  # color fomrat check
                    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                    batch[0] = img  # here is supposed to be the fail
                except ValueError:  # image has wrong amount of channels
                    wrong_colors.append(path)

        if remove_bads(unreadables, wrong_colors, 1):
            # if this is true, then remove_bads has assesed this dataset is ok after removing the bad paths,
            # therefore function had finished it's job and can break from the loop
            break  # from loop 1
        else:
            data_path = ""  # activating the part in start of loop 1


# %% [code]
def normal_split_data():
    # inputs global quantities, then divide IMAGES_PATHS to some other global vars
    # inputs EPOCHS
    global TRAIN_PATHS, VALIDATE_PATHS, TEST_PATHS, EPOCHS
    while True:  # doesn't stop until quantities are ok
        train_quantity = int_verify(
            "\nEnter how many images to take for training proccess, "
            "including validation in 2:8 ratio (hundreds~thousands):    ",
            min_=BATCH_SIZE*5)
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
    global TEST_PATHS
    if len(IMAGES_PATHS) != 1:  # if there are more than one image
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
        TEST_PATHS = IMAGES_PATHS[:quantity]  # cuts IMAGES_PATHS to length equal to quantity
        myprint("Paths randomly assigned.", 'i')

    else:  # if there is only one image
        pass


# %% [markdown]
# # Create the Model

# %% [code]
def colorize():
    # create the autoencoder using CNN, inspired greatly from the internet.
    # return the autoencoder without compiling
    # autoencoder is a combination of an encoder (compressed input) and a decoder (decompression output)
    # The input will be a grayscale image and output will be a RGB image.
    # During encoding phase, used convolution layers with LeakyReLU activation.
    # Increased the number of filters as these layers are added.
    # For decoding steps, used UpSampling layer followed by convolution layer.
    # The basic idea is that the encoding phase takes an input and compresses the input by passing it through
    # different convolutional layers. Now using this compressed representation of the input,
    # the decoding phase produces the final output. During decoding phase, transposed convolution layers or upsampling
    # layers are typically used. Because the decoding phase only sees the compressed representation of original input,
    # it might miss important features of image that were lost during the encoding phase.
    # That’s where the skip-connections come to the rescue.
    # They provide the output of each step in encoding phase to the corresponding step in decoding
    # phase so that the decoder can utilize these information as well.
    def conv2d(layer_input, filters):
        # Layers used during downsampling
        d = Conv2D(filters, kernel_size=4, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, from_encode, filters):
        # Layers used during upsampling
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=4, strides=1, padding='same', activation='relu')(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, from_encode])
        return u

    gf = 32  # number of filters
    # Image input
    d0 = Input(shape=(256, 256, 1))

    # Downsampling
    d1 = conv2d(d0, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='sigmoid')(u7)

    return Model(d0, output_img)


def model_making(only_load=False):
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
            path = path_verify("\nEnter path for model.", ['file'])
            try:
                MODEL = load_model(path)
                break  # in this case the load is successfull
            except ValueError:
                myprint("This model is weights only or incompatible.", 'w')
        else:  # create model
            myprint('...', 'i')
            MODEL = colorize()
            # compiling, after some trying, I found that this optimizer makes the best results
            MODEL.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            myprint('Model created and compiled.', 'i')
            break

    yn = option_verify(
        "Would you like to:\n1. save visualiztion of model as an image.\n2. print summary of model.\n3. do nothing.",
        ['1', '2', '3'])
    if yn == '1':
        path = os.path.join(OUTPUT, "models", "Colorization_Model-visualization.png")
        # converts the model to dot format and save to path
        plot_model(MODEL, to_file=path, show_shapes=True)
        myprint("Saved visualiztion at:  {}".format(path), 'i')
    elif yn == '2':
        # prints a string summary of the network
        MODEL.summary()


# %% [markdown]
# # Data Generator Functions

# %% [code]
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


def test_gen():
    # generate testing data using data_transformer_generator
    num_of_batchs = 1  # only one batch
    batch_size = len(TEST_PATHS)  # this batch is on the test images
    paths = TEST_PATHS
    while True:  # needed in order to work in keras' functions
        yield next(data_transformer_generator(num_of_batchs, batch_size, paths))


def data_transformer_generator(num_of_batchs,
                               batch_size,
                               paths):
    # gets the number of batchs to do, the batchs' size, the paths to data,
    # yields the values the model needs to learn:
    # imports the images from paths into batchs, according to CHANNELS
    # standardizes, creates embedding, creates the AB layers and yield everything
    for n in range(num_of_batchs):  # for each batch, create an empty batch according to CHANNELS
        if CHANNELS == 0:
            # if grayscale w/ 2d
            batch = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        else:
            # RGB/RGBA/grayscale 3d
            batch = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)

        start, end = n * batch_size, (n + 1) * batch_size  # starting and ending points in paths
        for i, path in enumerate(paths[start:end]):  # for each image in batch
            img = imread(path)  # image as numpy array
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            batch[i] = img

        # standardization:
        if CHANNELS == 4:  # rgba2rgb standardize automatically
            batch = rgba2rgb(batch)  # turn rgba batch to rgb
        elif CHANNELS == 3 or (CHANNELS in [0, 1] and batch.max() > 1):
            # rgb OR grayscale w/ 0-255 values, not usual but possible
            batch = batch.astype('float64') / 255.  # standardize

        gray_batch = rgb2gray(batch)
        # turning batch to tensor. ex: (20, 256, 256) -> (20, 256, 256, 1)
        gray_batch = gray_batch.reshape(gray_batch.shape + (1,))

        yield gray_batch, batch


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

        filepath = os.path.join(
            OUTPUT, "models",
            "Colorization_Model-epoch_{0}epoch:03d{1}-{monitor}_{0}{monitor}:.4f{1}_.h5"
            .format('{', '}', monitor=monitor))
        # end up looking like: Colorization_Model-epoch_{epoch:03d}-{monitor}_{{monitor}:.4f}_.h5
        # when saved: Colorization_Model-epoch_015-loss_0.0125_.h5

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
    # deletes the previous
    dirpath = os.path.join(OUTPUT, "models")
    lst = os.listdir(dirpath)
    try:
        # if it is kaggle, then these are there too
        lst.remove("__notebook_source__.ipynb")
    except ValueError:
        pass

    try:
        lst.remove(".ipynb_checkpoints")
    except ValueError:
        pass

    try:
        # if visualization saved
        lst.remove("Colorization_Model-visualization.png")
    except ValueError:
        pass

    for f in lst[:-2]:
        os.remove(os.path.join(dirpath, f))

    yn = option_verify('\nSave model after training? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        filepath = os.path.join(dirpath,
                                "Colorization_END_Model_{}_.h5".format(time.strftime('%d.%m_%H.%M')))
        MODEL.save(filepath)


# %% [code]
def sampling_graphs():
    # sampling accuracy graph and loss graph if the user wants
    yn = option_verify('\nShow metrics graph? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        yn = option_verify('Save metrics graph? [y/n]    ', ['y', 'n'])
        for key in HIST.history.keys():
            if 'acc' in key:
                plt.plot(HIST.history[key], label=key)
        plt.legend()
        if yn == 'y':
            plt.savefig(os.path.join(OUTPUT, 'results', 'Accuracy_graph.png'))
        plt.show()
        for key in HIST.history.keys():
            if 'loss' in key:
                plt.plot(HIST.history[key], label=key)
        plt.legend()
        if yn == 'y':
            plt.savefig(os.path.join(OUTPUT, 'results', 'Loss_graph.png'))
        plt.show()


# %% [markdown]
# # Sample the Results

# %% [code]
a, b, c = ['1', '2', '3']
print(a)


# %% [code]
def sampling_images(grays=False):
    yn = option_verify('\nShow results of model colorization? [y/n]    ', ['y', 'n'])
    if yn == 'y':
        yn = option_verify('Save results of model colorization? [y/n]    ', ['y', 'n'])
        bw, original_color = next(test_gen())
        predicted_color = MODEL.predict_on_batch(bw)
        # plot input, actual and predictions side by side
        if not grays:
            fig, axes = plt.subplots(len(bw), 3, figsize=(10, 40))
        else:
            fig, axes = plt.subplots(len(bw), 2, figsize=(10, 40))
        for i in range(0, len(bw)):

            axes[i][0].imshow(np.squeeze(bw[i], axis=-1), cmap='gray')
            axes[i][0].set_title('Input B/W')

            axes[i][1].imshow(predicted_color[i])
            axes[i][1].set_title('Predicted Color')
            if not grays:
                axes[i][2].imshow(original_color[i])
                axes[i][2].set_title('Original Color')
            if yn == 'y':
                path = os.path.join(OUTPUT, "results")
                imsave(os.path.join(path, "img_" + str(i) + ".jpg"), predicted_color[i])
        for ax in axes.flat:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.tight_layout()


# %% [markdown]
# # RUN

# %% [code]
# start standard run
myprint(
    "Hello stranger, this is my python based deep learning project "
    "designed to learn how to color black and white images.",
    'i')

# global variables needed
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
CALLBACKS           # the callbacks for fit_generator
HIST                # information about training process for sampling
"""

if iskaggle():
    # kaggle default output folder
    OUTPUT = "/kaggle/working"
else:
    # make an output folder
    OUTPUT = os.path.join(os.getcwd(), "OUTPUT")
    os.mkdir(OUTPUT)
try:
    os.mkdir(os.path.join(OUTPUT, "results"))
    os.mkdir(os.path.join(OUTPUT, "models"))
except FileExistsError:
    pass

# starting from choice
action = option_verify(
    "What would you like to do?\n1 - model training\n2 - predicting color of BW images with existing model", ['1', '2'])

if action == '1':
    # normal activation route
    normal_check_data(find_start_dataset(gray=False))  # getting the paths to data and making sure their ok
    normal_split_data()  # getting quantities and splitting data"""
    model_making()  # make the model
    callbacks_making()  # make the callbacks for training
    training()  # training process w validation
    evaluating()  # evaluating process
    model_saving()  # maybe saving model
    sampling_graphs()  # sampling graphs of model metrics
    sampling_images()  # sampling results

else:
    # 'gray' activation route

    gray_check_data(find_start_dataset(gray=True))  # getting the paths to data and making sure their ok
    gray_split_data()  # getting quantities and splitting data
    model_making(only_load=True)  # make the model
    sampling_images(grays=True)  # sampling results

# paths for testing program:
# grayscale:
# '../input/fingerprint-dataset-for-fvc2000-db4-b/dataset_FVC2000_DB4_B/dataset/train_data/'
# good:
# '../input/flickr-image-dataset-30k/compressed/dataset-compressed/'
# some broken images:
# '../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/painting/'
# RGBA:
# '../input/flower-color-images/flower_images/flower_images/'
# '../input/flower-color-images/FlowerColorImages.h5'
