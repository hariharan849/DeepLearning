import pathlib, cv2, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Optional, Literal, Generator, Tuple
from matplotlib.patches import Wedge
from matplotlib.patheffects import withStroke
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import albumentations as A

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    import splitfolders
except:
    ! pip install split-folders
    import splitfolders

from tqdm.keras import TqdmCallback
#keras
import tensorflow as tf
from keras import layers, Model, optimizers, applications
from keras.models import load_model, Sequential
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback


AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataframe_from_image_paths(image_folder_path: str) -> pd.DataFrame:
    """ Get images, labels and converts to dataframe

        Args:
            image_folder_path(str): Folders containing images
    """
    images = list(pathlib.Path(image_folder_path).glob("*/*"))
    labels = [os.path.split(os.path.dirname(img_path))[-1] for img_path in images]
    dataframe = pd.DataFrame(zip(images, labels), columns=["image_path", "labels"])
    dataframe["image_path"] = dataframe["image_path"].astype('str')
    return dataframe

def split_images_from_dataframe(dataframe: pd.DataFrame, val_split_size: int=0.3, test_split_size: int=0.1, split_test: bool=False)->tuple[list, list, list]:
    """ Splits dataframe into train, validation and test(optional)

        Args:
            dataframe(pd.Dataframe): Dataset Dataframe to split
            val_split_size(int): Validation split size
            test_split_size(int): Test split size
            split_test(bool): Flag to split validation further to test
    """
    train, val = train_test_split(dataframe, test_size=val_split_size, stratify=dataframe["Labels"])
    test = None
    if split_test:
        val, test = train_test_split(val, test_size=test_split_size)
    return train, val, test

def plot_class_distribution_in_pie_chart(dataframe: pd.DataFrame):
    """ Plots dataset label distribution in pie chart

        Args:
            dataframe(pd.Dataframe): Dataset Dataframe to check class distribution
    """
    # Assuming your data is in a pandas DataFrame called 'train_df'
    animals_counts = dataframe['labels'].value_counts()

    fig, ax = plt.subplots()
    wedges, texts, _ = ax.pie(
        animals_counts.values.astype("float"), startangle=90,
        autopct='%1.1f%%', wedgeprops=dict(width=0.3, edgecolor='black')
    )

    # Add glow effect to each wedge
    for wedge in wedges:
        wedge.set_path_effects([withStroke(linewidth=6, foreground='cyan', alpha=0.4)])

    # Customize chart labels
    plt.legend(dataframe.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, facecolor='#222222')

    # Dark background for the cyberpunk look
    fig.patch.set_facecolor('#2c2c2c')
    ax.set_facecolor('#2c2c2c')

    # Title
    plt.title("Pie Chart", color="white", fontsize=16)

    plt.show()

def plot_class_distribution_in_count_chart(dataframe: pd.DataFrame):
    """ Plots dataset label distribution in bar chart

        Args:
            dataframe(pd.Dataframe): Dataset Dataframe to check class distribution
    """
    #count Plot

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(dataframe, x="labels", palette='pastel')

    # Annotate the count on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom')

    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_random_images_per_class(dataframe: pd.DataFrame, no_images:int=2):
    """ Plots images per label distribution

        Args:
            dataframe(pd.Dataframe): Dataset Dataframe to check class distribution
    """
    class_names = dataframe["labels"].unique()
    random_images = dataframe.sample(frac=1).sort_values(by="labels").groupby('labels').head(no_images)

    count = 0
    num_classes = len(class_names)

    plt.figure(figsize=(12, num_classes * 4))

    for index, (image_path, class_name) in random_images.iterrows():
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        count += 1
        plt.subplot(num_classes, 2, count)
        plt.imshow(image)
        plt.axis('off')
        plt.title(class_name)

def plot_percentage_split(train_dataset: np.array, val_dataset: np.array, test_dataset: Optional[np.array]=None):
    """ Plots dataset split distribution in pie chart

        Args:
            train_dataset(np.array): Train dataset splitted
            val_dataset(np.array): Validation dataset splitted
        KwArgs:
            test_dataset(np.array): Test dataset splitted
    """
    train_size = len(train_dataset)
    validation_size = len(val_dataset)
    test_size = len(test_dataset or [])
    # Dataset sizes
    sizes = [train_size, validation_size]
    labels = ['Train', 'Validation']
    colors = ['#66c2a5', '#fc8d62']

    if test_dataset:
        sizes.append(test_size)
        labels.append("Test")
        colors.append("#0000ff")


    def autopct_format(value):
        """Formats the autopct value to display the percentage and count."""
        total = sum(sizes)
        percentage = f'{value:.1f}%'
        count = int(value * total / 100)
        return f'{percentage}\n{count}'

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct=autopct_format, startangle=140)
    plt.title('Dataset Split Distribution', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()

def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def get_img_extension(image_folder_path:str)->list:
    """ Returns list of image extensions from image paths folder

        Args:
            image_folder_path(str): Image path directory
    """
    images = list(pathlib.Path(image_folder_path).glob("*/*"))
    return [os.path.splitext(img_path)[-1] for img_path in images]

def get_image_dataset_from_jpg(
        image_folder_path:str, img_height:int, img_width:int, output_folder:Optional[str]="",
        validation_split:Optional[float]=0.2, batch_size:Optional[int]=32
    )->tuple:
    """ Creates train and validation dataset from jpg image folder hierarchy

        Args:
            image_folder_path(str) : Folder containing images for training and validation
            img_height(int)        : Height of the image
            img_width(int)         : Width of the image
        KwArgs:
            output_folder(str)     : Output folder to create splitted images directory
            validation_split(float): Split ratio for validation
            batch_size(int)        : Batch size
    """
    train_split = 1.0-validation_split-0.1
    output_folder = output_folder or os.path.dirname(image_folder_path)

    train_dir, val_dir, test_dir = splitfolders.ratio(
        image_folder_path, output=output_folder, seed=1337, ratio = (train_split, validation_split, 0.1)
    )
    train_ds = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        shuffle=True,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        shuffle=False,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        shuffle=False,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def get_image_dataset_from_multiple_ext(
        dataframe:pd.DataFrame, img_height:int, img_width:int, output_folder:Optional[str]="",
        validation_split:Optional[float]=0.2, batch_size:Optional[int]=32
    )->tuple:
    """ Creates train and validation dataset from jpg image folder hierarchy

        Args:
            dataframe(pd.DataFrame): Dataframe containing images for training and validation
            img_height(int)        : Height of the image
            img_width(int)         : Width of the image
        KwArgs:
            output_folder(str)     : Output folder to create splitted images directory
            validation_split(float): Split ratio for validation
            batch_size(int)        : Batch size
    """
    train, val = train_test_split(dataframe, test_size=0.3, stratify=dataframe["labels"])
    val, test = train_test_split(val, test_size=0.1)

    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest"
    )

    valid_datagen = ImageDataGenerator(rescale=1/255.0)

    train_gen = train_datagen.flow_from_dataframe(
        train,
        x_col="image_path",
        y_col="labels",
        seed=45,
        batch_size=32,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode='categorical'
    )
    val_gen = valid_datagen.flow_from_dataframe(
        val,
        x_col="image_path",
        y_col="labels",
        seed=45,
        batch_size=32,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode='categorical'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_dataframe(
        test,
        x_col="image_path",
        y_col="labels",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
    )

    return train_gen, val_gen, test_gen

def get_train_val_test_dataset(
        image_folder_path: str, dataframe: pd.DataFrame, img_height: int, img_width: int,
        output_folder: Optional[str]="", validation_split: Optional[float]=0.2, batch_size: Optional[int]=32):
    """ Creates train and validation dataset based on jpg or not from image folder hierarchy

        Args:
            image_folder_path(str) : Folder containing images for training and validation
            dataframe(pd.DataFrame): Dataframe containing images for training and validation
            img_height(int)        : Height of the image
            img_width(int)         : Width of the image
        KwArgs:
            output_folder(str)     : Output folder to create splitted images directory
            validation_split(float): Split ratio for validation
            batch_size(int)        : Batch size
    """
    ext = get_img_extension(image_folder_path)
    if len(ext) == 1 and ext[0] == "jpg":
        return get_image_dataset_from_jpg(
            image_folder_path, img_height, img_width, output_folder="",
            validation_split=0.2, batch_size=32
        )
    else:
        return get_image_dataset_from_multiple_ext(
            dataframe, img_height, img_width, output_folder="",
            validation_split=0.2, batch_size=32
        )


class Albumentation:
    """ Perform albumentation augment on dataset
    """
    def __init__(self, transform: A.Compose):
        self._transforms = transform

    def perform_augmentation(self, image: np.array, img_size: int):
        """ Perform augmentation on image

            Args:
                image(np.array): Image array
                img_size(int): Image size as height, width
        """
        data = {"image": image}
        aug_data = self._transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img/255.0, tf.float32)
        return tf.image.resize(aug_img, size=[img_size, img_size])
    
    def process_data(self, image: np.array, label: np.array, img_size: int):
        """ Start performing image operation

            Args:
                image(np.array): Image array
                label(np.array): Label array
                img_size(int): Image size as height, width
        """
        aug_img = tf.numpy_function(func=self.perform_augmentation, inp=[image, img_size], Tout=tf.float32)
        return aug_img, label

    def set_shapes(self, img: np.array, label: np.array, img_shape: Optional[tuple]=(224, 224, 3)):
        """ Reset shapes

            Args:
                image(np.array): Image array
                label(np.array): Label array
                img_size(tuple): Image size as height, width, channels
        """
        img.set_shape(img_shape)
        label.set_shape([])
        return img, label
    
    def __call__(self, data: Generator, img_shape: Optional[tuple]=(224, 224, 3)):
        """ Apply albumentation
        """
        ds_alb = data.map(
            partial(self.process_data, img_size=img_shape[0]),
            num_parallel_calls=AUTOTUNE
        ).prefetch(AUTOTUNE)
        return ds_alb.map(
            partial(self.set_shapes, img_size=img_shape),
            num_parallel_calls=AUTOTUNE
        ).batch(32).prefetch(AUTOTUNE)


class TrainingMonitor(Callback):
  def __init__(self, fig_path, json_path=None, startAt=0):
    # store the output path for the figure, the path to the JSON
    # serialized file, and the starting epoch
    super(TrainingMonitor,self).__init__()
    self.fig_path = fig_path
    self.json_path = json_path
    self.startAt = startAt

  def on_train_begin(self, logs=None):
    logs = logs or {}
    # initialize the history dictionary
    self.H = {}

    # if the JSON history path exists, load the training history
    if self.json_path is not None:
      if os.path.exists(self.json_path):
        self.H = json.loads(open(self.json_path).read())

        # check to see if a starting epoch was supplied
        if self.startAt > 0:
          # loop over the entries in the history log and
          # trim any entries that are past the starting
          # epoch
          for k in self.H.keys():
            self.H[k] = self.H[k][:self.startAt]

  def on_epoch_end(self, epoch, logs={}):
    # loop over the logs and update the loss, accuracy, etc.
    # for the entire training process
    for (k, v) in logs.items():
      l = self.H.get(k, [])
      l.append(float(v))
      self.H[k] = l

      # check to see if the training history should be serialized to file
      if self.json_path is not None:
        f = open(self.json_path, "w")
        f.write(json.dumps(self.H))
        f.close()

      # ensure at least two epochs have passed before plotting (epoch starts at zero)
      if len(self.H["loss"]) > 1:
        # plot the training loss and accuracy
        N = np.arange(0, len(self.H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.H["loss"], label="train_loss")
        plt.plot(N, self.H["val_loss"], label="val_loss")
        plt.plot(N, self.H["acc"], label="train_acc")
        plt.plot(N, self.H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        plt.savefig(self.fig_path)
        plt.close()

class Callbacks:

    @staticmethod
    def training_monitor(fig_path: str, json_path: str):
        """ Monitor training at each epoch

        Args:
            fig_path(str): Path to save training plots
            json_path(str): Path to save training results as json
        """
        return TrainingMonitor(fig_path, json_path)

    @staticmethod
    def reduce_lr_on_plateau(metric_to_monitor='val_loss', mode="min", factor=0.2, patience=5, min_lr=0.001, min_delta=0.001):
        """ Callback to reduce learning rate on plateau

            KwArgs:
                metric_to_monitor(str): Quantity to be monitored.
                mode(str): When to stop training in mode(min, max, auto)
                factor(float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
                patience(int): Number of epochs with no improvement after which learning rate will be reduced.
                min_lr(float): Lower bound on the learning rate.
                min_delta(float): Threshold for measuring the new optimum, to only focus on significant changes.
        """
        return ReduceLROnPlateau(
            monitor=metric_to_monitor, factor=factor, patience=patience, min_lr=min_lr,
            min_delta=min_delta, mode=mode, verbose=1
        )
    
    @staticmethod
    def early_stopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True):
        """ Stop training when a monitored metric has stopped improving.

            KwArgs:     
                monitor(str): Quantity to be monitored.
                mode(str): When to stop training in mode(min, max, auto)
                patience(int): Number of epochs with no improvement after which training will be stopped
                restore_best_weights(bool):
        """
        return EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience, restore_best_weights=restore_best_weights)
    
    @staticmethod
    def model_checkpoint(dirname, monitor="val_loss", mode="min", save_best_only=True):
        """ save only the *best* model to disk based on the validation loss

            Args:
                dirname(str): Name of the directory to save model
            KwArgs:
                monitor(str): Quantity to be monitored
                mode(str): When to stop training in mode(min, max, auto)
                save_best_only(bool): restore model weights from the epoch with the best value of the monitored quantity
        """
        fname = os.path.join(dirname, "weights-{epoch:03d}-{val_loss:.4f}.keras")


        return ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    
    @staticmethod
    def tensorboard(log_dir="./logs", profile_batch=5):
        """ Enable visualizations for TensorBoard.
            KwArgs:
                log_dir(str): path of the directory to save the log files
                profile_batch(int): Profile the batch
        """
        return TensorBoard(log_dir=log_dir, profile_batch=profile_batch)

ModelNames = Literal["vgg16", "vgg19", "inception", "xception", "resnet"]
# Typing for a Keras data generator
KerasGenerator = Generator[Tuple[np.ndarray, np.ndarray], None, None]

class ClassificationModel:
    @staticmethod
    def build(model_name: ModelNames, num_class: int, input_shape: tuple[int, int, int]=(224, 224, 3), learning_rate: float=0.001)->Model:
        """ Returns keras pretrained model

            Args:
                model_name(str)     : Name of the model
                num_class(int)      : Total no of classes to classify
                input_shape(tuple)  : Image shape in (height, width, channels)
                learning_rate(float): Learning rate hyper parameter
        """
        loss = "categorical_crossentropy"
        activation = "softmax"
        if num_class == 2:
            activation = "sigmoid"
            loss = "binary_crossentropy"

        inputs = layers.Input(shape=input_shape, name="input_layer")
        base_model = ClassificationModel.get_base_model(model_name, inputs)
        base_model.trainable = False
        # Enable the last 5 layers of VGG16 to be trainable
        for layer in base_model.layers[-5:]:  # Last 5 layers
            layer.trainable = True

        augment_layer = Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2)
            ]
        )

        x = augment_layer(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)  # Add a fully connected layer
        x = layers.Dropout(0.5)(x)                   # Dropout layer for regularization
        outputs = layers.Dense(num_class, activation=activation, name="output_layer")(x)

        model = Model(inputs, outputs)
        model.compile(
            loss=loss, optimizer=optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"]
        )
        return model
    
    @staticmethod
    def train(model: Model, train_data, val_data, epochs: Optional[int]=10, callbacks: Optional[list]=None)->dict:
        """ Training process

            Args:
                model(Model): Tensorflow model to be used for prediction
                train_data(ImageDataGenerator): Train data to train
                val_data(ImageDataGenerator): Val data to train
            KwArgs:
                epochs(int): No of epochs to train
                callbacks(list): List of callbacks
        """
        history = model.fit(
            train_data,
            steps_per_epoch=train_data.samples // train_data.batch_size,
            epochs=epochs,
            validation_data=val_data,
            validation_steps=val_data.samples // val_data.batch_size,
            callbacks=callbacks or [], verbose=1
            )
        return history

    @staticmethod
    def get_base_model(model_name:ModelNames, inputs:layers.Input)->Model:
        """ Get the base model initialized

            Args:
                model_name(str): Name of the model
                inputs(layers.Input): Input layer
        """
        MODELS = {
            "vgg16": applications.VGG16,
            "vgg19": applications.VGG19,
            "inception": applications.InceptionV3,
            "xception": applications.Xception, # TensorFlow ONLY
            "resnet": applications.ResNet50
        }
        return MODELS[model_name](weights="imagenet", include_top=False, input_tensor=inputs)
    
    @staticmethod
    def predict(model: Model, model_name: ModelNames, image_path: str, input_shape: tuple[int, int]=(224, 224)):
        """ Predicts the image and outputs the results

            Args:
                model(Model): Tensorflow model to be used for prediction
                model_name(str): name of the model
                image_path(str): Image path to predict the model result

            KwArgs:
                input_shape(tuple): Image height and width
        """
        if model_name in ("inception", "xception"):
            input_shape = (299, 299)
            preprocess = preprocess_input

        image = load_img(image_path, target_size=input_shape)
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = preprocess(image)

        print(f"classifying image with {model_name}...")
        preds = model.predict(image)
        P = imagenet_utils.decode_predictions(preds)

        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            print(f"{i + 1}. {label}: {prob * 100:.2f}%")
        
        orig = cv2.imread(image_path)
        (imagenetID, label, prob) = P[0][0]
        predicted_image = cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.imshow(predicted_image)

    @staticmethod
    def predict_in_confusion_matrix(model: Model, test_generator: KerasGenerator):
        """ Predicts for whole test generator and visualizes in confusion matrix

            Args:
                model(Model): Model to be used for prediction
                test_generator(KerasGenerator): Test image generator to predict
        """
        # Generate predictions
        predictions = model.predict(test_generator)
        # Convert probabilities to class labels
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes  # Ground truth labels
        class_labels = list(test_generator.class_indices.keys())  # Class names

        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Print classification report
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))

        # Get a batch of images from the generator
        test_images, _ = next(test_generator)

        # Plot some images with their predictions
        num_images = 10
        plt.figure(figsize=(15, 8))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(test_images[i])
            plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[predicted_classes[i]]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def evaluate(model: Model, test_generator: KerasGenerator):
        """ Evaluates the model

            Args:
                model(Model): Model to be used for prediction
                test_generator(KerasGenerator): Test image generator to predict
        """
        loss, accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# dataset creation to dataframe
image_folder = ''
dataframe = get_dataframe_from_image_paths(image_folder)
class_names = dataframe["labels"].unique()

# visualisation
plot_class_distribution_in_pie_chart(dataframe)
plot_class_distribution_in_count_chart(dataframe)
plot_random_images_per_class(dataframe)

#datasplit
img_height = img_width = 224
channels = 3
output_folder = os.path.join(os.path.dirname(image_folder), "dataset")
train_gen, val_gen, test_gen = get_train_val_test_dataset(
    image_folder, dataframe, img_height, img_width,
    output_folder=output_folder
)

train_transform = A.Compose([
    A.RandomCrop((img_height, img_width, channels)),
    A.HorizontalFlip(),
    A.RandomBrightness(limit=0.2),
    A.Rotate(limit=40),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.HorizontalFlip(),
    A.RandomContrast(limit=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the Albumentations transform for testing
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_gen = Albumentation(train_transform)(train_gen, (img_height, img_width, channels))
test_gen = Albumentation(test_transform)(test_gen, (img_height, img_width, channels))

ckpoint_path = os.path.join(os.path.dirname(image_folder), "ck_pt")
if not os.path.exists(ckpoint_path):
    os.makedirs(ckpoint_path)
tensorboard_path = os.path.join(os.path.dirname(image_folder), "tensorboard")
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

fig_path = os.path.sep.join(["/content/sample_data/output", "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(["/content/sample_data/json", "{}.json".format(os.getpid())])

if not os.path.exists(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))
json_path = os.path.join(os.path.dirname(image_folder), "json")
if not os.path.exists(os.path.dirname(json_path)):
    os.makedirs(os.path.dirname(json_path))
callbacks = [
    Callbacks.reduce_lr_on_plateau(),
    Callbacks.early_stopping(),
    Callbacks.model_checkpoint(ckpoint_path),
    Callbacks.tensorboard(log_dir=tensorboard_path),
    Callbacks.training_monitor(fig_path, json_path)
]

model = ClassificationModel.build("vgg16", 10)
ClassificationModel.train(model, train_gen, val_gen, callbacks=callbacks)
ClassificationModel.predict_in_confusion_matrix(model, test_gen)
ClassificationModel.evaluate(model, test_gen)

img, label = test_gen.take(0)
ClassificationModel.predict(model, "vgg16", img)
