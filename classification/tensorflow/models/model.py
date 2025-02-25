# -*- coding: utf-8 -*-
"""
Model

This module contains the ClassificationModel class which provides methods to build, train, and evaluate deep learning models.
"""

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Optional, Literal, Generator, Tuple
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from keras import layers, Model, optimizers, applications
from keras.models import load_model, Sequential

from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img


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

            Returns:
                Model: A compiled Keras model
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
                layers.Normalization(),
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation",
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

            Returns:
                dict: Training history
        """
        #caclulate steps per epoch
        if isinstance(train_data, tf.data.Dataset):
            steps_per_epoch = train_data.cardinality().numpy()  # Number of batches
            val_steps = val_data.cardinality().numpy()
        else:
            steps_per_epoch = train_data.samples // train_data.batch_size
            val_steps = val_data.samples // val_data.batch_size
        
        history = model.fit(
            train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_data,
            validation_steps=val_steps,
            callbacks=callbacks or [], verbose=1
            )
        return history

    @staticmethod
    def get_base_model(model_name:ModelNames, inputs:layers.Input)->Model:
        """ Get the base model initialized

            Args:
                model_name(str): Name of the model
                inputs(layers.Input): Input layer

            Returns:
                Model: A Keras base model
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
    def predict(model: Model, model_name: ModelNames, true_label: str, class_labels: list, image_path: str, input_shape: tuple[int, int]=(224, 224)):
        """ Predicts the image and outputs the results

            Args:
                model(Model): Tensorflow model to be used for prediction
                model_name(str): name of the model
                true_label(str): True label
                class_label(list): List of class label
                image_path(str): Image path to predict the model result

            KwArgs:
                input_shape(tuple): Image height and width
        """
        preprocess = imagenet_utils.preprocess_input
        if model_name in ("inception", "xception"):
            input_shape = (299, 299)
            preprocess = preprocess_input

        if isinstance(image_path, str):
            image = load_img(image_path, target_size=input_shape)
            image = img_to_array(image)
            orig_image = np.copy(image)
        else:
            image = image_path
            orig_image = np.copy(image)

        tf_image = np.expand_dims(image, axis=0)
        preprocessed_image = preprocess(tf_image)

        print(f"classifying image with {model_name}...")
        preds = model.predict(preprocessed_image)
        pred_label =  class_labels[np.argmax(preds, axis=-1)[0]]

        predicted_image = cv2.putText(orig_image.astype('uint8'), f"Pred Label: {pred_label}\nTrue Label: {true_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.imshow(predicted_image)

    @staticmethod
    def predict_in_confusion_matrix(model: Model, test_generator: KerasGenerator, class_labels: list[str], true_classes: list[int]):
        """ Predicts for whole test generator and visualizes in confusion matrix

            Args:
                model(Model): Model to be used for prediction
                test_generator(KerasGenerator): Test image generator to predict
                class_labels(list[str]): List of class labels
                true_classes(list[int]): List of true class indices
        """
        # Generate predictions
        predictions = model.predict(test_generator)
        # Convert probabilities to class labels
        predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = test_generator.classes  # Ground truth labels

        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(class_labels)))

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

            Returns:
                tuple: Loss and accuracy of the model
        """
        loss, accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


def train_model(model, train_gen, val_gen, test_gen, callbacks, epochs=10, model_name=""):
    """ Train model

        Args:
            model(Model): Tensorflow model to be used for training
            train_gen(ImageDataGenerator): Training data generator
            val_gen(ImageDataGenerator): Validation data generator
            test_gen(ImageDataGenerator): Test data generator
            callbacks(list): List of callbacks
            epochs(int): Number of epochs to train
            model_name(str): Name of the model
    """
    class_indices = train_gen.class_indices
    class_labels = list(train_gen.class_indices.keys())  # Class names
    true_classes = train_gen.classes
    print("Class Indices:", class_indices)

    ClassificationModel.train(model, train_gen, val_gen, epochs=epochs, callbacks=callbacks)
    ClassificationModel.predict_in_confusion_matrix(model, test_gen, class_labels, true_classes)
    ClassificationModel.evaluate(model, test_gen)

    batch_images, batch_labels = next(test_gen)

    batch_image = batch_images[0]
    batch_label = batch_labels[0]
    true_label = class_labels[test_gen.classes[0]]
    print(f"Images shape: {batch_image.shape}")
    print(f"Labels shape: {batch_label.shape}")

    ClassificationModel.predict(model, model_name, true_label, class_labels, batch_image)


class BaseModel:
    def __init__(self, no_of_classes, input_shape=(224, 224, 3), learning_rate: float=0.001):
        self.input_shape = input_shape
        self.classes = no_of_classes
        self.learning_rate = learning_rate

class ResnetFromScratch(BaseModel):
    @staticmethod
    def identity_block(x, filter_size):
        x_skip = x

        x = layers.Conv2D(filter_size, 3, padding="same")(x)  # Remove strides=2
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter_size, 3, padding="same")(x)  # Remove strides=2
        x = layers.BatchNormalization(axis=-1)(x)

        x = layers.Add()([x, x_skip])
        x = layers.Activation('relu')(x)

        return x

    @staticmethod
    def conv_block(x, filter_size):
        x_skip = x

        x = layers.Conv2D(filter_size, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filter_size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x_skip = layers.Conv2D(filter_size, 1, strides=2, padding="same")(x_skip)  # Fix x_skip

        x = layers.Add()([x, x_skip])
        x = layers.Activation('relu')(x)

        return x


    def build(self):
        loss, activation = ("binary_crossentropy", "sigmoid") if self.classes == 2 else ("categorical_crossentropy", "softmax")

        input = layers.Input(self.input_shape)
        augment_layer = Sequential(
            [
                layers.Normalization(),
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation",
        )
        x = augment_layer(input)
        # x = layers.ZeroPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(64, 7, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        block_layers = [3, 4, 6, 3]
        filter_size = [64, 128, 256, 512]

        for  i in range(len(block_layers)):
            if i == 0:
                for j in range(block_layers[i]):
                    x = ResnetFromScratch.identity_block(x, filter_size[i])
            else:
                x = ResnetFromScratch.conv_block(x, filter_size[i])
                for j in range(block_layers[i]-1):
                    x = ResnetFromScratch.identity_block(x, filter_size[i])

        x = layers.AveragePooling2D((2, 2), padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        output = layers.Dense(self.classes, activation=activation)(x)  # Fix missing (x)
        model = Model(input, output)
        model.compile(
            loss=loss, optimizer=optimizers.Adam(learning_rate=self.learning_rate), metrics=["accuracy"]
        )
        return model

class InceptionModelFromScratch(BaseModel):
    def build(self):
        loss, activation = ("binary_crossentropy", "sigmoid") if self.classes == 2 else ("categorical_crossentropy", "softmax")

        input = layers.Input(self.input_shape)
        augment_layer = Sequential(
            [
                layers.Normalization(),
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation",
        )
        x = augment_layer(input)
        x = layers.Conv2D(64, 7, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = layers.Conv2D(64, 1, padding="same")(x)
        x = layers.Conv2D(192, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        inception_3a = InceptionModelFromScratch.inception_module(x, 64, 96, 128, 16, 32, 32)
        inception_3b = InceptionModelFromScratch.inception_module(inception_3a, 128, 128, 192, 32, 96, 64)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(inception_3b)

        inception_4a = InceptionModelFromScratch.inception_module(x, 192, 96, 208, 16, 48, 64)
        inception_4b = InceptionModelFromScratch.inception_module(inception_4a, 160, 112, 224, 24, 64, 64)
        inception_4c = InceptionModelFromScratch.inception_module(inception_4b, 128, 128, 256, 24, 64, 64)
        inception_4d = InceptionModelFromScratch.inception_module(inception_4c, 112, 144, 288, 32, 64, 64)

        inception_5a = InceptionModelFromScratch.inception_module(inception_4d, 256, 160, 320, 32, 128, 128)
        inception_5b = InceptionModelFromScratch.inception_module(inception_5a, 384, 192, 384, 48, 128, 128)
        
        x = layers.AveragePooling2D((2, 2), padding="same")(inception_5b)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        output = layers.Dense(self.classes, activation=activation)(x)
        model = Model(input, output)
        model.compile(
            loss=loss, optimizer=optimizers.Adam(learning_rate=self.learning_rate), metrics=["accuracy"]
        )
        return model
    
    @staticmethod
    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
        conv_1x1 = layers.Conv2D(filters_1x1, 1, padding="same", activation="relu")(x)

        conv_3x3 = layers.Conv2D(filters_3x3_reduce, 1, padding="same", activation="relu")(x)
        conv_3x3 = layers.Conv2D(filters_3x3, 3, padding="same", activation="relu")(conv_3x3)

        conv_5x5 = layers.Conv2D(filters_5x5_reduce, 1, padding="same", activation="relu")(x)
        conv_5x5 = layers.Conv2D(filters_5x5, 5, padding="same", activation="relu")(conv_5x5)

        pool_proj = layers.MaxPool2D(3, strides=1, padding="same")(x)
        pool_proj = layers.Conv2D(filters_pool_proj, 1, padding="same", activation="relu")(pool_proj)

        return layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    

class DenseNetFromScratch(BaseModel):
    def build(self):
        loss, activation = ("binary_crossentropy", "sigmoid") if self.classes == 2 else ("categorical_crossentropy", "softmax")

        input = layers.Input(self.input_shape)
        augment_layer = Sequential(
            [
                layers.Normalization(),
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation",
        )
        x = augment_layer(input)
        x = layers.Conv2D(64, 7, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = DenseNetFromScratch.dense_block(x, 64, 6)
        x = DenseNetFromScratch.transition_block(x, 64)

        x = DenseNetFromScratch.dense_block(x, 128, 12)
        x = DenseNetFromScratch.transition_block(x, 128)

        x = DenseNetFromScratch.dense_block(x, 256, 24)
        x = DenseNetFromScratch.transition_block(x, 256)

        x = DenseNetFromScratch.dense_block(x, 512, 16)

        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(self.classes, activation=activation)(x)
        model = Model(input, output)
        model.compile(
            loss=loss, optimizer=optimizers.Adam(learning_rate=self.learning_rate), metrics=["accuracy"]
        )
        return model

    @staticmethod
    def dense_block(x, filters, blocks):
        for i in range(blocks):
            x = DenseNetFromScratch.conv_block(x, filters)
        return x

    @staticmethod
    def conv_block(x, filters):
        x_skip = x

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Concatenate()([x_skip, x])

        return x
    
    @staticmethod
    def transition_block(x, filters):
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.AveragePooling2D(2, strides=2)(x)

        return x


class MobileNetFromScratch(BaseModel):
    def build(self):
        loss, activation = ("binary_crossentropy", "sigmoid") if self.classes == 2 else ("categorical_crossentropy", "softmax")

        input = layers.Input(self.input_shape)
        augment_layer = Sequential(
            [
                layers.Normalization(),
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomContrast(factor=0.1),
                layers.RandomFlip("horizontal"),
                layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation",
        )
        x = augment_layer(input)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = MobileNetFromScratch.conv_block(x, 64)
        x = MobileNetFromScratch.conv_block(x, 128)
        x = MobileNetFromScratch.conv_block(x, 128)
        x = MobileNetFromScratch.conv_block(x, 256)
        x = MobileNetFromScratch.conv_block(x, 256)
        x = MobileNetFromScratch.conv_block(x, 512)

        x = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(self.classes, activation=activation)(x)
        model = Model(input, output)
        model.compile(
            loss=loss, optimizer=optimizers.Adam(learning_rate=self.learning_rate), metrics=["accuracy"]
        )
        return model

    @staticmethod
    def conv_block(x, filters):
        x = layers.DepthwiseConv2D(3, padding="same")(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x