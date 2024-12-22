import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Optional, Literal, Generator, Tuple
from sklearn.metrics import confusion_matrix, classification_report

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


def train_model(model_name, num_classes, train_gen, val_gen, test_gen, callbacks):
    model = ClassificationModel.build(model_name, num_classes)
    ClassificationModel.train(model, train_gen, val_gen, callbacks=callbacks)
    ClassificationModel.predict_in_confusion_matrix(model, test_gen)
    ClassificationModel.evaluate(model, test_gen)

    img, label = test_gen.take(0)
    ClassificationModel.predict(model, model_name, img)