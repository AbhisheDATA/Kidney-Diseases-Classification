import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from kidney_diseases_classification.entity.config_entity import TrainingConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        train_datagenerator = ImageDataGenerator(rescale = 1/255.0,
                        rotation_range=15,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range = 0.2,
                        vertical_flip=True,
                        horizontal_flip = True,
                        fill_mode="reflect")
        
        self.train_generator = train_datagenerator.flow_from_directory(
                            directory=self.config.training_data,
                            target_size=self.config.params_image_size[:-1],
                            class_mode='categorical', 
                            batch_size = self.config.params_batch_size)
        
        val_datagen = ImageDataGenerator(rescale = 1/255.0)


        self.valid_generator = train_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            target_size=self.config.params_image_size[:-1],
            class_mode='categorical',
            batch_size = self.config.params_batch_size)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.model.fit(
                    self.train_generator,
                    steps_per_epoch=len(self.train_generator),
                    epochs=self.config.params_epochs,
                    validation_data=self.valid_generator)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )