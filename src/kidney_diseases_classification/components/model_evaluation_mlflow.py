import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from kidney_diseases_classification.entity.config_entity import EvaluationConfig
from kidney_diseases_classification.utils.common import read_yaml, create_directories,save_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        test_datagen = ImageDataGenerator(rescale = 1/255.0,
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range = 0.2,
                            vertical_flip=True,
                            horizontal_flip = True,
                            fill_mode="reflect")


        self.test_generator = test_datagen.flow_from_directory(
            directory=self.config.test_data,
            target_size=self.config.params_image_size[:-1],
            class_mode='categorical',
            batch_size = self.config.params_batch_size,
            shuffle = False)



    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.set_experiment("experiment")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="InceptionV3Model")
            else:
                mlflow.keras.log_model(self.model, "model")