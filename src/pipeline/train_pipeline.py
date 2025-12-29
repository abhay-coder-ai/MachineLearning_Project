from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Modeltrainer
from src.exception import CustomException
from src.logger import logging
import sys

if __name__ == "__main__":
    try:
        logging.info("Training pipeline started")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data transformation completed")

        trainer = Modeltrainer()
        trainer.initiate_model_trainer(
            train_arr,
            test_arr,
            "artifacts/preprocessor.pkl"
        )
        logging.info("Model training completed")

    except Exception as e:
        raise CustomException(e, sys)
