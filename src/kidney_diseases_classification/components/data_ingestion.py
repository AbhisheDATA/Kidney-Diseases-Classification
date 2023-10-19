import os
import zipfile
import gdown
from kidney_diseases_classification import logger
from kidney_diseases_classification.entity.config_entity import DataIngestionConfig
from kidney_diseases_classification.utils.common import get_size
import splitfolders
from pathlib import Path 

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def split_folder(self):
        """
        split_folder_path: str
        split the dataset in train test val
        Function returns None
        """
        split_path=self.config.root_dir
        os.makedirs(split_path, exist_ok=True)
        pa = os.getcwd()
        full_path = os.path.join(pa,self.config.root_dir, self.config.data_dir)
        full_path = Path(full_path)
        splitfolders.ratio(full_path,output=self.config.split_dir,seed=7,ratio=(0.8,0.1, 0.1))
