from pydantic_settings import BaseSettings

class Settings(BaseSettings):
      bank_data_path: str
      bank_lg_trained_model: str
      bank_sc_trained_model: str
      bank_load_trained_data_lg: str
      bank_load_trained_data_sc: str
      cancer_data_path: str
      cancer_lg_trained_model: str
      cancer_sc_trained_model: str
      cancer_load_trained_data_lg: str
      cancer_load_trained_data_sc: str
      
      class Config:
            env_file = ".env"  # Tell Pydantic to load from .env

settings = Settings()