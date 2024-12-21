from landpy import ImageLand, Debugger
from abc import ABC, abstractmethod
import tensorflow as tf
import os
import requests

from landpy.pretrained_models.config import DOWNLOAD_DIR


class Model(ABC):
    def __init__(self, model_url=None, input_type=None, output_type=None, debugger: Debugger = Debugger(debug=False)):
        self._debugger: Debugger = debugger
        self.input_type = input_type
        self.output_type = output_type
        self.__model_url = model_url

        self.model_name = os.path.basename(model_url)
        self.model_format = self.__model_url.split('.')[-1].lower()

        self.__validate_format(self.model_format, debugger)

        self.model_path = os.path.join(DOWNLOAD_DIR, f"{self.model_name}")

        self.__download_model_if_not_exist()
        self._interpreter = self._load_model()

    @abstractmethod
    def _load_model(self):
        pass
    
    @staticmethod
    def __validate_format(model_format, debugger: Debugger):
        if model_format not in ['h5', 'tflite']:
            debugger.log_exception('Model type must be h5 or tflite!', [
                                   'Model.__init__', 'Model'])

    def __download_model_if_not_exist(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if not os.path.exists(self.model_path):
            self._debugger.log(
                f"{self.model_name} not found. Downloading from { self.__model_url}")
            self._download_model()
        else:
            self._debugger.log(
                f"{self.model_name} found at {self.model_path}.")

    @Debugger.method_decorator('Downloading model')
    def _download_model(self):
        response = requests.get(self.__model_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percentage = (downloaded_size / total_size) * 100
                            self._debugger.log(
                                f"Downloading {self.model_name}: {percentage:.2f}%", end="\r")
                        else:
                            self._debugger.log(
                                f"Downloading {self.model_name}: {downloaded_size} bytes", end="\r")

            self._debugger.log(f"{self.model_name} downloaded to {self.model_path}.")
        else:
            self._debugger.log_exception(
                f"Failed to download model from {self.github_url}. Status code: {response.status_code}")

    # def __load_model(self):
    #     if self.model_format == 'h5':
    #         return load_tf(self.model_path, self._debugger)
    #     elif  self.model_format == 'tflite':
    #         return load_tflite(self.model_path, self._debugger)

    # @Debugger.method_decorator('Inferencing by model', ['Model.inference', 'Model'])
    # def inference(self, input):
    #     if self.input_type is not None and type(input) is not self.input_type:
    #         self._debugger.log_exception(
    #             "inference input type must be " + str(self.input_type), self._debugger.get_last_log_categories())

    #     if self.model_format == 'h5':
    #         return inference_by_tf(self._interpreter, input, self._debugger)
    #     elif  self.model_format == 'tflite':
    #         return inference_by_tflite(self._interpreter, input, self._debugger)


# Example usage
if __name__ == "__main__":
    # Replace with your actual GitHub raw URL
    model_url = "https://github.com/bateni1380/fine_tuned_models/raw/refs/heads/main/model_files/face_landmark.tflite"
    model = Model(model_url)
    print("Model loaded successfully.")
