import os

class Config:
    def __init__(self):
        self.large_model_api_url = os.getenv("LARGE_MODEL_API_URL", "http://default-api-url.com")

    def get_large_model_api_url(self):
        return self.large_model_api_url

    def set_large_model_api_url(self, url):
        self.large_model_api_url = url
        os.environ["LARGE_MODEL_API_URL"] = url
