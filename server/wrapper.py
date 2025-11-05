from models.images import Image
from models.music import Music
from models.text import Text


class EnsembleWrapper:
    def __init__(self):
        self.models = [Image(), Music(), Text()]

    def predict(self, x):
        return 0
