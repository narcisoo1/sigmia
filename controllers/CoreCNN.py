import os

class CoreCNN:
    def __init__(self):
        pass

    def load_folders(self, path):
        self.PATH = path
        self.classes = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return len(self.classes)

    def runCNN(self,variable):
        pass
