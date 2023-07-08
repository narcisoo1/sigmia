import os

class CoreCNN:
    """
    Classe principal para a lógica do CNN (Convolutional Neural Network).
    """

    def __init__(self):
        """
        Construtor da classe CoreCNN.
        """
        pass

    def load_folders(self, path):
        """
        Carrega as pastas de um diretório específico.

        Args:
            path (str): Caminho para o diretório contendo as pastas.

        Returns:
            int: Número de classes (pastas) encontradas.
        """
        self.PATH = path
        self.classes = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return len(self.classes)
