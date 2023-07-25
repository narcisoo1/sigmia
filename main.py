from PyQt5 import QtWidgets
from controllers.MainController import ApplicationWindow
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtCore import Qt
import sys


def set_light_palette(app):
    # Defina a paleta de cores da aplicação como uma paleta clara
    palette = app.palette()
    palette.setColor(palette.Window, Qt.white)  # Cor do fundo da janela
    palette.setColor(palette.WindowText, Qt.black)  # Cor do texto da janela
    # Defina outras cores de acordo com suas necessidades

    app.setPalette(palette)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('GTK'))
    set_light_palette(app)
    application = ApplicationWindow()

    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
