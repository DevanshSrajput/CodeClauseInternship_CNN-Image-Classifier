import sys
from PyQt5.QtWidgets import QApplication
from src.gui import ImageClassifierGUI

def main():
    app = QApplication(sys.argv)
    window = ImageClassifierGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()