import sys

from View.SPTI import SPTI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SPTI()
    ex.show()
    sys.exit(app.exec_())
