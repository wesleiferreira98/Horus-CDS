import sys
import os
import traceback
import faulthandler

from View.SPTI import SPTI
from PyQt5.QtWidgets import QApplication

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_LOG = os.path.join(BASE_DIR, "horus_crash.log")


def log_uncaught_exception(exc_type, exc_value, exc_tb):
    with open(CRASH_LOG, "a", encoding="utf-8") as crash_file:
        crash_file.write("\n=== Unhandled Python exception ===\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=crash_file)
    traceback.print_exception(exc_type, exc_value, exc_tb)


if __name__ == '__main__':
    with open(CRASH_LOG, "a", encoding="utf-8") as crash_file:
        crash_file.write("\n=== Horus-CDS session start ===\n")
        faulthandler.enable(file=crash_file, all_threads=True)

    sys.excepthook = log_uncaught_exception
    app = QApplication(sys.argv)
    ex = SPTI()
    ex.show()
    sys.exit(app.exec_())
