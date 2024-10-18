from PyQt5.QtWidgets import  QMessageBox, QProgressBar
class CustomMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(5, 30, 200, 25)
        self.progress_bar.hide()

        # Configurações de estilo
        self.setStyleSheet("""
            QWidget {
                background-color: #9dcfff;
                border-radius: 10px;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                text-align: center;
                font-size: 16px;
                padding: 10px 20px;
                margin: 4px 2px;
                border-radius: 8px;
            }
             QProgressBar {
                border: none;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background-color: #1E90FF; /* Cor da barra de progresso */
                width: 10px; /* Largura da barra de progresso */
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
        """)

    def set_progress_value(self, value):
        self.progress_bar.setValue(value)

    def show_progress_bar(self):
        self.progress_bar.show()

    def hide_progress_bar(self):
        self.progress_bar.hide()