from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class LogTrain(QWidget):
    def __init__(self, model_name, estimated_time):
        super().__init__()
        self.model_name = model_name
        self.estimated_time = estimated_time
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Training Output')

        # Centralized text widget for model optimization message
        self.message_label = QLabel(f"Otimizando o modelo {self.model_name}, tempo estimado de {self.estimated_time} minutos", self)
        self.message_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.message_label)

        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #9dcfff;
                border-radius: 10px;
            }
            QLabel {
                color: #1e25ff;
                font-size: 20px;  /* Aumentando o tamanho do texto */
                padding: 10px;
                margin: 4px 2px;
            }
        """)

        self.show()
    def update_estimated_time(self, estimated_time):
        self.message_label.setText(f"Otimizando o modelo {self.model_name}, tempo estimado de {estimated_time}")

