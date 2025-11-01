"""
Widget de Console para exibir logs de treinamento em tempo real
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QObject
from PyQt5.QtGui import QFont, QTextCursor
import sys
from io import StringIO


class TrainingConsole(QWidget):
    """Console integrado para exibir logs de treinamento"""
    
    # Signal para adicionar texto de forma thread-safe
    log_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.capturing = False
        
        # Conectar signal ao slot
        self.log_signal.connect(self.append_text)
        
    def initUI(self):
        """Inicializa a interface do console"""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Console de Treinamento")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1E90FF;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Botao de toggle (expandir/recolher)
        self.toggle_button = QPushButton("▼ Recolher Detalhes")
        self.toggle_button.setFixedWidth(150)
        self.toggle_button.clicked.connect(self.toggle_console)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        header_layout.addWidget(self.toggle_button)
        
        # Botao para limpar
        self.clear_button = QPushButton("Limpar")
        self.clear_button.setFixedWidth(100)
        self.clear_button.clicked.connect(self.clear_console)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #FF4500;
            }
        """)
        header_layout.addWidget(self.clear_button)
        
        layout.addLayout(header_layout)
        
        # Text area para logs
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setMinimumHeight(250)
        self.console_text.setMaximumHeight(400)
        
        # Configurar fonte monoespaçada com suporte Unicode
        font = QFont("Monospace", 9)
        font.setStyleHint(QFont.TypeWriter)
        self.console_text.setFont(font)
        
        # Estilo do console
        self.console_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #3f3f3f;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        
        layout.addWidget(self.console_text)
        
        self.setLayout(layout)
    
    @pyqtSlot(str)
    def append_text(self, text):
        """Adiciona texto ao console"""
        # Limpar códigos ANSI (caracteres de formatação do terminal)
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # Substituir caracteres Unicode problemáticos por alternativas ASCII
        text = text.replace('━', '=')  # Barra horizontal para barra de progresso
        
        # Colorir diferentes tipos de mensagens
        if "ERRO" in text.upper() or "ERROR" in text.upper():
            color = "#ff6b6b"
        elif "AVISO" in text.upper() or "WARNING" in text.upper():
            color = "#ffd93d"
        elif "SUCESSO" in text.upper() or "SUCCESS" in text.upper():
            color = "#6bcf7f"
        elif "GPU" in text.upper():
            color = "#4ecdc4"
        elif any(word in text for word in ["Treino:", "Teste:", "amostras", "Features"]):
            color = "#95e1d3"
        elif "━━━" in text or "step" in text.lower():  # Barra de progresso do keras
            color = "#a8dadc"
        else:
            color = "#d4d4d4"
        
        # Adicionar texto com cor
        self.console_text.setTextColor(Qt.white)
        cursor = self.console_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console_text.setTextCursor(cursor)
        
        # Escapar HTML para evitar problemas
        from html import escape
        text_escaped = escape(text)
        
        # Inserir HTML para colorir
        self.console_text.insertHtml(f'<span style="color: {color};">{text_escaped}</span>')
        self.console_text.insertPlainText("\n")
        
        # Auto-scroll para o final
        scrollbar = self.console_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_console(self):
        """Limpa o console"""
        self.console_text.clear()
        self.append_text("=== Console limpo ===")
    
    def toggle_console(self):
        """Expande ou recolhe o console"""
        if self.console_text.isVisible():
            # Recolher
            self.console_text.hide()
            self.clear_button.hide()
            self.toggle_button.setText("▶ Exibir Detalhes")
        else:
            # Expandir
            self.console_text.show()
            self.clear_button.show()
            self.toggle_button.setText("▼ Recolher Detalhes")
    
    def start_capture(self):
        """Inicia captura de stdout/stderr"""
        self.clear_console()
        self.append_text("=== Iniciando treinamento ===")
        self.append_text("")
        self.capturing = True
        
        # Garantir que console esteja expandido ao iniciar treinamento
        if not self.console_text.isVisible():
            self.toggle_console()


class ConsoleCapture(QObject):
    """Classe thread-safe para capturar prints e redirecionar para o console"""
    
    def __init__(self, console_widget, original_stream=None):
        super().__init__()
        self.console_widget = console_widget
        self.original_stream = original_stream
        self.buffer = []
    
    def write(self, text):
        """Sobrescreve write para enviar ao console de forma thread-safe"""
        try:
            if text and text.strip():  # Ignorar strings vazias
                # Usar signal para thread-safety
                if self.console_widget and hasattr(self.console_widget, 'log_signal'):
                    self.console_widget.log_signal.emit(text.rstrip())
                
                # Também enviar para o stream original (terminal)
                if self.original_stream:
                    try:
                        self.original_stream.write(text)
                        self.original_stream.flush()
                    except:
                        pass  # Ignorar erros no stream original
        except Exception as e:
            # Em caso de erro, apenas continuar
            if self.original_stream:
                try:
                    self.original_stream.write(text)
                except:
                    pass
    
    def flush(self):
        """Flush necessário para compatibilidade"""
        try:
            if self.original_stream:
                self.original_stream.flush()
        except:
            pass
