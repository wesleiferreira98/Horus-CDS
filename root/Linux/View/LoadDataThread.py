from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
import time
class LoadDataThread(QThread):
    update_progress = pyqtSignal(int)
    data_loaded = pyqtSignal(pd.DataFrame)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        chunk_size = 10000  # Ajuste conforme necessário
        chunks = []
        total_size = sum(1 for _ in open(self.file_path)) - 1  # Total de linhas menos o cabeçalho
        rows_loaded = 0

        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            chunks.append(chunk)
            rows_loaded += len(chunk)
            progress_percent = int((rows_loaded / total_size) * 100)
            self.update_progress.emit(progress_percent)
            time.sleep(0.1)  # Permitir que a UI processe eventos

        data = pd.concat(chunks, ignore_index=True)
        self.data_loaded.emit(data)