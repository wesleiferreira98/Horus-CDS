from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from contextlib import redirect_stdout
class ModelSummary():
     def __init__(self, model,filename,X_test,y_test):
        self.model= model
        self.filename = filename
        self.X_test =X_test,
        self.y_test = y_test
        
     def save_model_summary(self):
        # Open a canvas
        c = canvas.Canvas(self.filename, pagesize=letter)
        
        # Before evaluating the model
        c.drawString(100, 750, f"Shape of X_test: {self.X_test}")
        c.drawString(100, 730, f"Shape of y_test: {self.y_test}")
        c.drawString(100, 710, "Model summary:")
        
        # Capture the model summary as a string
        
        
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.model.summary()
        model_summary = buffer.getvalue().split('\n')

        # Write model summary to PDF
        y = 690
        for line in model_summary:
            c.drawString(100, y, line)
            y -= 10  # Move to next line

        # Save the canvas to the PDF file
        c.save()