import os
from modelSummary.ModelSummary import ModelSummary
from fpdf import FPDF  # Biblioteca para gerar PDFs
import pandas as pd
import csv

class RelatorioDosModelos:
    def __init__(self, model, models_and_results, metrics, model_type="Old"):
        """
        model_type: "Old" para modelos originais ou "New" para modelos corrigidos
        """
        self.model = model
        self.models_and_results = models_and_results
        self.metrics = metrics
        # Definir base_dir como root/Linux
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Definir subpasta baseada no tipo de modelo
        if model_type == "New":
            sub_folder = "DadosDoPostreino/ModelosNew"
        else:
            sub_folder = "DadosDoPostreino/ModelosOlds"
        
        self.output_directoryCSV = os.path.join(self.base_dir, sub_folder, "RelatorioDosModelos(CSV)")
        self.output_directoryPDF = os.path.join(self.base_dir, sub_folder, "RelatorioDosModelos(PDF)")
        self.output_directoryTXT = os.path.join(self.base_dir, sub_folder, "RelatorioDosModelos(TXT)")
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directoryCSV, exist_ok=True)
        os.makedirs(self.output_directoryPDF, exist_ok=True)
        os.makedirs(self.output_directoryTXT, exist_ok=True)

    def save_reports_CSV_PDF(self):
        for model_name, (csv_data, model_summary) in self.models_and_results.items():
            # Save CSV
            csv_filename = os.path.join(self.output_directoryCSV, f"{model_name}_results.csv")
            csv_data.to_csv(csv_filename, index=False)
            
            # Save PDF
            pdf_filename = os.path.join(self.output_directoryPDF, f"{model_name}_summary.pdf")
            self.save_model_summary_pdf(self.model, model_summary, pdf_filename)
            
            # Save metrics
            metrics_txt_filename = os.path.join(self.output_directoryTXT, f"{model_name}_metrics.txt")
            self.save_metrics_txt(metrics_txt_filename)

            metrics_pdf_filename = os.path.join(self.output_directoryPDF, f"{model_name}_metrics.pdf")
            self.save_metrics_pdf(metrics_pdf_filename)

    def save_reports_CSV(self):
        for model_name, csv_data in self.models_and_results.items():
            # Save CSV
            csv_filename = os.path.join(self.output_directoryCSV, f"{model_name}_results.csv")
            csv_data[0].to_csv(csv_filename, index=False)
    
    def save_reports_CSV_KNN_ARIMA_RF(self,filename):
        for model_name, csv_data in self.models_and_results.items():
            # Save CSV
            csv_filename = os.path.join(self.output_directory, filename)
            csv_data[0].to_csv(csv_filename, index=False)

    def save_model_summary_pdf(self, model, model_summary, filename):
        # Extrair os parâmetros necessários do model_summary
        model = model_summary.model
        X_test = model_summary.X_test
        y_test = model_summary.y_test

        # Instanciar ModelSummary e salvar o sumário do modelo em PDF
        model_summary_instance = ModelSummary(model, filename, X_test, y_test)
        model_summary_instance.save_model_summary()

    def save_metrics_txt(self, filename):
        with open(filename, 'w') as f:
            for metric_name, metric_value in self.metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")


    def save_metrics_pdf_KNN_ARIMA_RF(self, filename):
         metrics_pdf_filename = os.path.join(self.output_directoryPDF, filename)
         self.save_metrics_pdf(metrics_pdf_filename)

    def save_metrics_pdf(self, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="Model Metrics", ln=True, align='C')
        
        for metric_name, metric_value in self.metrics.items():
            pdf.cell(200, 10, txt=f"{metric_name}: {metric_value}", ln=True, align='L')
        
        pdf.output(filename)

    def save_shared_metrics(self, shared_csv_file="shared_model_metrics.csv"):
        csv_file_path = os.path.join(self.output_directoryCSV, shared_csv_file)
        file_exists = os.path.isfile(csv_file_path)

        existing_models = set()

        # Se o arquivo existir, leia os modelos já presentes
        if file_exists:
            with open(csv_file_path, mode='r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    existing_models.add(row['Model Name'])

        # Verifique se o modelo atual já está no arquivo
        current_model_name = self.model.__class__.__name__
        if current_model_name not in existing_models:
            with open(csv_file_path, mode='a', newline='') as csv_file:
                fieldnames = ['Model Name', 'MSE', 'RMSE', 'R²']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()  # Escreva o cabeçalho apenas se o arquivo não existir

                writer.writerow({
                    'Model Name': current_model_name,
                    'MSE': self.metrics.get('MSE', 'N/A'),
                    'RMSE': self.metrics.get('RMSE', 'N/A'),
                    'R²': self.metrics.get('R²', 'N/A')
                })

    def save_shared_metrics_list(self, mse_list, rmse_list, modelname, shared_csv_file="shared_model_metrics_list.csv"):
        # Diretório de saída
        print("MODELLL Name",modelname)
        
        # Caminho completo do arquivo CSV
        csv_file_path = os.path.join(self.output_directoryCSV, shared_csv_file)
        file_exists = os.path.isfile(csv_file_path)

        # Se o arquivo existir, leia os modelos já presentes
        existing_models = set()
        if file_exists:
            with open(csv_file_path, mode='r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    existing_models.add(row['Model Name'])

        # Verifique se o modelo atual já está no arquivo
        if (modelname not in existing_models):
            print("MODELLL Name",modelname)
            with open(csv_file_path, mode='a', newline='') as csv_file:
                fieldnames = ['Model Name', 'MSE List', 'RMSE List']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                # Se o arquivo não existir, escreva o cabeçalho
                if not file_exists:
                    writer.writeheader()

                # Converta as listas para strings para serem gravadas no CSV
                mse_list_str = ','.join(map(str, mse_list))
                rmse_list_str = ','.join(map(str, rmse_list))

                # Escreva os dados no arquivo CSV
                writer.writerow({
                    'Model Name': modelname,
                    'MSE List': mse_list_str,
                    'RMSE List': rmse_list_str
                })
    
    def save_shared_difference_list(self, difference, modelname, shared_csv_file="shared_model_difference_list.csv"):
        # Caminho completo do arquivo CSV
        csv_file_path = os.path.join(self.output_directoryCSV, shared_csv_file)
        file_exists = os.path.isfile(csv_file_path)

        # Se o arquivo existir, leia os modelos já presentes
        existing_models = set()
        if file_exists:
            with open(csv_file_path, mode='r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    existing_models.add(row['Model Name'])

        # Verifique se o modelo atual já está no arquivo
        if modelname not in existing_models:
            with open(csv_file_path, mode='a', newline='') as csv_file:
                fieldnames = ['Model Name', 'Difference']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                # Se o arquivo não existir, escreva o cabeçalho
                if not file_exists:
                    writer.writeheader()

                # Converta a lista de diferenças para uma string (achatar array se necessário)
                if hasattr(difference, 'flatten'):
                    difference_flat = difference.flatten()
                else:
                    difference_flat = difference
                difference_str = ','.join(map(str, difference_flat))

                # Escreva os dados no arquivo CSV
                writer.writerow({
                    'Model Name': modelname,
                    'Difference': difference_str
                })