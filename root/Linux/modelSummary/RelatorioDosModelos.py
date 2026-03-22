import os
from modelSummary.ModelSummary import ModelSummary
from fpdf import FPDF  # Biblioteca para gerar PDFs
import pandas as pd
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

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

    def save_roc_pr_curves_regression(self, y_true_categories, y_pred_values, modelname, thresholds, class_labels=None):
        if class_labels is None:
            class_labels = ['ilegal', 'suspeito', 'válido']

        if thresholds is None or len(thresholds) != 2:
            raise ValueError("thresholds deve conter exatamente dois valores: limite_ilegal_suspeito e limite_suspeito_valido")

        low_threshold, high_threshold = thresholds
        y_true_categories = np.asarray(y_true_categories)
        y_pred_values = np.asarray(y_pred_values).reshape(-1)

        if len(y_true_categories) != len(y_pred_values):
            raise ValueError("y_true_categories e y_pred_values devem ter o mesmo tamanho")

        valid_mask = np.isin(y_true_categories, class_labels)
        y_true_categories = y_true_categories[valid_mask]
        y_pred_values = y_pred_values[valid_mask]

        if len(y_true_categories) == 0:
            raise ValueError("Nenhuma categoria válida encontrada para gerar ROC/PR")

        center_threshold = (low_threshold + high_threshold) / 2.0

        illegal_score = low_threshold - y_pred_values
        suspeito_score = -np.abs(y_pred_values - center_threshold)
        valido_score = y_pred_values - high_threshold

        raw_scores = np.column_stack([illegal_score, suspeito_score, valido_score]).astype(float)
        raw_scores = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
        exp_scores = np.exp(raw_scores)
        class_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y_true_bin = label_binarize(y_true_categories, classes=class_labels)

        output_curve_dir = os.path.join(
            self.base_dir,
            "DadosDoPostreino/ModelosNew" if "New" in self.output_directoryCSV else "DadosDoPostreino/ModelosOlds",
            "CurvasROC_PR"
        )
        os.makedirs(output_curve_dir, exist_ok=True)

        roc_results = {}
        pr_results = {}
        valid_indices = []

        fig_roc = plt.figure(figsize=(10, 8))
        for idx, label in enumerate(class_labels):
            if len(np.unique(y_true_bin[:, idx])) < 2:
                roc_results[label] = None
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], class_scores[:, idx])
            roc_auc = auc(fpr, tpr)
            roc_results[label] = roc_auc
            valid_indices.append(idx)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curvas ROC One-vs-Rest - {modelname}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_curve_dir, f'roc_curve_{modelname}.jpg')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close(fig_roc)

        fig_pr = plt.figure(figsize=(10, 8))
        for idx, label in enumerate(class_labels):
            if len(np.unique(y_true_bin[:, idx])) < 2:
                pr_results[label] = None
                continue
            precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], class_scores[:, idx])
            ap = average_precision_score(y_true_bin[:, idx], class_scores[:, idx])
            pr_results[label] = ap
            plt.plot(recall, precision, lw=2, label=f'{label} (AP={ap:.4f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curvas Precision-Recall One-vs-Rest - {modelname}')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join(output_curve_dir, f'pr_curve_{modelname}.jpg')
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close(fig_pr)

        macro_roc = np.mean([roc_results[label] for label in class_labels if roc_results[label] is not None]) if valid_indices else None
        macro_pr = np.mean([pr_results[label] for label in class_labels if pr_results[label] is not None]) if valid_indices else None

        txt_path = os.path.join(output_curve_dir, f'roc_pr_metrics_{modelname}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"ROC/PR METRICS - {modelname}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Threshold ilegal/suspeito: {low_threshold}\n")
            f.write(f"Threshold suspeito/valido: {high_threshold}\n\n")
            for label in class_labels:
                f.write(f"{label}:\n")
                f.write(f"  ROC AUC: {roc_results[label] if roc_results[label] is not None else 'N/A'}\n")
                f.write(f"  Average Precision: {pr_results[label] if pr_results[label] is not None else 'N/A'}\n")
            f.write("\n")
            f.write(f"Macro ROC AUC: {macro_roc if macro_roc is not None else 'N/A'}\n")
            f.write(f"Macro Average Precision: {macro_pr if macro_pr is not None else 'N/A'}\n")

        return {
            'roc_path': roc_path,
            'pr_path': pr_path,
            'txt_path': txt_path,
            'roc_auc': roc_results,
            'average_precision': pr_results,
            'macro_roc_auc': macro_roc,
            'macro_average_precision': macro_pr,
        }
