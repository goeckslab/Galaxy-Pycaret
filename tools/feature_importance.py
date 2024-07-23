import base64
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    def __init__(self, data_path, target_col, task_type, output_dir):
        self.data_path = data_path
        self.target_col = target_col
        self.task_type = task_type
        self.output_dir = output_dir
        self.data = pd.read_csv(data_path, sep=None, engine='python')
        self.target = self.data.columns[int(target_col) - 1]
        self.data.columns = self.data.columns.str.replace('.', '_')
        self.data = self.data.fillna(self.data.median(numeric_only=True))
        self.exp = ClassificationExperiment() if task_type == 'classification' else RegressionExperiment()
        self.plots = {}

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        setup_params = {
            'target': self.target,
            'session_id': 123,
            'html': True,
            'log_experiment': False,
            'system_log': False
        }
        self.exp.setup(self.data, **setup_params)

    def save_coefficients(self):
        model = self.exp.create_model('lr')
        coef_df = pd.DataFrame({
            'Feature': self.data.columns.drop(self.target),
            'Coefficient': model.coef_[0]
        })
        coef_html = coef_df.to_html(index=False)
        return coef_html

    def save_tree_importance(self):
        model = self.exp.create_model('rf')
        importances = model.feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': self.data.columns.drop(self.target),
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plot_path = os.path.join(self.output_dir, 'tree_importance.png')
        plt.savefig(plot_path)
        plt.close()
        self.plots['tree_importance'] = plot_path

    def save_shap_values(self):
        model = self.exp.create_model('lightgbm')
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(self.data.drop(columns=[self.target]))
        shap.summary_plot(shap_values, self.data.drop(columns=[self.target]), show=False)
        plt.title('Shap (LightGBM)')
        plot_path = os.path.join(self.output_dir, 'shap_summary.png')
        plt.savefig(plot_path)
        plt.close()
        self.plots['shap_summary'] = plot_path

    def generate_feature_importance(self):
        coef_html = self.save_coefficients()
        self.save_tree_importance()
        self.save_shap_values()
        return coef_html

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def save_html_report(self, coef_html):
        LOG.info("Saving HTML report")

        # Read and encode plot images
        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            encoded_image = self.encode_image_to_base64(plot_path)
            plots_html += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').capitalize()}</h3>
                <img src="data:image/png;base64,{encoded_image}" alt="{plot_name}">
            </div>
            """

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PyCaret Feature Importance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f4f4f4;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    background: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                }}
                h2 {{
                    border-bottom: 2px solid #4CAF50;
                    color: #4CAF50;
                    padding-bottom: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .plot {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>PyCaret Feature Importance Report</h1>
                <h2>Coefficients(based on a trained Logistic Regression Model)</h2>
                <div>{coef_html}</div>
                <h2>Plots</h2>
                {plots_html}
            </div>
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "feature_importance_report.html"), "w") as file:
            file.write(html_content)

    def run(self):
        LOG.info("Running feature importance analysis")
        self.setup_pycaret()
        coef_html = self.generate_feature_importance()
        self.save_html_report(coef_html)
        LOG.info("Feature importance analysis completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--target_col", type=int, help="Index of the target column (1-based)")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], help="Task type: classification or regression")
    parser.add_argument("--output_dir", type=str, help="Directory to save the outputs")
    args = parser.parse_args()

    analyzer = FeatureImportanceAnalyzer(args.data_path, args.target_col, args.task_type, args.output_dir)
    analyzer.run()
