import sys
import pandas as pd
from pycaret.classification import ClassificationExperiment
import os
import logging
from dashboard import generate_dashboard
from jinja_report.generate_report import main as generate_report 
import base64

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_file, target_col, output_dir):
        self.exp = ClassificationExperiment()
        self.input_file = input_file
        self.target_col = target_col
        self.output_dir = output_dir
        self.data = None
        self.target = None
        self.best_model = None
        self.results = None
        self.plots = {}

    def load_data(self):
        LOG.info(f"Loading data from {self.input_file}")
        self.data = pd.read_csv(self.input_file, sep=None, engine='python')
        names = self.data.columns.to_list()
        self.target = names[int(self.target_col)-1]
        self.data = self.data.fillna(self.data.median(numeric_only=True))
        self.data.columns = self.data.columns.str.replace('.', '_')

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        self.exp.setup(self.data, target=self.target, 
              session_id=123, html=True, 
              log_experiment=False, system_log=False)

    def train_model(self):
        LOG.info("Training and selecting the best model")
        self.best_model = self.exp.compare_models()
        self.results = self.exp.pull()

    def save_model(self):
        LOG.info("Saving the model")
        self.exp.save_model(self.best_model, "model.pkl")

    def generate_plots(self):
        LOG.info("Generating and saving plots")
        # Generate PyCaret plots
        plots = ['auc', 'confusion_matrix', 
                 'threshold', 
                 'pr', 'error', 
                 'class_report', 'learning', 
                 'calibration', 'vc', 
                 'dimension', 
                 'manifold', 'rfe', 
                 'feature', 'feature_all']
        for plot_name in plots:
            plot_path = self.exp.plot_model(self.best_model, plot=plot_name, 
                                            save=True)
            self.plots[plot_name] = plot_path

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
        
    def save_html_report(self):
        LOG.info("Saving HTML report")

        model_name = type(self.best_model).__name__
        
        report_data = {
            "title": "PyCaret Model Training Report",
            'Best Model': [
                {
                    'type': 'table',
                    'src': os.path.join(self.output_dir, 'best_model.csv'),
                    'label': f'Best Model: {model_name}'
                }
            ],
            'Comparison Results': [
                {
                    'type': 'table',
                    'src': os.path.join(self.output_dir, 'comparison_results.csv'),
                    'label': 'Comparison Result  <br> The scoring grid with average cross-validation scores'
                }
            ],
            "Plots": []
        }

        # Save model summary
        best_model_params = pd.DataFrame(self.best_model.get_params().items(), columns=['Parameter', 'Value'])
        best_model_params.to_csv(os.path.join(self.output_dir, 'best_model.csv'), index=False)

        # Save comparison results
        self.results.to_csv(os.path.join(self.output_dir, "comparison_results.csv"))

        # Add plots to the report data
        for plot_name, plot_path in self.plots.items():
            encoded_image = self.encode_image_to_base64(plot_path)
            report_data['Plots'].append({
                'type': 'html',
                'src': f'data:image/png;base64,{encoded_image}',
                'label': plot_name.capitalize()
            })

        generate_report(inputs=report_data, outfile=os.path.join(self.output_dir, "comparison_result.html"))

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        dashboard = generate_dashboard(self.exp, self.best_model)
        dashboard.save_html("dashboard.html")

    def run(self):
        self.load_data()
        self.setup_pycaret()
        self.train_model()
        self.save_model()
        self.generate_plots()
        self.save_html_report()
        self.save_dashboard()

if __name__ == "__main__":
    input_file = sys.argv[1]
    target_col = sys.argv[2]
    output_dir = sys.argv[3]

    trainer = ModelTrainer(input_file, target_col, output_dir)
    trainer.run()
