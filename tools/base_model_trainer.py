import base64
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class BaseModelTrainer:

    def __init__(self, input_file, target_col, output_dir, **kwargs):
        self.exp = None  # This will be set in the subclass
        self.input_file = input_file
        self.target_col = target_col
        self.output_dir = output_dir
        self.data = None
        self.target = None
        self.best_model = None
        self.results = None
        self.plots = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.setup_params = {}

        LOG.info(f"Model kwargs: {self.__dict__}")

    def load_data(self):
        LOG.info(f"Loading data from {self.input_file}")
        self.data = pd.read_csv(self.input_file, sep=None, engine='python')
        names = self.data.columns.to_list()
        self.target = names[int(self.target_col)-1]
        self.data = self.data.fillna(self.data.median(numeric_only=True))
        self.data.columns = self.data.columns.str.replace('.', '_')

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        self.setup_params = {
            'target': self.target,
            'session_id': 123,
            'html': True,
            'log_experiment': False,
            'system_log': False
        }

        if hasattr(self, 'train_size') and self.train_size is not None:
            self.setup_params['train_size'] = self.train_size

        if hasattr(self, 'normalize') and self.normalize is not None:
            self.setup_params['normalize'] = self.normalize

        if hasattr(self, 'feature_selection') and \
                self.feature_selection is not None:
            self.setup_params['feature_selection'] = self.feature_selection

        if hasattr(self, 'cross_validation') and \
                self.cross_validation is not None \
                and self.cross_validation is False:
            self.setup_params['cross_validation'] = self.cross_validation

        if hasattr(self, 'cross_validation') and \
                self.cross_validation is not None:
            if hasattr(self, 'cross_validation_folds'):
                self.setup_params['fold'] = self.cross_validation_folds

        if hasattr(self, 'remove_outliers') and \
                self.remove_outliers is not None:
            self.setup_params['remove_outliers'] = self.remove_outliers

        if hasattr(self, 'remove_multicollinearity') and \
                self.remove_multicollinearity is not None:
            self.setup_params['remove_multicollinearity'] = \
                self.remove_multicollinearity

        if hasattr(self, 'polynomial_features') and \
                self.polynomial_features is not None:
            self.setup_params['polynomial_features'] = self.polynomial_features

        if hasattr(self, 'fix_imbalance') and \
                self.fix_imbalance is not None:
            self.setup_params['fix_imbalance'] = self.fix_imbalance

        LOG.info(self.setup_params)
        self.exp.setup(self.data, **self.setup_params)

    def train_model(self):
        LOG.info("Training and selecting the best model")
        self.best_model = self.exp.compare_models()
        self.results = self.exp.pull()

    def save_model(self):
        LOG.info("Saving the model")
        self.exp.save_model(self.best_model, "model")

    def generate_plots(self):
        raise NotImplementedError("Subclasses should implement this method")

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def save_html_report(self):
        LOG.info("Saving HTML report")

        model_name = type(self.best_model).__name__
        excluded_params = ['html', 'log_experiment', 'system_log']
        filtered_setup_params = {
            k: v
            for k, v in self.setup_params.items() if k not in excluded_params
        }
        setup_params_table = pd.DataFrame(
            list(filtered_setup_params.items()),
            columns=['Parameter', 'Value'])
        # Save model summary
        best_model_params = pd.DataFrame(
            self.best_model.get_params().items(),
            columns=['Parameter', 'Value'])
        best_model_params.to_csv(
            os.path.join(self.output_dir, 'best_model.csv'),
            index=False)

        # Save comparison results
        self.results.to_csv(os.path.join(
            self.output_dir, "comparison_results.csv"))

        # Read and encode plot images
        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            encoded_image = self.encode_image_to_base64(plot_path)
            plots_html += f"""
            <div class="plot">
                <h3>{plot_name.capitalize()}</h3>
                <img src="data:image/png;base64,
                {encoded_image}" alt="{plot_name}">
            </div>
            """

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width,
            initial-scale=1.0">
            <title>PyCaret Model Training Report</title>
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
                <h1>PyCaret Model Training Report</h1>
                <h2>Setup Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    {setup_params_table.to_html(index=False,
                                            header=False, classes='table')}
                </table>
                <h2>Best Model: {model_name}</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    {best_model_params.to_html(index=False,
                                            header=False, classes='table')}
                </table>
                <h2>Comparison Results</h2>
                <table>
                    {self.results.to_html(index=False,
                                        classes='table')}
                </table>
                <h2>Plots</h2>
                {plots_html}
            </div>
        </body>
        </html>
        """

        with open(os.path.join(
                self.output_dir, "comparison_result.html"), "w") as file:
            file.write(html_content)

    def save_dashboard(self):
        raise NotImplementedError("Subclasses should implement this method")

    def run(self):
        self.load_data()
        self.setup_pycaret()
        self.train_model()
        self.save_model()
        self.generate_plots()
        self.save_html_report()
        self.save_dashboard()
