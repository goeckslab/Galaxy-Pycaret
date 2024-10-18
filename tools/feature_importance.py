import base64
import logging
import os

import matplotlib.pyplot as plt

import pandas as pd

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(
            self,
            task_type,
            output_dir,
            data_path=None,
            data=None,
            target_col=None):

        if data is not None:
            self.data = data
            LOG.info("Data loaded from memory")
        else:
            self.target_col = target_col
            self.data = pd.read_csv(data_path, sep=None, engine='python')
            self.data.columns = self.data.columns.str.replace('.', '_')
            self.data = self.data.fillna(self.data.median(numeric_only=True))
        self.task_type = task_type
        self.target = self.data.columns[int(target_col) - 1]
        self.exp = ClassificationExperiment() \
            if task_type == 'classification' \
            else RegressionExperiment()
        self.plots = {}
        self.output_dir = output_dir

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        setup_params = {
            'target': self.target,
            'session_id': 123,
            'html': True,
            'log_experiment': False,
            'system_log': False
        }
        LOG.info(self.task_type)
        LOG.info(self.exp)
        self.exp.setup(self.data, **setup_params)

    # def save_coefficients(self):
    #     model = self.exp.create_model('lr')
    #     coef_df = pd.DataFrame({
    #         'Feature': self.data.columns.drop(self.target),
    #         'Coefficient': model.coef_[0]
    #     })
    #     coef_html = coef_df.to_html(index=False)
    #     return coef_html

    def save_tree_importance(self):
        model = self.exp.create_model('rf')
        importances = model.feature_importances_
        processed_features = self.exp.get_config('X_transformed').columns
        LOG.debug(f"Feature importances: {importances}")
        LOG.debug(f"Features: {processed_features}")
        feature_importances = pd.DataFrame({
            'Feature': processed_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importances['Feature'],
            feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plot_path = os.path.join(
            self.output_dir,
            'tree_importance.png')
        plt.savefig(plot_path)
        plt.close() 
        self.plots['tree_importance'] = plot_path

    def save_shap_values(self):
        model = self.exp.create_model('lightgbm')
        import shap
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(
            self.data.drop(columns=[self.target]))
        shap.summary_plot(shap_values, self.data.drop(
            columns=[self.target]), show=False)
        plt.title('Shap (LightGBM)')
        plot_path = os.path.join(
            self.output_dir, 'shap_summary.png')
        plt.savefig(plot_path)
        plt.close()
        self.plots['shap_summary'] = plot_path

    def generate_feature_importance(self):
        # coef_html = self.save_coefficients()
        self.save_tree_importance()
        self.save_shap_values()

    def encode_image_to_base64(self, img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def generate_html_report(self):
        LOG.info("Generating HTML report")

        # Read and encode plot images
        plots_html = ""
        for plot_name, plot_path in self.plots.items():
            encoded_image = self.encode_image_to_base64(plot_path)
            plots_html += f"""
            <div class="plot" id="{plot_name}">
                <h2>{'Feature importance analysis from a'
                    'trained Random Forest'
                    if plot_name == 'tree_importance'
                    else 'SHAP Summary from a trained lightgbm'}</h2>
                <h3>{'Use gini impurity for'
                    'calculating feature importance for classification'
                    'and Variance Reduction for regression'
                  if plot_name == 'tree_importance'
                  else ''}</h3>
                <img src="data:image/png;base64,
                {encoded_image}" alt="{plot_name}">
            </div>
            """

        # Generate HTML content with tabs
        html_content = f"""
            <h1>PyCaret Feature Importance Report</h1>
            {plots_html}
        """

        return html_content

    def run(self):
        LOG.info("Running feature importance analysis")
        self.setup_pycaret()
        self.generate_feature_importance()
        html_content = self.generate_html_report()
        LOG.info("Feature importance analysis completed")
        return html_content


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument(
        "--data_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--target_col", type=int,
        help="Index of the target column (1-based)")
    parser.add_argument(
        "--task_type", type=str,
        choices=["classification", "regression"],
        help="Task type: classification or regression")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the outputs")
    args = parser.parse_args()

    analyzer = FeatureImportanceAnalyzer(
        args.data_path, args.target_col,
        args.task_type, args.output_dir)
    analyzer.run()
