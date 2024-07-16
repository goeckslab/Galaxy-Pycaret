import logging

from base_model_trainer import BaseModelTrainer
from pycaret.classification import ClassificationExperiment
from dashboard import generate_classifier_explainer_dashboard

LOG = logging.getLogger(__name__)


class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(self, input_file, target_col, output_dir, **kwargs):
        super().__init__(input_file, target_col, output_dir, **kwargs)
        self.exp = ClassificationExperiment()

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        dashboard = generate_classifier_explainer_dashboard(self.exp,
                                                            self.best_model)
        dashboard.save_html("dashboard.html")

    def generate_plots(self):
        LOG.info("Generating and saving plots")
        plots = ['auc', 'confusion_matrix', 'threshold', 'pr',
                 'error', 'class_report', 'learning', 'calibration',
                 'vc', 'dimension', 'manifold', 'rfe', 'feature',
                 'feature_all']
        for plot_name in plots:
            try:
                plot_path = self.exp.plot_model(self.best_model,
                                                plot=plot_name, save=True)
                self.plots[plot_name] = plot_path
            except Exception as e:
                LOG.error(f"Error generating plot {plot_name}: {e}")
                continue
