import logging

from base_model_trainer import BaseModelTrainer

from dashboard import generate_classifier_explainer_dashboard

from pycaret.classification import ClassificationExperiment

from utils import add_hr_to_html, add_plot_to_html

LOG = logging.getLogger(__name__)


class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(
            self,
            input_file,
            target_col,
            output_dir,
            task_type,
            random_seed,
            **kwargs):
        super().__init__(
            input_file,
            target_col,
            output_dir,
            task_type,
            random_seed,
            **kwargs)
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

    def generate_plots_explainer(self):
        LOG.info("Generating and saving plots from explainer")

        from explainerdashboard import ClassifierExplainer

        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed

        explainer = ClassifierExplainer(self.best_model, X_test, y_test)
        self.expaliner = explainer
        plots_explainer_html = ""

        try:
            fig_importance = explainer.plot_importances()
            plots_explainer_html += add_plot_to_html(fig_importance)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot importance(mean shap): {e}")

        try:
            fig_importance_perm = explainer.plot_importances(
                kind="permutation")
            plots_explainer_html += add_plot_to_html(fig_importance_perm)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot importance(permutation): {e}")

        # Uncomment and adjust if needed
        # try:
        #     fig_shap = explainer.plot_shap_summary()
        #     plots_explainer_html += add_plot_to_html(fig_shap,
        #       include_plotlyjs=False)
        # except Exception as e:
        #     LOG.error(f"Error generating plot shap: {e}")

        # try:
        #     fig_contributions = explainer.plot_contributions(
        #       index=0)
        #     plots_explainer_html += add_plot_to_html(
        #       fig_contributions, include_plotlyjs=False)
        # except Exception as e:
        #     LOG.error(f"Error generating plot contributions: {e}")

        # try:
        #     for feature in self.features_name:
        #         fig_dependence = explainer.plot_dependence(col=feature)
        #         plots_explainer_html += add_plot_to_html(fig_dependence)
        # except Exception as e:
        #     LOG.error(f"Error generating plot dependencies: {e}")

        try:
            for feature in self.features_name:
                fig_pdp = explainer.plot_pdp(feature)
                plots_explainer_html += add_plot_to_html(fig_pdp)
                plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot pdp: {e}")

        try:
            for feature in self.features_name:
                fig_interaction = explainer.plot_interaction(
                    col=feature, interact_col=feature)
                plots_explainer_html += add_plot_to_html(fig_interaction)
        except Exception as e:
            LOG.error(f"Error generating plot interactions: {e}")

        try:
            for feature in self.features_name:
                fig_interactions_importance = \
                    explainer.plot_interactions_importance(
                        col=feature)
                plots_explainer_html += add_plot_to_html(
                    fig_interactions_importance)
                plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot interactions importance: {e}")

        # try:
        #     for feature in self.features_name:
        #         fig_interactions_detailed = \
        #           explainer.plot_interactions_detailed(
        #               col=feature)
        #         plots_explainer_html += add_plot_to_html(
        #           fig_interactions_detailed)
        # except Exception as e:
        #     LOG.error(f"Error generating plot interactions detailed: {e}")

        try:
            fig_precision = explainer.plot_precision()
            plots_explainer_html += add_plot_to_html(fig_precision)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot precision: {e}")

        try:
            fig_cumulative_precision = explainer.plot_cumulative_precision()
            plots_explainer_html += add_plot_to_html(fig_cumulative_precision)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot cumulative precision: {e}")

        try:
            fig_classification = explainer.plot_classification()
            plots_explainer_html += add_plot_to_html(fig_classification)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot classification: {e}")

        try:
            fig_confusion_matrix = explainer.plot_confusion_matrix()
            plots_explainer_html += add_plot_to_html(fig_confusion_matrix)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot confusion matrix: {e}")

        try:
            fig_lift_curve = explainer.plot_lift_curve()
            plots_explainer_html += add_plot_to_html(fig_lift_curve)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot lift curve: {e}")

        try:
            fig_roc_auc = explainer.plot_roc_auc()
            plots_explainer_html += add_plot_to_html(fig_roc_auc)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot roc auc: {e}")

        try:
            fig_pr_auc = explainer.plot_pr_auc()
            plots_explainer_html += add_plot_to_html(fig_pr_auc)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot pr auc: {e}")

        self.plots_explainer_html = plots_explainer_html
