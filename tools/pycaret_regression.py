import logging

from base_model_trainer import BaseModelTrainer

from dashboard import generate_regression_explainer_dashboard

from pycaret.regression import RegressionExperiment

from utils import add_hr_to_html, add_plot_to_html

LOG = logging.getLogger(__name__)


class RegressionModelTrainer(BaseModelTrainer):
    def __init__(
            self,
            input_file,
            target_col,
            output_dir,
            task_type,
            random_seed,
            test_file=None,
            **kwargs):
        super().__init__(
            input_file,
            target_col,
            output_dir,
            task_type,
            random_seed,
            test_file,
            **kwargs)
        self.exp = RegressionExperiment()

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        dashboard = generate_regression_explainer_dashboard(self.exp,
                                                            self.best_model)
        dashboard.save_html("dashboard.html")

    def generate_plots(self):
        LOG.info("Generating and saving plots")
        plots = ['residuals', 'error', 'cooks',
                 'learning', 'vc', 'manifold',
                 'rfe', 'feature', 'feature_all']
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

        from explainerdashboard import RegressionExplainer

        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed

        explainer = RegressionExplainer(self.best_model, X_test, y_test)
        self.expaliner = explainer
        plots_explainer_html = ""

        try:
            fig_importance = explainer.plot_importances()
            plots_explainer_html += add_plot_to_html(fig_importance)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot importance: {e}")

        try:
            fig_importance_permutation = \
                explainer.plot_importances_permutation(
                    kind="permutation")
            plots_explainer_html += add_plot_to_html(
                fig_importance_permutation)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot importance permutation: {e}")

        try:
            for feature in self.features_name:
                fig_shap = explainer.plot_pdp(feature)
                plots_explainer_html += add_plot_to_html(fig_shap)
                plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot shap dependence: {e}")

        # try:
        #     for feature in self.features_name:
        #         fig_interaction = explainer.plot_interaction(col=feature)
        #         plots_explainer_html += add_plot_to_html(fig_interaction)
        # except Exception as e:
        #     LOG.error(f"Error generating plot shap interaction: {e}")

        try:
            for feature in self.features_name:
                fig_interactions_importance = \
                    explainer.plot_interactions_importance(
                        col=feature)
                plots_explainer_html += add_plot_to_html(
                    fig_interactions_importance)
                plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot shap summary: {e}")

        # Regression specific plots
        try:
            fig_pred_actual = explainer.plot_predicted_vs_actual()
            plots_explainer_html += add_plot_to_html(fig_pred_actual)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot prediction vs actual: {e}")

        try:
            fig_residuals = explainer.plot_residuals()
            plots_explainer_html += add_plot_to_html(fig_residuals)
            plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot residuals: {e}")

        try:
            for feature in self.features_name:
                fig_residuals_vs_feature = \
                    explainer.plot_residuals_vs_feature(feature)
                plots_explainer_html += add_plot_to_html(
                    fig_residuals_vs_feature)
                plots_explainer_html += add_hr_to_html()
        except Exception as e:
            LOG.error(f"Error generating plot residuals vs feature: {e}")

        self.plots_explainer_html = plots_explainer_html
