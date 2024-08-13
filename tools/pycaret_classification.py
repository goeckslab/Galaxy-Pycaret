import logging

from base_model_trainer import BaseModelTrainer

from dashboard import generate_classifier_explainer_dashboard

from pycaret.classification import ClassificationExperiment

LOG = logging.getLogger(__name__)


class ClassificationModelTrainer(BaseModelTrainer):
    def __init__(
            self,
            input_file,
            target_col,
            output_dir,
            task_type,
            **kwargs):
        super().__init__(
            input_file, target_col, output_dir, task_type, **kwargs)
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
        from pycaret.classification import get_config
        
        X_test = get_config('X_test')
        y_test = get_config('y_test')

        explainer = ClassifierExplainer(self.best_model, X_test, y_test)
        plots_explainer_html = ""
        
        try:
            fig_importance = explainer.plot_importances()
            fig_html = fig_importance.to_html(full_html=False, include_plotlyjs="cdn")
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot importance(mean shap): {e}")
        
        try:
            fig_importance = explainer.plot_importances(type="permutation")
            fig_html = fig_importance.to_html(full_html=False, include_plotlyjs="cdn")
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot importance(permutation): {e}")
        
        try:
            fig_shap = explainer.plot_shap_summary()
            fig_html = fig_shap.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot shap: {e}")
        
        try:
            fig_contributions = explainer.plot_contributions()
            fig_html = fig_contributions.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot contributions: {e}")
        
        try:
            fig_dependence = explainer.plot_dependencies()
            fig_html = fig_dependence.to_html(full_html=False, include_plotlyjs=False)  
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot dependencies: {e}")

        try:
            fig_pdp = explainer.plot_pdp()
            fig_html = fig_pdp.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot pdp: {e}")
        
        try:
            fig_interaction = explainer.plot_interactions()
            fig_html = fig_interaction.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot interactions: {e}")
        
        try:
            fig_interactions_importance = explainer.plot_interactions_importance()
            fig_html = fig_interactions_importance.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot interactions importance: {e}")
        
        try:
            fig_interactions_detailed = explainer.plot_interactions_detailed
            fig_html = fig_interactions_detailed.to_html(full_html=False, include_plotlyjs=False)
            plots_explainer_html += fig_html
        except Exception as e:
            LOG.error(f"Error generating plot interactions detailed: {e}")

        


