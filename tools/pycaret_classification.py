import sys
import pandas as pd
from pycaret.classification import setup, compare_models, save_model, plot_model, pull
import os
import logging
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_file, target_col, output_dir):
        self.input_file = input_file
        self.target_col = target_col
        self.output_dir = output_dir
        self.data = None
        self.target = None
        self.best_model = None
        self.results = None

    def load_data(self):
        LOG.info(f"Loading data from {self.input_file}")
        self.data = pd.read_csv(self.input_file, sep=None, engine='python')
        names = self.data.columns.to_list()
        self.target = names[int(self.target_col)-1]
        self.data = self.data.fillna(self.data.median(numeric_only=True))
        self.data.columns = self.data.columns.str.replace('.', '_')

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        setup(self.data, target=self.target, session_id=123, html=True, log_experiment=False)

    def train_model(self):
        LOG.info("Training and selecting the best model")
        original_stdout = sys.stdout
        stderror = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        with open("training.log", 'w') as f:
            sys.stdout = f
            self.best_model = compare_models()
            print(self.best_model)
            sys.stdout = original_stdout
            sys.stderr = stderror
        self.results = pull()

    def save_model(self):
        LOG.info("Saving the model")
        save_model(self.best_model, "model.pkl")

    def generate_plots(self):
        LOG.info("Generating and saving plots")
        plot_model(self.best_model, plot='auc', save=self.output_dir)
        plot_model(self.best_model, plot='confusion_matrix', save=self.output_dir)
        os.rename(os.path.join(self.output_dir, "Confusion Matrix.png"), os.path.join(self.output_dir, "Confusion_Matrix.png"))

    def save_html_report(self):
        LOG.info("Saving HTML report")
        html_content_results = f"""
        <html>
        <head>
            <title>PyCaret Model Training Report</title>
        </head>
        <body>
            <h1>Model Training Report</h1>
            <h2>Best Model</h2>
            <pre>{self.best_model}</pre>
            <h2>Comparison Results</h2>
            {self.results.to_html()}
        </body>
        </html>
        """
        with open("comparison_result.html", 'w') as f:
            f.write(html_content_results)

    def save_dashboard(self):
        LOG.info("Saving explainer dashboard")
        explainer = ClassifierExplainer(self.best_model, self.data.drop(columns=self.target), self.data[self.target])
        dashboard = ExplainerDashboard(explainer)
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
