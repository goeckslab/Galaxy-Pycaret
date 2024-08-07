import os
from pycaret.classification import setup, create_model, pull
from pycaret.regression import setup as reg_setup, create_model as reg_create_model
import pandas as pd
import numpy as np
import json
import argparse

class MultiExperimentRunner:
    def __init__(self, data, target, model_name, task_type, output_dir, n_experiments=10, custom_params=None):
        self.data = data
        self.target = target
        self.model_name = model_name
        self.task_type = task_type
        self.n_experiments = n_experiments
        self.custom_params = custom_params or {}
        self.results = []
        self.output_dir = output_dir

    def run_experiments(self):
        for i in range(self.n_experiments):
            print(f"Running experiment {i+1}/{self.n_experiments}...")
            session_id = np.random.randint(1, 100000)
            self.setup_params = {
            'target': self.target,
            'session_id': session_id,
            'html': True,
            'log_experiment': False,
            'system_log': False
        }
            if self.task_type == 'classification':
                exp=setup(self.data, **self.setup_params)
                model = exp.create_model(self.model_name, **self.custom_params)
                results = pull()
                stat_dict = {
                    'Accuracy': results.loc[0, 'Accuracy'],
                    'AUC': results.loc[0, 'AUC'],
                    'Recall': results.loc[0, 'Recall'],
                    'Prec.': results.loc[0, 'Prec.'],
                    'F1': results.loc[0, 'F1'],
                    'Session ID': session_id
                }
            else:
                reg_exp=reg_setup(self.data, **self.setup_params)
                model = reg_exp.reg_create_model(self.model_name, **self.custom_params)
                results = pull()
                stat_dict = {
                    'MAE': results.loc[0, 'MAE'],
                    'MSE': results.loc[0, 'MSE'],
                    'RMSE': results.loc[0, 'RMSE'],
                    'R2': results.loc[0, 'R2'],
                    'RMSLE': results.loc[0, 'RMSLE'],
                    'MAPE': results.loc[0, 'MAPE'],
                    'Session ID': session_id
                }
            
            self.results.append(stat_dict)

    def save_results_to_csv(self, output_file='experiment_results.csv'):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(
                self.output_dir, "result.csv"), index=False)
        print(f"Results saved to {output_file}")

    def generate_report(self, output_file='experiment_report.html'):
        df = pd.DataFrame(self.results)
        summary = df.describe().loc[['mean', 'std', 'min', 'max']]
        html_content = f"""
        <html>
        <head><title>Experiment Report</title></head>
        <body>
        <h1>Model: {self.model_name}</h1>
        <h2>Task Type: {self.task_type.capitalize()}</h2>
        <h3>Number of Experiments: {self.n_experiments}</h3>
        <h3>Summary Statistics:</h3>
        {summary.to_html()}
        </body>
        </html>
        """
        with open(os.path.join(
                self.output_dir, "result.html"), 'w') as f:
            f.write(html_content)
        print(f"Report saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multiple experiments and generate a report.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the input data file (CSV).")
    parser.add_argument('--target_column', type=str, required=True, help="Name of the target column.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name to use for the experiment.")
    parser.add_argument('--task_type', type=str, required=True, choices=['classification', 'regression'], help="Type of task: classification or regression.")
    parser.add_argument('--n_experiments', type=int, default=10, help="Number of experiments to run.")
    parser.add_argument('--custom_params', type=str, default="{}", required=False, help="Custom hyperparameters in JSON format.")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory.")

    args = parser.parse_args()

    data = pd.read_csv(args.data_file, sep=None, engine='python')
    data = data.apply(pd.to_numeric, errors='coerce')
    names = data.columns.to_list()
    target = names[int(args.target_column)-1]
    model_name = args.model_name
    task_type = args.task_type
    n_experiments = args.n_experiments
    custom_params = None if args.custom_params == '{}' else json.loads(args.custom_params)

    runner = MultiExperimentRunner(data, target, model_name, task_type, args.output_dir, n_experiments, custom_params)
    runner.run_experiments()
    runner.save_results_to_csv()
    runner.generate_report()