import argparse
import logging
import os
import shutil
import tempfile

from generate_md import generate_report_from_path

import h5py

import joblib

import pandas as pd

from pycaret.classification import ClassificationExperiment
from pycaret.classification import predict_model as classify_predict
from pycaret.regression import RegressionExperiment
from pycaret.regression import predict_model as regress_predict

from sklearn.metrics import average_precision_score

LOG = logging.getLogger(__name__)


class PyCaretModelEvaluator:
    def __init__(self, model_path, task, target):
        self.model_path = model_path
        self.task = task.lower()
        self.model = self.load_h5_model()
        self.target = target if target != "None" else None

    def load_h5_model(self):
        """Load a PyCaret model from an HDF5 file."""
        with h5py.File(self.model_path, 'r') as f:
            model_bytes = bytes(f['model'][()])
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(model_bytes)
                temp_file.seek(0)
                loaded_model = joblib.load(temp_file.name)
        return loaded_model

    def evaluate(self, data_path):
        """Evaluate the model using the specified data."""
        raise NotImplementedError("Subclasses must implement this method")


class ClassificationEvaluator(PyCaretModelEvaluator):
    def evaluate(self, data_path):
        metrics = None
        plot_paths = {}
        data = pd.read_csv(data_path, engine='python', sep=None)
        if self.target:
            exp = ClassificationExperiment()
            names = data.columns.to_list()
            LOG.error(f"Column names: {names}")
            target_index = int(self.target)-1
            target_name = names[target_index]
            exp.setup(data, target=target_name, test_data=data, index=False)
            exp.add_metric(id='PR-AUC-Weighted',
                            name='PR-AUC-Weighted',
                            target='pred_proba',
                            score_func=average_precision_score,
                            average='weighted'
                            )
            predictions = exp.predict_model(self.model)
            metrics = exp.pull()
            plots = ['confusion_matrix', 'auc', 'threshold', 'pr',
                 'error', 'class_report', 'learning', 'calibration',
                 'vc', 'dimension', 'manifold', 'rfe', 'feature',
                 'feature_all']
            for plot_name in plots:
                try:
                    if plot_name == 'auc' and not exp.is_multiclass:
                        plot_path = exp.plot_model(self.model,
                                                    plot=plot_name,
                                                    save=True,
                                                    plot_kwargs={
                                                        'micro': False,
                                                        'macro': False,
                                                        'per_class': False,
                                                        'binary': True
                                                        }
                                                    )
                        plot_paths[plot_name] = plot_path
                        continue

                    plot_path = exp.plot_model(self.model,
                                                    plot=plot_name, save=True)
                    plot_paths[plot_name] = plot_path
                except Exception as e:
                    LOG.error(f"Error generating plot {plot_name}: {e}")
                    continue

        else:
            LOG.error(dir(self.model))
            exp = ClassificationExperiment()
            exp.setup(data, target=None, test_data=data, index=False)
            predictions = exp.predict_model(self.model, data=data)
        
        return predictions, metrics, plot_paths


class RegressionEvaluator(PyCaretModelEvaluator):
    def evaluate(self, data_path):
        metrics = None
        plot_paths = {}
        data = pd.read_csv(data_path)
        if self.target:
            names = data.columns.to_list()
            target_index = int(self.target)-1
            target_name = names[target_index]
            exp = RegressionExperiment()
            exp.setup(data, target=target_name, test_data=data)
            predictions = exp.predict_model(self.model)
            metrics = exp.pull()
            plots = ['residuals', 'error', 'cooks',
                 'learning', 'vc', 'manifold',
                 'rfe', 'feature', 'feature_all']
            for plot_name in plots:
                try:
                    plot_path = exp.plot_model(self.model,
                                            plot=plot_name, save=True)
                    plot_paths[plot_name] = plot_path
                except Exception as e:
                    LOG.error(f"Error generating plot {plot_name}: {e}")
                    continue
        else:
            predictions = regress_predict(self.model, data=data)
        
        return predictions, metrics, plot_paths

def generate_md(plots, metrics):
    LOG.error(plots)
    if not os.path.exists("markdown"):
        basepath = os.mkdir("markdown")
    if not os.path.exists("markdown/Evaluation"):
        basepath = os.mkdir("markdown/Evaluation")
    for plot, path in plots.items():
        shutil.copy(path, "markdown/Evaluation/")
    LOG.error(type(metrics))
    metrics.to_csv("markdown/Evaluation/metrics.csv", index=False)
    generate_report_from_path("markdown", "evaluation.pdf", format="pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PyCaret model stored in HDF5 format.")
    parser.add_argument("--model_path",
                        type=str,
                        help="Path to the HDF5 model file.")
    parser.add_argument("--data_path",
                        type=str,
                        help="Path to the evaluation data CSV file.")
    parser.add_argument("--task",
                        type=str,
                        choices=["classification", "regression"],
                        help="Specify the task: classification or regression.")
    parser.add_argument("--target",
                        default=None,
                        help="Column number of the target")
    args = parser.parse_args()

    if args.task == "classification":
        evaluator = ClassificationEvaluator(args.model_path, args.task, args.target)
    elif args.task == "regression":
        evaluator = RegressionEvaluator(args.model_path, args.task)
    else:
        raise ValueError(
            "Unsupported task type. Use 'classification' or 'regression'.")

    predictions, metrics, plots = evaluator.evaluate(args.data_path)

    predictions.to_csv("predictions.csv", index=False)
    if args.target:
        generate_md(plots, metrics)
