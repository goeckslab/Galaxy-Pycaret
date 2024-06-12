import sys
import logging

from pycaret_classification import ModelTrainer

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

input_file = sys.argv[1]
target_col = sys.argv[2]
output_dir = sys.argv[3]
model_type = sys.argv[4]

if model_type == "classification":
    trainer = ModelTrainer(input_file, target_col, output_dir)
    trainer.run()