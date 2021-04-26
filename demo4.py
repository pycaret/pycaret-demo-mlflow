from pycaret.regression import *
from pycaret.datasets import get_data
import mlflow

def run():
    data = get_data('insurance')
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    s = setup(data, target = 'charges', session_id = 123, silent = True, 
                log_experiment = True, experiment_name = 'insurance_demo4', log_plots = True)
    models = ['lr', 'dt', 'lightgbm', 'rf']
    all_models = [create_model(i) for i in models]
        
if __name__ == "__main__":
    run()

# run mlflow backend
# mlflow ui --backend-store-uri sqlite:///mlruns.db