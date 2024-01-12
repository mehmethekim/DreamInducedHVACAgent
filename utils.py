import os
def initialize_logging(model_name):
    models_dir = "models/"+ model_name
    logs_dir = "logs/"+ model_name
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir,models_dir
