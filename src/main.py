import argparse

from make_data_processing.data_processing import run_data_processing
# from make_monitoring.monitoring import run_model_monitoring
from make_predictions.predictions import run_make_predictions
from make_production_model.production_model import run_production_model
from make_train.train import run_train

if __name__ == "__main__":
    """
    Main Function to call function linked to arguement passed in command line.
    """
    args = argparse.ArgumentParser()
    args.add_argument("--step", default="")
    args.add_argument("--data_location", default="")
    parsed_args = args.parse_args()

    if parsed_args.step == "data_processing":
        run_data_processing()
    if parsed_args.step == "train":
        run_train()
    if parsed_args.step == "select_production_model":
        run_production_model()
    if parsed_args.step == "make_predictions":
        if parsed_args.data_location != "":
            run_make_predictions(parsed_args.data_location)
        else:
            run_make_predictions("")
    # if parsed_args.step == "make_monitoring":
    #     run_model_monitoring()
    if parsed_args.step == "test_options":
        if parsed_args.data_location != "":
            print(f"step = {parsed_args.step}, location = {parsed_args.data_location}")
        print(f"setp = {parsed_args.step}")
