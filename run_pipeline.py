"""Build model artifacts and reports from the configured sales dataset."""
from src import data_preprocessing, feature_engineering, train_model, evaluate_model, eda

if __name__ == "__main__":
    data_preprocessing.run_pipeline()
    feature_engineering.run_pipeline()
    train_model.run_pipeline()
    evaluate_model.run_evaluation()
    eda.run_eda()
