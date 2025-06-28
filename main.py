from src.train_model import train_and_save_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    DATA_PATH = "data/tasks_150_large.csv"
    MODEL_PATH = "models/task_classifier.pkl"

    X_test, y_test, model = train_and_save_model(DATA_PATH, MODEL_PATH)
    evaluate_model(model, X_test, y_test)
