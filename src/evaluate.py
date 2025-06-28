from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, preds, average='weighted', zero_division=0))
