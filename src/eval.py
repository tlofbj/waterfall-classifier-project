import joblib, argparse, seaborn, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.load_data import load_data
from termcolor import cprint


def evaluate(X_test, y_test, model_path):
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cprint(f"Accuracy: {accuracy * 100:.2f}%", on_color="on_green", attrs=['bold'])
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", report)
    
    cm = confusion_matrix(y_test, y_pred)
    seaborn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("-t", "--test_dataset_dir", type=str, default="data/test/processed", help="directory of datatest for testing (images should be preprocessed!)")
    parser.add_argument("-m", "--model_path", type=str, default="models/svm_model.pkl", help="model path")
    args = parser.parse_args()

    X, y = load_data(args.test_dataset_dir)
    evaluate(X, y, args.model_path)
    
