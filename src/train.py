import joblib, argparse
from sklearn.svm import SVC
from termcolor import cprint
from utils.load_data import load_data


def train(X, y, model_save_path, C=1.0, kernel="linear"):
    svm = SVC(C=C, kernel=kernel)
    svm.fit(X, y) # Train model
    joblib.dump(svm, model_save_path) # Save model
    cprint(f"Training: Successful (saved to {model_save_path})", on_color="on_green", attrs=['bold'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-m", "--model_save_path", type=str, default="models/svm_model.pkl", help="model path")
    parser.add_argument("-c", "--C", type=float, default="1.0", help="regularization parameter")
    parser.add_argument("-k", "--kernel", type=str, default="poly", help="model path")
    args = parser.parse_args()

    X, y = load_data("data/train/processed")
    train(X, y, args.model_save_path, args.C, args.kernel)
    
