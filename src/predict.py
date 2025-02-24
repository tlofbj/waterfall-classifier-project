import cv2, joblib, argparse
from termcolor import cprint
from utils.preprocess import modify_image


def load_raw(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    return modify_image(image).flatten()


def predict(flattened_image, model_path):
    model = joblib.load(model_path)
    prediction = model.predict([flattened_image])
    cprint(f"Prediction: {prediction[0]}", on_color="on_green", attrs=['bold'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a waterfall")
    parser.add_argument("image_path", type=str, help="image path")
    parser.add_argument("-m", "--model_path", type=str, default="models/svm_model.pkl", help="model path")
    args = parser.parse_args()

    flattened_image = load_raw(args.image_path)
    prediction = predict(flattened_image, args.model_path)

