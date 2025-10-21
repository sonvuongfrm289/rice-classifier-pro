import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import argparse

CLASS_NAMES = ['Class1','Class2','Class3', '...', 'Class10']

def load_and_preprocess(img_path, img_size=(64,64)):
    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def main(args):
    model = tf.keras.models.load_model(args.model)
    x = load_and_preprocess(args.image, img_size=(64,64))
    pred = model.predict(x)
    idx = np.argmax(pred)
    print(f"Predicted class: {CLASS_NAMES[idx]} with confidence {pred[0][idx]*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/rice_model_fast.h5')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    main(args)