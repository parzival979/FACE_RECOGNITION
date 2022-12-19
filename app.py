from flask import render_template, Flask, request
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    print(tf.__version__)
    return render_template('basic.html')


@app.route('/', methods=['POST'])
def prediction():
    model_path = 'static/data/final_model.h5'
    model = load_model(model_path)
    imageFile = request.files['image_file']
    image_path = "static/data/TESTED_PHOTOS/"+imageFile.filename
    path = [image_path]
    imageFile.save(image_path)
    # applying haar cascade to detect face
    import cv2
    face_cascade = cv2.CascadeClassifier(
        'static/data/haarcascade_frontalface_alt.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # crop the face
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(image_path, roi_color)

    data_batch = create_data_batch(X=path)

    predictions = model.predict(data_batch)
    predictions_label = get_predicted_label(predictions[0])

    return render_template('basic.html', prediction_label=predictions_label)


labels_csv = pd.read_csv("static/data/req_names.csv")
labels = labels_csv['Name'].to_numpy()
unique_breeds = np.unique(labels)


def get_predicted_label(predictions):

    return unique_breeds[np.argmax(predictions)]


def load_model(model_path):

    model = tf.keras.models.load_model(model_path, custom_objects={
                                       "KerasLayer": hub.KerasLayer})
    return model


BATCH_SIZE = 32


def create_data_batch(X, y=None, batch_size=BATCH_SIZE):

    print('Creating data batches for test data')
    data = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X)))  # just making a dataset out of data
    data = data.map(image_tensors).batch(batch_size)
    return data


IMG_SIZE = 224


def image_tensors(img_path, img_size=IMG_SIZE):

    # read an image file
    image = tf.io.read_file(img_path)

    # Turning the image file into tensors with 3 colors [Red,Green,Blue]
    image = tf.image.decode_jpeg(image, channels=3)

    # Normalizing our image to size 0-1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Resizing our image to 224x224
    image = tf.image.resize(image, size=[img_size, img_size])

    return image


if __name__ == '__main__':
    app.run(debug=True)
