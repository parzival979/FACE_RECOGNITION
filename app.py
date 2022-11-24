from flask import render_template,Flask,request
import numpy as np,pandas as pd ,tensorflow as tf,tensorflow_hub as hub 

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    print(tf.__version__)
    return render_template('basic.html')

@app.route('/',methods=['POST'])
def prediction():
    model_path='static/data/20221124-2253%-full_model.h5'
    model=load_model(model_path)

    imageFile=request.files['image_file']
    image_path="static/data/TESTED_PHOTOS"+imageFile.filename
    path=[image_path]
    imageFile.save(image_path)

    data_batch=create_data_batch(X=path)

    predictions=model.predict(data_batch)
    predictions_label=get_predicted_label(predictions[0])

    return render_template('basic.html',prediction_label=predictions_label)


labels_csv=pd.read_csv("static/data/req_names.csv")
labels=labels_csv['Name'].to_numpy()
unique_breeds=np.unique(labels)

def get_predicted_label(predictions):
  """
  Get the required label i.e dog breed for a particular prediction
  """

  return unique_breeds[np.argmax(predictions)]


def load_model(model_path):
  """
  This function is used to load a model from a given location
  """

  model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})
  return model

BATCH_SIZE=32 

def create_data_batch(X,y=None,batch_size=BATCH_SIZE):
  """
  Creates batches of given batch size from our data.
  It also accepts data without labels.
  """

  print('Creating data batches for test data')
  data=tf.data.Dataset.from_tensor_slices((tf.constant(X))) #just making a dataset out of data 
  data=data.map(image_tensors).batch(batch_size)
  return data 

IMG_SIZE=224

def image_tensors(img_path,img_size=IMG_SIZE):
  """
  Taking an image path and turning it into sensors 
  """

  #read an image file 
  image=tf.io.read_file(img_path)
  
  #Turning the image file into tensors with 3 colors [Red,Green,Blue]
  image=tf.image.decode_jpeg(image,channels=3)

  #Normalizing our image to size 0-1
  image=tf.image.convert_image_dtype(image,dtype=tf.float32)
            
  #Resizing our image to 224x224 
  image =tf.image.resize(image,size=[img_size,img_size])
    
  return image 


if __name__=='__main__':
    app.run(debug=True)