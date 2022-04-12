from flask import Flask,request,jsonify
import numpy as np

import tensorflow as tf
from PIL import Image
import re


# model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))


def predict():
    # cgpa = request.form.get('cgpa')
    # iq = request.form.get('iq')
    # profile_score = request.form.get('profile_score')
    # input_query = np.array([[cgpa,iq,profile_score]])
    # result = model.predict(input_query)[0]
   

    imgData = request.get_data()
    convertImage(imgData)
    input_data = imread('output.png',mode='L')
    # x = np.invert(x)
    # x = imresize(x,(28,28))
    # x = x.reshape(1,28,28,1)
    
    
# -----------------------------------------------------------------------------------------------

    positions=['downdog','goddess','plank','tree','warrior']

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="project.tflite") # tflite file path
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    input_data=Image.open("/content/drive/MyDrive/Yoga posture detection/dataset/WhatsApp Image 2022-03-19 at 8.16.48 PM.jpeg")   # test img path
    input_data=input_data.resize((224,224))
    input_data=np.expand_dims(input_data,axis=0)
    input_data=np.array(input_data,dtype="float32")
    input_data=input_data/255

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    # print(max(output_data[0]))

    result=(positions[np.where(output_data[0] == max(output_data[0]))[0][0]])
    return jsonify({'placement':str(result)})

    
#-----------------------------------------------------------------------------------------------




    # im=im.resize((224,224))
    # im=np.expand_dims(im,axis=0)
    # im=np.array(im,dtype="float32")
    # im=im/255
    # pred=model.predict([im])
    # # print(max(pred[0])*100)
    # print(pred)


    # with graph.as_default():
    #   out = model.predict(x)
    #   print(out)
    #   print(np.argmax(out,axis=1))

    #   response = np.array_str(np.argmax(out,axis=1))
    #   return response	


if __name__ == '__main__':
    app.run(debug=True)
