from flask import Flask, request, jsonify
import numpy as np
import pickle
# model = pickle.load(open('model.pkl','rb'))
from keras.models import load_model

# from keras.applications.vgg16 import preprocess_input
model = load_model('/content/drive/MyDrive/Yoga posture detection/YogaNet_model_1_1.h5')
app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


def predict():
    # cgpa = request.form.get('cgpa')
    # iq = request.form.get('iq')
    # profile_score = request.form.get('profile_score')
    # input_query = np.array([[cgpa,iq,profile_score]])
    # result = model.predict(input_query)[0]
    # return jsonify({'placement':str(result)})

    imgData = request.get_data()
    convertImage(imgData)
    im = imread('output.png', mode='L')
    # x = np.invert(x)
    # x = imresize(x,(28,28))
    # x = x.reshape(1,28,28,1)

    im = im.resize((224, 224))
    im = np.expand_dims(im, axis=0)
    im = np.array(im, dtype="float32")
    im = im / 255
    pred = model.predict([im])
    # print(max(pred[0])*100)
    print(pred)

    # with graph.as_default():
    #   out = model.predict(x)
    #   print(out)
    #   print(np.argmax(out,axis=1))

    #   response = np.array_str(np.argmax(out,axis=1))
    #   return response


if __name__ == '__main__':
    app.run(debug=True)
