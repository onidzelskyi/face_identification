import base64

import flask
from flask import Flask
from flask_cors import CORS, cross_origin

from utils import FaceRecognition


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def detect():
    fr = FaceRecognition()
    fr.process('group.jpg', 'single.jpg')
    
    content = None
    with open('test.jpg', 'rb') as image:
        content = base64.b64encode(image.read())

    resp = flask.Response()
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-Type'] = 'image/jpeg'
    resp.mimetype = 'image/gif'
    resp.data = content

    return resp


@app.route("/", methods=['POST'])
def upload():
    file_name = '{}.jpg'.format(flask.request.environ['QUERY_STRING'])
    flask.request.files['file'].save(file_name)
    return ''


app.run(debug=True,host='0.0.0.0',port=5000)