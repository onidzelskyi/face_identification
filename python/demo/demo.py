import base64

import flask
from flask import Flask, session
from flask_cors import CORS, cross_origin

from flask_sqlalchemy import SQLAlchemy

from utils import FaceRecognition


app = Flask(__name__)
CORS(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)


class Request(db.Model):
    '''Request model save processing data in database'''
    id = db.Column(db.Integer, primary_key=True)
    create_date = db.Column(db.DateTime)
    ip_addr = db.Column(db.String)
    group_image = db.Column(db.String)
    single_image = db.Column(db.String)
    result_image = db.Column(db.String)


@app.route('/', methods=['GET'])
def detect():
    username = session
    print(username)
    return ''
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
    username = request.cookies.get('username')
    print(username)
    file_name = '{}.jpg'.format(flask.request.environ['QUERY_STRING'])
    flask.request.files['file'].save(file_name)
    return ''


app.run(debug=True,host='0.0.0.0',port=5000)