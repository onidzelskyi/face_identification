import base64

import flask
from flask import Flask, render_template
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

import my_utils


app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_ECHO'] = True


db = SQLAlchemy(app)


class FaceProcess(db.Model):
    """Store Face process"""
    __tablename__ = 'process_info'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_image_file_path = db.Column(db.String)
    single_image_file_path = db.Column(db.String)
    group_faces_info = db.Column(db.String)
    single_faces_info = db.Column(db.String)
    predicted_face = db.Column(db.Integer)

    def __init__(self, group_file, single_file, group_faces, single_faces, predicted):
        self.group_image_file_path = group_file
        self.single_image_file_path = single_file
        self.group_faces_info = group_faces
        self.single_faces_info = single_faces
        self.predicted_face = predicted


db.create_all()


class FaceImage():
    def __init__(self, image_file):
        with open(image_file, 'rb') as infile:
            self.image_content = base64.b64encode(infile.read()).decode('UTF-8')


@app.route('/', methods=['GET'])
def detect():
    fr = my_utils.FaceRecognition()
    fr.process('group.jpg', 'single.jpg')

    return render_template('show_entries.html',
                           entries=fr.group.get_original_faces())

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


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)