# import base64
import os
import uuid
import flask
from flask import Flask, render_template, session
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint

import my_utils


STATUS_SUCCESS = '1'
STATUS_FAIL = '-1'
STATUS_NEW = '0'


app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_ECHO'] = True
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


db = SQLAlchemy(app)

# path = None


class FaceProcess(db.Model):
    """Store Face process"""
    __tablename__ = 'process_info'

    id = db.Column(db.BLOB, primary_key=True, unique=True)
    group_faces_info = db.Column(db.String)
    single_faces_info = db.Column(db.String)
    predicted_face = db.Column(db.Integer)
    status = db.Column(db.Enum(STATUS_NEW, STATUS_SUCCESS, STATUS_FAIL), name='status', default=STATUS_NEW)
    execution_date = db.Column(db.DateTime, default=db.func.now())
    execution_time = db.Column(db.TIMESTAMP, default=None)
    UniqueConstraint('id', name='uix_1')

    def __init__(self, id, group_faces, single_faces, predicted):
        self.id = id
        self.group_faces_info = group_faces
        self.single_faces_info = single_faces
        self.predicted_face = predicted


db.create_all()


# class FaceImage():
#     def __init__(self, image_file):
#         with open(image_file, 'rb') as infile:
#             self.image_content = base64.b64encode(infile.read()).decode('UTF-8')


@app.route('/', methods=['GET'])
def todo():
    return render_template('demo.php')


@app.route('/todo', methods=['GET'])
def detect():
    fr = my_utils.FaceRecognition()
    fr.process('group.jpg', 'single.jpg', session)

    # Save result
    fr.save_images(session)

    # Close session
    session.pop('token', None)

    return render_template('show_result.html',
                           result=fr.group.get_result())
    # return render_template('show_entries.html',
    #                        entries=fr.group.get_original_faces())

    # content = None
    # with open('test.jpg', 'rb') as image:
    #     content = base64.b64encode(image.read())

    # resp = flask.Response()
    # resp.headers['Access-Control-Allow-Origin'] = '*'
    # resp.headers['Content-Type'] = 'image/jpeg'
    # resp.mimetype = 'image/gif'
    # resp.data = content
    #
    # return resp


@app.route("/", methods=['POST'])
def upload():
    global path

    if 'token' not in session:
        session['token'] = uuid.uuid4().hex

    path = './data/{}'.format(session['token'])

    if not os.path.isdir(path):
        os.makedirs(path)
    file_name = '{}/{}.jpg'.format(path, flask.request.environ['QUERY_STRING'])
    flask.request.files['file'].save(file_name)

    return session['token']


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)