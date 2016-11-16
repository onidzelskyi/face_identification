import numpy as np
import os
import io # For converting PIL image to bytes before base64 encode
# import uuid
import base64
from PIL import Image, ImageDraw

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from demo import db, FaceProcess


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "google_cloud_vision.json")

DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials, discoveryServiceUrl=DISCOVERY_URL)


class ImageObj(object):
    """Image object with faces."""
    def __init__(self, face_file, session, faces=None, name='group'):
        # List of detected faces. Face is the dict with the info (e.g. coordinates, etc.)
        self.faces = faces

        # Image as the binary string. Can be sent to Google vision api
        path = './data/{}/{}'.format(session['token'], face_file)
        with open(path, 'rb') as image:
            self.image_content = image.read()

        # Image object
        path = './data/{}/{}'.format(session['token'], face_file)
        self.image = Image.open(path)

        # Image as a matrix
        self.image_matrix = np.asarray(self.image.convert('L'))
        # Matrix of unrolled faces in vectors
        self.faces_matrix = None
        # List of row faces
        self.raw_faces = None
        # Name of image: group or single
        self.image_name = name
        self.original_faces = []

    def set_faces(self, faces):
        self.faces = faces

    def get_faces(self):
        return self.faces

    def get_number_of_faces(self):
        return len(self.faces)

    def get_name(self):
        return self.image_name

    def get_image_content(self):
        return self.image_content

    def get_original_faces(self):
        return self.original_faces

    def get_result(self):
        io_buf = io.BytesIO()
        self.image.save(io_buf, format='jpeg')
        return base64.b64encode(io_buf.getvalue())

    def debug_faces(self, mean_value):
        self.faces_matrix = np.zeros(shape=(len(self.faces), int(mean_value ** 2)))
        for i in range(len(self.faces)):
            vertices = self.faces[i]['fdBoundingPoly']['vertices']
            box = (vertices[0]['x'], vertices[1]['y'], vertices[1]['x'], vertices[2]['y'],)
            original_face = self.image.crop(box)
            self.original_faces.append(base64.b64encode(original_face.tobytes()))#.encode('utf-8'))
            #original_face.save('{}_face_{}.jpg'.format(self.image_name, i+1))
            scaled_face = original_face.resize((int(mean_value), int(mean_value)))
            #scaled_face.save('{}_scaled_face_{}.jpg'.format(self.image_name, i+1))
            converted_face = scaled_face.convert('L')
            #converted_face.save(('{}_converted_face_{}.jpg'.format(self.image_name, i+1)))
            rotated_face = converted_face.rotate(self.faces[i]['rollAngle'])
            #rotated_face.save(('{}_aligned_face_{}.jpg'.format(self.image_name, i+1)))
            #self.faces_matrix[i] = np.asmatrix(converted_face).A1
            self.faces_matrix[i] = np.asmatrix(rotated_face).A1


class FaceRecognition(object):
    """Class face recognition."""
    ENGINE_GVA = "GVA"
    max_results = 16
    
    def __init__(self, engine=ENGINE_GVA):
        # Engine
        self.engine = engine
        # Images
        self.group, self.group_matrix = None, None
        self.single, self.single_matrix = None, None
        self.predicted = None

    def process(self, group, single, session):
        """Main entry point"""
        # Step I. Init image objects
        self.group = ImageObj(group, session, name='group')
        self.single = ImageObj(single, session, name='single')

        # Save images to store
        # token = uuid.uuid4()
        # self.path = './data/{}'.format(token.hex)
        # self.save_images(image_objs=[self.group, self.single], names=['group', 'single'])

        # Step II. Detect faces
        self.detect_faces()

        # Step III. Create matrix
        self.create_matrix()

        # Step IV. Calc dissimilarity
        self.calc_dissimilarity()

        # Step V. Show result
        self.show_result()

        # Save processed data in DB
        face_process = FaceProcess(session['token'],
                                   str(self.group.faces),
                                   str(self.single.faces),
                                   int(self.predicted))
        db.session.add(face_process)
        db.session.commit()

    def detect_faces(self):
        '''Google Vision API.'''
        for image_object in [self.group, self.single]:
            batch_request = [{'image': {'content': base64.b64encode(image_object.get_image_content()).decode('UTF-8')},
                              'features': [{'type': 'FACE_DETECTION', 'maxResults': self.max_results}]}]

            service = get_vision_service()
            request = service.images().annotate(body={'requests': batch_request})
            response = request.execute()

            if len(response['responses'][0]) == 0:
                raise ValueError('No faces were detected in {}. Exit.'.format(image_object.get_name()))

            image_object.set_faces(response['responses'][0]['faceAnnotations'])
            print('{}: {} faces'.format(image_object.get_name(), image_object.get_number_of_faces()))

    def create_matrix(self):
        '''Transform faces to matrices.'''
        LL = []
        for image_object in [self.group, self.single]:
            #DD = []
            for face in image_object.get_faces():
                vertices = face['fdBoundingPoly']['vertices']
                #lll = np.fabs((vertices[0]['y'] - vertices[2]['y']) * (vertices[0]['x'] - vertices[1]['x']))
                lll = np.fabs((vertices[0]['y'] - vertices[2]['y']))
                LL.append(lll)
                #image_object.raw_faces.append(image_object.image_matrix[vertices[0]['y']:vertices[2]['y'], vertices[0]['x']:vertices[1]['x']])

        # TODO: one from mean, moda, mediana
        # mean_value = np.rint(np.average(LL))
        # Or use 100 by 100 image
        mean_value = 100

        for image_object in [self.group, self.single]:
            image_object.debug_faces(mean_value)

    def calc_dissimilarity(self):
        #import pdb; pdb.set_trace()
        self.predicted = np.argmin(np.linalg.norm(self.group.faces_matrix-self.single.faces_matrix, axis=1))

    def show_result(self):
        draw = ImageDraw.Draw(self.group.image)
        box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in self.group.faces[self.predicted]['fdBoundingPoly']['vertices']]
        #box = (self.group.faces[self.predicted]['fdBoundingPoly']['vertices'].get('x', 0.0), self.group.faces[self.predicted]['fdBoundingPoly']['vertices'].get('y', 0.0))
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        
        # self.save_images(image_objs=[self.group], names=['result'])

    def save_images(self, session):
        # if not os.path.isdir(self.path):
        #     os.makedirs(self.path)
        # for image_obj, name in zip(image_objs, names):
        #     image_obj.image.save('{}/{}.jpg'.format(path, name))
        path = './data/{}'.format(session['token'])
        self.group.image.save('{}/result.jpg'.format(path))
