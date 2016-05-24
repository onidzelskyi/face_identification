#import cv2
import numpy as np
import os
import base64
from PIL import Image, ImageDraw

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_cloud_vision.json"

DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1', credentials=credentials,
                           discoveryServiceUrl=DISCOVERY_URL)


class ImageObj(object):
    """Image object with faces."""
    def __init__(self, face_file, faces=None):
        # List of detected faces. Face is the dict with the info (e.g. coordinates, etc.)
        self.faces = faces
        # Image as the binary string. Can be sent to Google vision api
        with open(face_file, 'rb') as image:
            self.image_content = image.read()
        # Image object
        self.image = Image.open(face_file)
        # Image as a matrix
        self.image_matrix = np.asarray(self.image.convert('L'))
        # Matrix of unrolled faces in vectors
        self.faces_matrix = None
        # List of row faces
        self.raw_faces = None

    def set_faces(self, faces):
        self.faces = faces

    def get_image_content(self):
        return self.image_content


class FaceRecognition(object):
    """Class face recognition."""
    
    ENGINE_GVA = "GVA"
    ENGINE_CV2 = "CV2"

    # OpenCV
    #face_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    scale_factor = 1.3
    min_neighbors = 5
    max_results = 16

    def __init__(self, engine=ENGINE_GVA):
        # Engine
        self.engine = engine
        # Images
        self.group, self.group_matrix = None, None
        self.single, self.single_matrix = None, None
        self.predicted = None

    def process(self, group, single):
        """Main entry point"""
        # Step I. Init image objects
        self.group = ImageObj(group)
        self.single = ImageObj(single)
                
        # Step II. Detect faces
        self.detect_faces()

        # Step III. Create matrix
        self.create_matrix()

        # Step IV. Calc dissimilarity
        self.calc_dissimilarity()

        # Step V. Show result
        self.show_result()
        
    '''
    # load image
    def load_image(self):
        for (image_file, rgb, gray) in [(self.group_file, self.group_img, self.group_img_gray),
                                        (self.single_file, self.single_img, self.single_img_gray)]:
            rgb = Image.open(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    # show image
    def showImage(self, group=True):
        pass

    # detect faces
    def detectFaces(self, group=True):
        if group: self.group_faces = self.face_detector.detectMultiScale(self.group_img_gray, self.scale_factor, self.min_neighbors)
        else: self.single_faces = self.face_detector.detectMultiScale(self.single_img_gray, self.scale_factor, self.min_neighbors)


    #
    def euclidian(self):
        for i, I in enumerate(self.X_test):
            D = self.X - I
            N = np.linalg.norm(D**2, axis=-1)
            self.target[i] = np.argmin(N)

    #
    def lbp(self):
        import skimage.feature
        for i, I in enumerate(self.X):
            img = I.reshape(I.shape[0]**.5, I.shape[0]**.5)
            featured_img = skimage.feature.local_binary_pattern(img, 24, 8, method="uniform")
            self.X[i] = featured_img.reshape(1, featured_img.shape[0]*featured_img.shape[0])
        for i, I in enumerate(self.X_test):
            img = I.reshape(I.shape[0]**.5, I.shape[0]**.5)
            featured_img = skimage.feature.local_binary_pattern(img, 24, 8, method="uniform")
            self.X_test[i] = featured_img.reshape(1, featured_img.shape[0]*featured_img.shape[0])
        self.chi()

    #
    def faceVerify(self, group_img_file_name, single_img_file_name, method="euclidian"):
        # Load images
        self.loadImage(group_img_file_name)
        self.loadImage(single_img_file_name, group=False)
        
        # detect faces
        self.detectFaces()
        self.detectFaces(group=False)
        
        # Mean face size
        x_mean = int(math.floor(self.group_faces[:,3].mean()))
        # Create matrix 
        self.X = np.zeros(shape=(self.group_faces.shape[0], x_mean**2))
        self.X_test = np.zeros(shape=(self.single_faces.shape[0], x_mean**2))
        self.target = [None]*self.single_faces.shape[0]
        
        # calc
        if method=="euclidian": self.euclidian()
        elif method == "lbp": self.lbp()
        else: raise NotImplementedError
            
        # Show results
        self.showResults()
            
    # Show results
    def showResults(self):
        (x,y,w,h) = self.group_faces[self.target[0]]
        (xs,ys,ws,hs) = self.single_faces[0]
        cv2.rectangle(self.group_img,(x,y),(x+w,y+h),(0,0,255),4)
        cv2.rectangle(self.single_img,(xs,ys),(xs+ws,ys+hs),(0,0,255),4)
        cv2.imwrite("A.jpg", self.group_img)
        cv2.imwrite("B.jpg", self.single_img)

    # Chi-square distance
    def chi(self):
        for i, I in enumerate(self.X_test):
            D = np.zeros(shape=(self.X.shape[0],1))
            for j, x in enumerate(self.X):
                nom = (x-I)**2
                den = x+I
                d = nom[den!=0]/den[den!=0]
                D[j] = np.sum(d)
            self.target[i] = np.argmin(D)
        
    def test(self, g_files, s_files):
        import glob
        G, S = glob.glob(g_files), glob.glob(s_files) 

        A = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY).reshape(1,10000) for img in G]
        B = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY).reshape(1,10000) for img in S]

        import mahotas.features.lbp as lbp
        a = [lbp(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY), 3, 17) for img in G]
        b = [lbp(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY), 3, 17) for img in S]

        #import skimage.feature as ft
        #a = [ft.local_binary_pattern(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY), 24, 3, "uniform").reshape(1, 10000) for img in G]
        #b = [ft.local_binary_pattern(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY), 24, 3, "uniform").reshape(1, 10000) for img in S]

        print "LBP features:  ", a[0].shape

        idx = 0
        D = 0
        for i, I in enumerate(a):
            nom = (I-b[0])**2
            den = (I+b[0])
            d = np.sum(nom[den!=0.]/den[den!=0.])
            print d,
            if not D or D > d: 
                D = d
                idx = i
        print "Best candidate: ", idx
        img1 = A[idx].reshape(100,100)
        img2 = B[0].reshape(100,100)

        cv2.imwrite("A.jpg", img1)
        cv2.imwrite("B.jpg", img2)
    '''

    def detect_faces(self):
        '''Google Vision API.'''
        for image_object in [self.group, self.single]:
            batch_request = [{'image': {'content': base64.b64encode(image_object.get_image_content()).decode('UTF-8')},
                              'features': [{'type': 'FACE_DETECTION', 'maxResults': self.max_results}]}]

            service = get_vision_service()
            request = service.images().annotate(body={'requests': batch_request})
            response = request.execute()
            image_object.set_faces(response['responses'][0]['faceAnnotations'])

    def create_matrix(self):
        '''Transform faces to matrices.'''
        LL = []
        for image_object in [self.group, self.single]:
            #DD = []
            for face in image_object.faces:
                vertices = face['fdBoundingPoly']['vertices']
                lll = np.fabs((vertices[0]['y'] - vertices[2]['y']) * (vertices[0]['x'] - vertices[1]['x']))
                LL.append(lll)
                #image_object.raw_faces.append(image_object.image_matrix[vertices[0]['y']:vertices[2]['y'], vertices[0]['x']:vertices[1]['x']])

        # TODO: one from mean, moda, mediana
        mean_value = np.rint(np.average(LL))

        for image_object in [self.group, self.single]:
            image_object.faces_matrix = np.zeros(shape=(len(image_object.faces), mean_value))
            for i in range(len(image_object.faces)):
                box = (vertices[0]['x'], vertices[1]['y'], vertices[1]['x'], vertices[2]['y'],)
                image_object.faces_matrix[i] = np.asmatrix(image_object.image.crop(box).resize((int(mean_value), int(mean_value)))).A1

    def calc_dissimilarity(self):
        self.predicted = np.argmin(np.abs(np.sum((self.group.faces_matrix - self.single.faces_matrix), axis=1)))

    def show_result(self):
        draw = ImageDraw.Draw(self.group.image)
        box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in self.group.faces[self.predicted]['fdBoundingPoly']['vertices']]
        #box = (self.group.faces[self.predicted]['fdBoundingPoly']['vertices'].get('x', 0.0), self.group.faces[self.predicted]['fdBoundingPoly']['vertices'].get('y', 0.0))
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        self.group.image.save('test.jpg')