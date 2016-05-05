#import cv2
import numpy as np
import math
from PIL import Image


class ImageObj(object):
    """Image object with faces."""
    def __init__(self, file_name, img_rgb=None, img_gray=None, faces=None):
        self.file_name = file_name
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.faces = faces


class FaceRecognition(object):
    """Class face recognition."""
    
    ENGINE_GVA = "GVA"
    ENGINE_CV2 = "CV2"

    # OpenCV
    face_detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    scale_factor = 1.3
    min_neighbors = 5
    max_results = 16

    def __init__(self, engine=FaceRecognition.ENGINE_GVA):
        # Engine
        self.engine = engine
        # Images
        self.group = None
        self.single = None
        
        self.X = None
        self.X_test = None
        self.target = []
        
    def process(self, group, single):
        """Main entry point"""
        # Step I. Init image objects
        self.group = ImageObj(group)
        self.single = ImageObj(single)
        
        self.load_image()
        
        # Step II. Detect faces
        self.detect_face()
        
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


    # Google Vision API section
    def detect_face(self):
    for (image_content, faces) in [(self.group_img, self.group_faces), (self.single_img, self.single_faces)]:
        if self.engine == ENGINE_CV2:
            faces = self.face_detector.detectMultiScale(image_content, scale_factor, min_neighbors)
        else:
            batch_request = [{'image': {'content': base64.b64encode(image_content).decode('UTF-8')},
                              'features': [{'type': 'FACE_DETECTION', 'maxResults': self.max_results}]}]

            service = get_vision_service()
            request = service.images().annotate(body={'requests': batch_request,})
            response = request.execute()
            faces = response['responses'][0]['faceAnnotations']

