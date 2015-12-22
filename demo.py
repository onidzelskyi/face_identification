#!/usr/bin/python

import numpy as np
import cv2
import sys
import math


INTERPOLATION = cv2.INTER_CUBIC
PCA_energy = .99
DEBUG = 1
GROUP_IMG_GRAY_FACES = "group_faces.jpeg"
SINGLE_IMG_GRAY_FACES = "single_faces.jpeg"


face_detector = [
                 cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'),
                 cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'),
                 cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml'),
                 cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml'),
                 cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_profileface.xml')
                 ]


eye_detector = [
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_lefteye_2splits.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_big.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_small.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml'),
                cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml')
                ]


##
# Get images from command line
#
def getImgFiles():
    group_img = sys.argv[1]
    single_img = sys.argv[2]
    return group_img, single_img


##
# Read images from file
#
def readImages(group_img, single_img):
    G = cv2.imread(group_img)
    S = cv2.imread(single_img)
    return G, S


##
# Convert images from RGB to gray
#
def rgb2gray(G, S):
    G_gray = cv2.cvtColor(G, cv2.COLOR_BGR2GRAY)
    S_gray = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
    return G_gray, S_gray


##
# Detect faces
#
def detectFaces(img, file):
    faces = detectFacesFromImg(img)
    # debug
    if DEBUG==1 and faces is not None:
        print "Num best faces candidates: ", len(faces)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
        cv2.imwrite(file, img)

    return faces


##
# Detect faces
#
def detectFacesFromImg(I):
    # Initialize empty list of face candidates
    faces = []
    
    # Choose best face detector by max detected faces
    for fd in face_detector:
        
        # get matrix of faces candidates
        A = fd.detectMultiScale(I, 1.3, 5)
        
        # if any faces were extracted
        if len(A)>0:
            
            if DEBUG==1:
                print "Number of faces candidates before checking eyes: ", len(A)

            # Check if face has 2 eyes
            #A = checkEyes(A, I)

            if DEBUG==1:
                print "Number of faces candidates after checking eyes: ", len(A)
            
            # Check if faces left in matrix A
            if len(A)>0:
                faces.append((A, len(A),))

    # debug
    if DEBUG==1:
        print "List num of faces for each of face detector: ", [face[1] for face in faces]
    
    # Get firs max faces candidates
    B = sorted(faces, key=lambda X: X[1], reverse=True)
    
    # return matrix of max faces candidates if exists
    return B[0][0] if len(B) and B[0][1]>0 else None


##
# Detect faces
#
def checkEyes(A, I):
    # Save faces matrix A in temp matrix B
    B = A
    
    # debug
    if DEBUG==1:
        print "checkEyes before: ", A, type(A)
    
    # number of faces candidates
    rows = A.shape[0]

    # debug
    if DEBUG==1:
        print "Number of faces candidates for eye detection: ", rows
    
    # Check pair of eyes for each face candidate
    for idx in range(rows):

        # coordinates of face rectangle
        (x,y,w,h) = A[idx,:]
        
        # create face candidate subimage
        roi_gray = I[y:y+h, x:x+w]
        
        # detect eyes for face candidate subimage
        eyes = eye_detector[0].detectMultiScale(roi_gray)

        # debug
        if DEBUG==1:
            print "Num of eyes face candidate: #", idx, ": ", len(eyes)

        # check if face candidate has 2 eyes
        if len(eyes)!=2:
            # if no delete face candidate for temp face matrix B
            B = np.delete(A, idx,0)
        else:
            # If yes draw eyes on face candidate subimage
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    # debug
    if DEBUG==1:
        print "Face candidates after checking eyes: ", B, " num: ", len(B)
    cv2.imwrite("aaa.jpeg", I)
    
    # Return face matrix B
    return B

##
# Main function
#
def main():
    # Get images form command line arguments
    group_img, single_img = getImgFiles()

    # Read images
    G, S = readImages(group_img, single_img)
    #np.save("G", G)
    #np.save("S", S)
    #G = np.load("G.npy")
    #S = np.load("S.npy")

    # Convert color to gray image
    G_gray, S_gray = rgb2gray(G, S)
    #np.save("G_gray", G_gray)
    #np.save("S_gray", S_gray)
    #G_gray = np.load("G_gray.npy")
    #S_gray = np.load("S_gray.npy")

    # Detect faces
    G_faces = detectFaces(G_gray, GROUP_IMG_GRAY_FACES)
    S_faces = detectFaces(S_gray, SINGLE_IMG_GRAY_FACES)

    #print "G_faces: ", G_faces, "S_faces: ", S_faces
    #G_faces = np.array([[221,60,32,32],[327,45,36,36], [180,55,38,38], [253,55,40,40], [286,173,39,39],[230,179,35,35], [171,165,44,44], [394,179,49,49]])
    #S_faces = np.array([[191,37,202,202]])
    #np.save("G_faces", G_faces)
    #np.save("S_faces", S_faces)
    #G_faces = np.load("G_faces.npy")
    #S_faces = np.load("S_faces.npy")
    #print G_faces
    
    # Mean face size
    #A = np.zeros(shape=(G_faces.shape[0]+S_faces.shape[0], 1))
    A = np.zeros(shape=(G_faces.shape[0], 1))
    A[0:G_faces.shape[0],0] = G_faces[:,3]
    #A[G_faces.shape[0]:,0] = S_faces[:,3]
    #print A, A.shape, A.mean()
    x_mean = int(math.floor(A.mean()))
    #print x_mean
    X = np.zeros(shape=(G_faces.shape[0], x_mean*x_mean))
    X_test = np.zeros(shape=(S_faces.shape[0], x_mean*x_mean))
    #print X.shape
    
    # Extract group faces
    i = 1
    for (x,y,w,h) in G_faces:
        #cv2.rectangle(G,(x,y),(x+w,y+h),(0,0,255),4)
        # face image
        img = G_gray[y:y+h, x:x+w]
        # image shape
        height, width = img.shape[:2]
        #print height, width
        # resize image
        res = cv2.resize(img, (x_mean, x_mean), interpolation=INTERPOLATION)
        #print res.shape
        # Unfold matrix to vector
        res1 = res.reshape(1, res.shape[0]*res.shape[1])
        #print res1.shape
        # Add image vector to the matrix
        X[i-1,:] = res1
        #img_name = "G_" + str(i) + ".jpg"
        #cv2.imwrite(img_name, res)
        i = i+1
    #np.save("X", X)
    #print X.shape

    #cv2.namedWindow("demo",cv2.WINDOW_NORMAL)
    #cv2.imshow("demo", G)
    #cv2.waitKey(0);
    #cv2.destroyAllWindows()

    # Extract single faces
    i = 1
    for (x,y,w,h) in S_faces:
        img = S_gray[y:y+h, x:x+w]
        res = cv2.resize(img, (x_mean, x_mean), interpolation=INTERPOLATION)
        res1 = res.reshape(1, res.shape[0]*res.shape[1])
        #img_name = "S_" + str(i) + ".jpg"
        X_test[i-1,:] = res1
        #cv2.imwrite(img_name, img)
        i = i+1
    #np.save("X_test", X_test)
    
    # Normalization
    mu = np.mean(X, axis=0)
    #X = (X-mu)/np.std(X)
    X_m = (X-mu)
    #np.save("X_m", X_m)
    #print X
    #print mu, mu.shape

    mu = np.mean(X_test, axis=0)
    #X_test = (X_test-mu)/np.std(X_test)
    X_test_m = (X_test-mu)
    #np.save("X_test_m", X_test_m)

    # PCA
    cov = np.dot(np.transpose(X), X)/X.shape[0]
    #np.save("cov", cov)
    #print "cov: ", cov, cov.shape
    U, s, V = np.linalg.svd(cov, full_matrices=True)
    #print "s: ", s.shape
    #np.save("U", U)
    #np.save("s", s)
    #np.save("V", V)
    #U = np.load("U.npy")
    #s = np.load("s.npy")
    for k in range(s.shape[0]):
        #print "k: ", k
        sum = np.sum(s[0:k])/np.sum(s[:])
        if sum >= PCA_energy:
            #print "K: ", k
            break

    # Eigenfaces matrix
    L = U[:,0:k]
    #np.save("L",L)
    #print "L: ", L.shape
    E = np.dot(X, L)
    #np.save("E",E)
    #print "E: ", E.shape

    # Face indentification
    for i in range(X_test.shape[0]):
        I = X_test[i,:]
        #print "I: ", I.shape
        # Reduce image size to K features
        e = np.dot(I, L)
        #np.save("e",e)
        #print "e: ", e.shape
        # Subtract test image form data images matrix
        D = E - e
        #np.save("D", D)
        # Norm
        N = np.linalg.norm(D**2, axis=-1)
        #np.save("N", N)
        #print "E: ", E
        #print "e: ", e
        #print "D: ", D
        #print "N: ", N
        # Index of suggested image
        idx = np.argmin(N)
        #print "idx: ", idx, "value: ", N[idx]
        (x,y,w,h) = G_faces[idx]
        (xs,ys,ws,hs) = S_faces[i]
        cv2.rectangle(G,(x,y),(x+w,y+h),(0,0,255),4)
        cv2.rectangle(S,(xs,ys),(xs+ws,ys+hs),(0,0,255),4)

    # Concatenate images
    #x_min = np.min(np.array([G.shape[0], S.shape[0]]))
    #imlist = [G,S]
    #id_min = np.argmin(np.array([G.shape[1], S.shape[1]]))
    #y_max = max(np.array([G.shape[1], S.shape[1]]))
    #print "scale: ", float(S.shape[0])/float(G.shape[0])
    #print "scale: ", float(S.shape[1])/float(G.shape[1])
    #scaled = cv2.resize(G, None, fx=float(S.shape[0])/float(G.shape[0]), fy=float(S.shape[1])/float(G.shape[1]), interpolation=INTERPOLATION)
    #scaled = cv2.resize(G, (S.shape[1],S.shape[0],))
    #print "S: ", S.shape, "scaled: ", scaled.shape
    #R = np.concatenate((S,scaled), axis=1)

    cv2.imwrite("G1.jpeg", G)
    cv2.imwrite("S1.jpeg", S)
    print "ok"
    #cv2.namedWindow("demo",cv2.WINDOW_NORMAL)
    #cv2.imshow("demo", G)
    #cv2.waitKey(0);
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()