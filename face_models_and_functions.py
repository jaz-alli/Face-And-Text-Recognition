import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from skimage.feature import hog

#loading models
vgg16_cnn = load_model('vgg16_cnn.h5')
cnn = load_model('cnn.h5')
svm_hog = joblib.load('svm_hog.pkl')
mlp_hog = load_model('mlp_hog.h5')
svm_surf = joblib.load('svm_surf.pkl')
mlp_surf = load_model('mlp_surf.h5')
kmeans = joblib.load('kmeans_surf.pkl')


#get faces in image for testing
def get_faces(img):

    #use haar cascade to detect faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=3)
    
    #initialise output lists - faces, x, and y coordinates
    faces_list=[]
    x_list=[]
    y_list=[]
    
    #get face positions and crop faces
    for (x, y, w, h) in faces:
        
        x_list.append(x)
        y_list.append(y)
        
        if x != None:
            
            if y > 200:
                y = y -100
                h = h + 100
            # print(x,y,w,h)
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img,(300,300))
            faces_list.append(crop_img)
        else:
            return
    return faces_list, x_list, y_list


#make prediciton using vgg16cnn model
def vgg16cnn_predict(faces, x_pos, y_pos):
    
    count = 0

    #loop over faces to make prediction for each face
    for face in faces:
        face = cv2.resize(face,(224,224))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        prediction = vgg16_cnn.predict(face.reshape(1, 224, 224, 3))
        pred_num = np.argmax(prediction)
        final_pred = (pred_match_cnn(pred_num))
        print(final_pred, x_pos[count], y_pos[count])

        count = count + 1


#make prediction using cnn model
def cnn_predict(faces, x_pos, y_pos):
    
    count = 0

    #loop over faces to make prediction for each face
    for face in faces:
        face = cv2.resize(face,(128,128))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        prediction = cnn.predict(face.reshape(1, 128, 128, 1))
        pred_num = np.argmax(prediction)
        final_pred = (pred_match_cnn(pred_num))
        print(final_pred, x_pos[count], y_pos[count])

        count = count + 1


#make prediciton using svm hog model
def svm_hog_predict(faces, x_pos, y_pos):

    count = 0
    hog_features = []

    #loop over faces to make prediction for each face
    for face in faces:

        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face,(300,300))
        pixels_per_cell = 16
        fd,hog_image = hog(face, orientations=8, pixels_per_cell=(pixels_per_cell, pixels_per_cell),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
        hog_features.append(fd)
        final_pred = svm_hog.predict(hog_features)
        print(final_pred, x_pos[count], y_pos[count])
        hog_features = []
        count = count + 1


#make prediction using mlp hog model
def mlp_hog_predict(faces, x_pos, y_pos):

    count = 0
    hog_features = []

    #loop over faces to make prediction for each face
    for face in faces:

        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face,(300,300))
        pixels_per_cell = 16
        fd,hog_image = hog(face, orientations=8, pixels_per_cell=(pixels_per_cell, pixels_per_cell),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
        hog_features.append(fd)
        X_test = np.reshape(fd, (28800,))
        print(X_test.shape)
        final_pred = mlp_hog.predict(X_test)
        print(final_pred, x_pos[count], y_pos[count])
        # hog_features = []
        count = count + 1


#make prediciton using svm surf model
def svm_surf_predict(faces, x_pos, y_pos):

    count = 0
    surf_features = []
    k=256

    #loop over faces to make prediction for each face
    for face in faces:

        #get surf features
        surf = cv2.xfeatures2d.SURF_create()
        kp, desc = surf.detectAndCompute(face, None)
        surf_features.append(desc)

        #input surf features into kmeans model
        kmeans_features = []
        for d in surf_features:
            c = kmeans.predict(d)
            kmeans_features.append(np.array([np.sum(c == ci) for ci in range(k)]))

        #predict on kmeans output    
        kmeans_features = np.array(kmeans_features)
        final_pred = svm_surf.predict(kmeans_features)
        print(final_pred, x_pos[count], y_pos[count])
        surf_features = []
        count = count + 1


#make prediction using mlp surf model
def mlp_surf_predict(faces, x_pos, y_pos):

    count = 0
    surf_features = []
    k=256

    #loop over faces to make prediction for each face
    for face in faces:

        #get surf features
        surf = cv2.xfeatures2d.SURF_create()
        kp, desc = surf.detectAndCompute(face, None)
        surf_features.append(desc)

        #input surf features into kmeans model
        kmeans_features = []
        for d in surf_features:
            c = kmeans.predict(d)
            kmeans_features.append(np.array([np.sum(c == ci) for ci in range(k)]))

        #predict on kmeans output     
        kmeans_features = np.array(kmeans_features)
        prediction = mlp_surf.predict(kmeans_features)
        final_pred = np.argmax(prediction)
        print(final_pred, x_pos[count], y_pos[count])
        surf_features = []
        count = count + 1


#match prediciton to unique ID of person for CNN output
def pred_match_cnn(prediction):

    if prediction == 0:
        result = 1
    elif prediction == 1:
        result = 2
    elif prediction == 2:
        result = 3
    elif prediction == 3:
        result = 4
    elif prediction == 4:
        result = 5
    elif prediction == 5:
        result = 6
    elif prediction == 6:
        result = 7
    elif prediction == 7:
        result = 8
    elif prediction == 8:
        result = 9
    elif prediction == 9:
        result = 10
    elif prediction == 10:
        result = 11
    elif prediction == 11:
        result = 12
    elif prediction == 12:
        result = 13
    elif prediction == 13:
        result = 14
    elif prediction == 14:
        result = 15
    elif prediction == 15:
        result = 16
    elif prediction == 16:
        result = 17
    elif prediction == 17:
        result = 20
    elif prediction == 18:
        result = 21
    elif prediction == 19:
        result = 22
    elif prediction == 20:
        result = 24
    elif prediction == 21:
        result = 33
    elif prediction == 22:
        result = 34
    elif prediction == 23:
        result = 36
    elif prediction == 24:
        result = 37
    elif prediction == 25:
        result = 38
    elif prediction == 26:
        result = 39
    elif prediction == 27:
        result = 40
    elif prediction == 28:
        result = 41
    elif prediction == 29:
        result = 42
    elif prediction == 30:
        result = 43
    elif prediction == 31:
        result = 44
    elif prediction == 32:
        result = 45
    elif prediction == 33:
        result = 46
    elif prediction == 34:
        result = 47
    elif prediction == 35:
        result = 48
    elif prediction == 36:
        result = 49
    elif prediction == 37:
        result = 50
    elif prediction == 38:
        result = 51
    elif prediction == 39:
        result = 52
    elif prediction == 40:
        result = 53
    elif prediction == 41:
        result = 54
    elif prediction == 42:
        result = 55
    elif prediction == 43:
        result = 56
    elif prediction == 44:
        result = 57
    elif prediction == 45:
        result = 58
    elif prediction == 46:
        result = 59
    elif prediction == 47:
        result = 60
    elif prediction == 48:
        result = 61
    elif prediction == 49:
        result = 62
    elif prediction == 50:
        result = 63
    elif prediction == 51:
        result = 64
    elif prediction == 52:
        result = 65
    elif prediction == 53:
        result = 66
    elif prediction == 54:
        result = 67
    elif prediction == 55:
        result = 68
    elif prediction == 56:
        result = 69
    elif prediction == 57:
        result = 70
    elif prediction == 58:
        result = 71
    elif prediction == 59:
        result = 72
    elif prediction == 60:
        result = 73
    elif prediction == 61:
        result = 74
    elif prediction == 62:
        result = 75
    elif prediction == 63:
        result = 76
    elif prediction == 64:
        result = 77
    elif prediction == 65:
        result = 78
    elif prediction == 66:
        result = 79
    elif prediction == 67:
        result = 80
    else:
        result = 81

    return result









