import os
from pathlib import Path

import cv2
import mxnet as mx
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import normalize


def get_embedding(model, img):
    # data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(data.shape)
    # data = np.transpose(data, (2,0,1))
    # data = np.expand_dims(data, axis=0)
    # data = mx.nd.array(data)
    blob = cv2.dnn.blobFromImage(img, 1.0, [112, 112], [0, 0, 0, 0], True, False, cv2.CV_32F)
    data = mx.nd.array(blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    return model.get_outputs()[0].asnumpy().flatten()


def pairwise_swap(list):
    """Performs pairwise swap of a list.

    Swaps the order of the probe image and corresponding reference image 
    to match the folder structure. The place of element n and element n+1 are swapped.
    This way the list is ordered accordingly before calculating the feature difference
    with reference image before corresponding probe image.

    Parameters: 
    list: List
        Unordered list of file names.

    Returns: 
    list: List
        Ordered list of file names.
    """

    l = len(list)&~1
    list[1:l:2],list[:l:2] = list[:l:2],list[1:l:2]
    return list


def files_in_directory(rootPath, ext):
    """Reads and returns files in a directory.

    The file names of the provided directory are first sorted
    and then appended to a list. The sorting expects file names 
    of type (x-probe/reference.jpeg) where x denotes a number. 
    Pairwise swap is applied to order the image pairs in the list correctly. 
    The reference image of the image pairs is always listed before the corresponding probe image.

    Parameters
    ----------
    rootPath: String
        Path of the wanted directory.
    ext: String
        The type of file, e.g. '.jpg', '.jpeg'.

    Returns
    -------
    files: List
        List of all file names in the provided directory.
    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(rootPath):
        for file in sorted(f):      # Always reads probe images before corresponding reference images
            if ext in file:
                files.append(os.path.join(r, file))
    
    files = pairwise_swap(files)
    return files


def run_feature_difference(path, model):
    """Script for feature difference calculation.

    Reads images from the provided path, applies feature extraction and writes feature vectors to files.
    Data from files are then loaded and the difference between a probe image and reference image is caluclated.

    Parameters
    ----------
    path: String
        The path to the directory containing images. 
    model: class mxnet.module.module.Module()
        The model used for feature extraction.

    Returns
    -------
    featureDiff: List
        The difference in feature vectors between the 
        probe images and reference images provided by path.
    """

    fileList = files_in_directory(path, '.jpg')
    fileData = []
    featureDiff = []

    if not len(fileList) % 2:
        for file in range(len(fileList)):
            try: 
                img = cv2.imread(fileList[file])
                img = cv2.resize(img, (112, 112), interpolation = cv2.INTER_AREA)
                embedding = get_embedding(model, img)
                embedding = np.reshape(embedding, (49, 512))
                #embedding = np.reshape(embedding, (7, 7, 512))
                fileName = os.path.splitext(os.path.basename(fileList[file]))[0]
                writeFeaturesToFile(fileName, embedding)
                fileData.append(loadFeatures('features/' + fileName))

                if file % 2:
                    featureDiff.append(fileData[file] - fileData[file - 1]) # Probe[i] - Reference[i-1]

            except Exception as e:
                print("Error occurred main " + str(e))
    else: 
        print("Error: Path needs to consist of even number of files" )
        
    return featureDiff


def extractionFeature_cropped_image(img, strSavingFilePath, model):
    # img = cv2.imread(strInputImagePath)
    # img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    embedding = get_embedding(model, img)
    # list_noamrlized = normalize([embedding], norm='l2').tolist()[0]
    fileName = strSavingFilePath
    writeFeaturesToFile(fileName, embedding)


def extractionFeature_cropped_image_without_saving(img, model):
    # img = cv2.imread(strInputImagePath)
    # img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    embedding = get_embedding(model, img)
    list_noamrlized = normalize([embedding], norm='l2')
    return np.array(list_noamrlized[0]).T



def extractionFeatureForSingleImage(strInputImagePath, strSavingFilePath, model):
    img = cv2.imread(strInputImagePath)
    # img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    embedding = get_embedding(model, img)
    # embedding = np.reshape(embedding, (49, 512))
    # embedding = np.reshape(embedding, (7, 7, 512))
    list_noamrlized = normalize([embedding], norm='l2')
    fileName = strSavingFilePath
    writeFeaturesToFile(fileName, list_noamrlized)


def writeFeaturesToFile(fileName, features):
    """Writes feature vectors to .txt-file.

    Creates a folder named 'features' if it does not already exist. 
    Thereafter a .txt-file is created and all feature vectors are written to that spesific file.
    Each array-element equals one row in the text file. 
embedding = {ndarray: (49, 512)} [[-0.01651963 -0.00028176  0.00268294 ... -0.02883925 -0.00047491,  -0.00480966], [ 0.04184509 -0.07110211 -0.07828045 ... -0.00061134 -0.01181545,  -0.00728464], [ 0.01754834  0.01895548 -0.06040277 ... -0.04187313  0.02159425,  -0.0604819 ], ..., [ 0.012...View as Array
    Parameters
    ----------
    fileName: String
        The name of the imagefile. 
    features: numpy.ndarray
        Array with feature vectors extracted from fileName. 
    """

    path = os.getcwd()
    featureFolder = os.path.join(path, 'features')

    if not os.path.exists(featureFolder):
        os.mkdir(featureFolder)

    np.savetxt(fileName, features, fmt='%10.7f')

    '''
    with open('features/' + imageName + '.csv', 'w') as fileHandler:
        s = str(features)
        fileHandler.write(s)

    fw = open('features/' + imageName + '.txt', 'w')
    for value in range(len(features)):
        fw.write(str(features[value]) + '\n')
    fw.close()
    '''


def loadFeatures(fileName):
    """Load data from a text file. 

    Each row in the text file must have the same number of values.
    Each row in the text file corresponds to an array-element. 

    Parameters
    ----------
    fileName: String
        Name of the file. 

    Returns
    ------- 
    out: numpy.ndarray
        Array of the text file data.
    """

    return np.loadtxt(fileName, dtype=float)

    '''
    return np.array([[float(w) for w in line.split()] for line in open(fileName, "r")])

    fileObj = open(fileName, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()
    return words
    '''
    

def runSVM(bonaFeaturesTrain, morphFeaturesTrain, bonaFeaturesTest, morphFeaturesTest):
    """Trains support vector machines on all difference feature vectors and classifies the data.

    Creates four arrays corresponding to x_train, y_train, x_test and y_test. 
    FeaturesTrain and featuresTest are concatenated to a (98, 512) shape array and 
    targetTrain and targetTest are concatenated and flattened to a (98, ) shape array.
    bonaFeaturesTrain[i] and morphFeaturesTrain[i] equals difference feature 
    number [i] which is of shape (49, 512).

    Parameters
    ----------
    bonaFeaturesTrain: List
        List of difference feature vectors from training-data/bonafide-pairs folder.
    morphFeaturesTrain: List
        List of difference feature vectors from training-data/morphed-pairs folder.
    bonaFeaturesTest: List
        List of difference feature vectors from test-data/bonafide-pairs folder.
    morphFeaturesTest: List
        List of difference feature vectors from test-data/morphed-pairs folder.
    
    Returns
    -------
    morphScores: List
        List of zeroes and ones.
    morphAccuracy: List
        List of the accuracy score for difference features.
    """

    morphScores = []
    morphAccuracy = []
    clf = svm.SVC(kernel = 'linear')

    for i in range(len(bonaFeaturesTrain)): #Loops through the difference features and trains svms.
        featuresTrain = np.concatenate((bonaFeaturesTrain[i], morphFeaturesTrain[i]))
        targetTrain = np.concatenate((np.ones((1, len(bonaFeaturesTrain[0]))), np.zeros((1, len(morphFeaturesTrain[0]))))).flatten()
        featuresTest = np.concatenate((bonaFeaturesTest[i], morphFeaturesTest[i]))
        targetTest = np.concatenate((np.ones((1, len(bonaFeaturesTest[0]))), np.zeros((1, len(morphFeaturesTest[0]))))).flatten() 
 
        clf.fit(featuresTrain, targetTrain)

        morphScores.append(clf.predict(featuresTest))
        morphAccuracy.append(metrics.accuracy_score(targetTest, morphScores[i]))

    return morphScores, morphAccuracy
