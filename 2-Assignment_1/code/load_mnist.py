import os
import numpy as np
import imageio
import glob


# temp_paths
NP_DATA_STORE_PATH = "MNIST"

def load_mnist():
    if os.path.isdir(f"{NP_DATA_STORE_PATH}/npz"):
        # load MNISt from numpy save done in previous iteration if any
        X_train = np.load(f"{NP_DATA_STORE_PATH}/npz/x_train.npy")
        Y_train = np.load(f"{NP_DATA_STORE_PATH}/npz/y_train.npy")
        X_test = np.load(f"{NP_DATA_STORE_PATH}/npz/x_test.npy")
        Y_test = np.load(f"{NP_DATA_STORE_PATH}/npz/y_test.npy")
    
    else:        
        # Loads the MNIST dataset from png images
        NUM_LABELS = 10        
        # create list of image objects
        test_images = []
        test_labels = []    
        
        for label in range(NUM_LABELS):
            for image_path in glob.glob(f"{NP_DATA_STORE_PATH}/Test/{str(label)}/*.png"):
                image = imageio.imread(image_path)
                test_images.append(image)
                letter = [0 for _ in range(0,NUM_LABELS)]    
                letter[label] = 1
                test_labels.append(letter)  
                
        # create list of image objects
        train_images = []
        train_labels = []    
        
        for label in range(NUM_LABELS):
            for image_path in glob.glob(f"{NP_DATA_STORE_PATH}/Train/{str(label)}/*.png"):
                image = imageio.imread(image_path)
                train_images.append(image)
                letter = [0 for _ in range(0,NUM_LABELS)]    
                letter[label] = 1
                train_labels.append(letter)                  
                
        X_train= np.array(train_images).reshape(-1,784)/255.0
        Y_train= np.array(train_labels)
        X_test= np.array(test_images).reshape(-1,784)/255.0
        Y_test= np.array(test_labels)
            
        # save arrays
        os.makedirs(f"{NP_DATA_STORE_PATH}/npz")
        np.save(f"{NP_DATA_STORE_PATH}/npz/x_train", X_train)
        np.save(f"{NP_DATA_STORE_PATH}/npz/y_train", Y_train)
        np.save(f"{NP_DATA_STORE_PATH}/npz/x_test", X_test)
        np.save(f"{NP_DATA_STORE_PATH}/npz/y_test", Y_test)

    return X_train, Y_train, X_test, Y_test
