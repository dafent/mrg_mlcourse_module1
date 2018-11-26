import struct as st
import numpy as np
from sklearn.metrics import classification_report
import itertools
from collections import Counter
from sklearn.metrics import accuracy_score
import argparse
from zipfile import ZipFile
import gzip, shutil

parser = argparse.ArgumentParser()
parser.add_argument('-x_test_dir')
parser.add_argument('-y_test_dir')
parser.add_argument('-model_input_dir')
args = parser.parse_args()


def ungzip(filename):
    new_filename = filename[:-3]
    with gzip.open(filename, 'r') as f_in, open(new_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return new_filename


def read_idx(filename):
    with open (filename, 'rb') as f:
        zero, data_types, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype = np.uint8).reshape(shape)


zipped_model = ZipFile(args.model_input_dir, 'r')
zipped_model.extractall('./')
zipped_model.close()
new_model_input_dir = args.model_input_dir[:-4].split('/')[-1]

with open(new_model_input_dir, 'r') as f:
    input_data = f.read().split('\n')
    train_data = np.array([[int(pixel) for pixel in image.split(',')] for image in input_data[:60000]]) #np.reshape(raw_train, (60000, 784))
    train_label = np.array([int(label) for label in input_data[60000:] if label != '']) #read_idx([args[y_train_dir])#"train-labels.idx1-ubyte")


# images
test_images_filename = ungzip(args.x_test_dir)
raw_test = read_idx(test_images_filename)
test_data = np.reshape(raw_test, (10000, 784))
# labels
test_labels_filename = ungzip(args.y_test_dir)
test_label = read_idx(test_labels_filename)

norms = 255
X = train_data/norms
Y = train_label
X_test = test_data/norms
Y_true = test_label
print("data preparation ends")

class My_knn:
    
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        
    def fit(self, train_x, train_y):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        
    def predict(self, test_x):
        
        count = 0
        pred_y = []
        
        for elem in test_x:
            
            dist_kmin = []
            dist = []
            pred_y_counter = []
            
            for ind, x in enumerate(self.train_x):
                dist.append((My_knn.distance(elem, x), ind))
                
            dist_kmin = sorted(dist, key = lambda x: x[0])[:self.k]
            pred_y_count = Counter(self.train_y[[item[1] for item in dist_kmin]])
            pred_y.append(pred_y_count.most_common(1)[0][0])
            if count % 100 == 0:
                print("image number {} is done".format(count))
            count += 1
            
        return pred_y
    
    @staticmethod
    def distance(im_1, im_2):
        p = 3
        im_1 = np.array(im_1)
        im_2 = np.array(im_2)
        dist1 = np.linalg.norm(im_1 - im_2)
        return dist1


KNN = My_knn(5)

KNN.fit(X, Y)

Y_pred = KNN.predict(X_test)
print(classification_report(Y_true, Y_pred))

