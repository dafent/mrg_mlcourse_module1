import struct as st
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import itertools
from collections import Counter
from sklearn.metrics import accuracy_score
import argparse
from zipfile import ZipFile
import gzip, shutil
from sys import exit


parser = argparse.ArgumentParser()
parser.add_argument('-x_train_dir')
parser.add_argument('-y_train_dir')
parser.add_argument('-model_output_dir')
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
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

# images
train_images_filename = ungzip(args.x_train_dir)
raw_train = read_idx(train_images_filename)
train_data = np.reshape(raw_train, (60000, 784))
# labels
train_labels_filename = ungzip(args.y_train_dir)
train_label = read_idx(train_labels_filename)


with open(args.model_output_dir, 'w') as f:
    for image in train_data:
        f.write(','.join([str(x) for x in image]))
        f.write('\n')
    for label in train_label:
        f.write(str(label))
        f.write('\n')


norms = 255
train_data = train_data/norms

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.05)


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
            # if count % 500 == 0:
            #print("count: {}, pred_y: {}".format(count, pred_y[count]))
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

KNN.fit(X_train, Y_train)

Y_pred = KNN.predict(X_test)
print(classification_report(Y_test, Y_pred))



