import sys
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score


if len(sys.argv) < 4:
    print('Needs 3 arguments - \n'
          '1. Batch size during training\n'
          '2. Batch size during testing\n'
          '3. No. of epochs\n')
    exit(0)

SEED = 10
np.random.seed(SEED)

# command line both for train and test
DATADIR = '/store/causalIR/model-aware-qpp/input_data/'   # (1)

# this is annoying... but this is how Conv2D layer in Keras works!
# A matrix is treated a grayscale image, i.e. am image with num_channels = 1
NUMCHANNELS = 1
# HIDDEN_LAYER_DIM = 16    # (2)
# Num top docs (Default: 10)
K = 10   # (5)
# M: bin-size (Default: 30)
M = 120  # (6)
BATCH_SIZE_TRAIN = int(sys.argv[1])   # (7 - depends on the total no. of ret docs)
BATCH_SIZE_TEST = int(sys.argv[2])
EPOCHS = int(sys.argv[3])  # (8)


class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document

    def __init__(self, qid, dataPathBase=DATADIR):
        self.qid = qid
        histFile = "{}/{}.hist".format(dataPathBase, self.qid)
        # df = pd.read_csv(histFile, delim_whitespace=True, header=None)
        # self.matrix = df.to_numpy()
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 4:]


class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            self.qid_b = l[1]

    def __str__(self):
        return "({}, {})".format(self.qid_a, self.qid_b)

    def getKey(self):
        return "{}-{}".format(self.qid_a, self.qid_b)


# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}

        with open(idpairLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance

allPairs_train = PairedInstanceIds(DATADIR + 'train_input/qid_ap.pairs')   # (3)
allPairsList_train = list(allPairs_train.data.values())

allPairs_test = PairedInstanceIds(DATADIR + 'test_input/qid_ap.pairs')    # (4)
allPairsList_test = list(allPairs_test.data.values())
# np.random.shuffle(allPairsList)
# num_pairs = len(allPairsList)

# TRAIN_RATIO=0.8
# num_training = int(TRAIN_RATIO * num_pairs)
#
# # get the ids
# train_pairs = allPairsList[0:num_training]
# # print('TRAIN SIZE : ', len(train_pairs))
# test_pairs = allPairsList[num_training:]
# # print('TEST PAIRS : ', len(test_pairs))

print ('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
print ('{}/{} pairs for training'.format(len(allPairsList_test), len(allPairsList_test)))

'''
The files need to be residing in the folder data/
Each file is a matrix of values that's read using 
'''

class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR, batch_size=BATCH_SIZE_TRAIN, dim=(K, M, NUMCHANNELS)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)) for i in range(2)]
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)

            w, h = a_data.matrix.shape
            a_data.matrix = a_data.matrix.reshape(w, h, 1)

            # print('B matrix size : ', b_data.matrix.shape)
            b_data.matrix = b_data.matrix.reshape(w, h, 1)

            X[0][i,] = a_data.matrix
            X[1][i,] = b_data.matrix
            Y[i] = paired_instance.class_label

        return X, Y


class PairCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR, batch_size=BATCH_SIZE_TEST, dim=(K, M, NUMCHANNELS)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print('INDEXES : ', len(indexes))

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        # print('LIST IDS : ', len(list_IDs))

        # Generate data
        X = self.__data_generation(list_IDs)
        # print('X value : ', X)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)) for i in range(2)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)

            w, h = a_data.matrix.shape
            a_data.matrix = a_data.matrix.reshape(w, h, 1)
            b_data.matrix = b_data.matrix.reshape(w, h, 1)

            X[0][i,] = a_data.matrix
            X[1][i,] = b_data.matrix

        return X


def build_siamese(input_shape):
    input_a = Input(shape=input_shape, dtype='float32')
    input_b = Input(shape=input_shape, dtype='float32')

    matrix_encoder = Sequential()
    matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    matrix_encoder.add(Dense(500))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Conv2D(64, (3, 3), activation='relu'))
    matrix_encoder.add(Dense(50))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Flatten())
    matrix_encoder.add(Dropout(0.2))
    matrix_encoder.add(Dense(10, activation='sigmoid'))

    encoded_a = matrix_encoder(input_a)
    encoded_b = matrix_encoder(input_b)
    merged_vector = concatenate([encoded_a, encoded_b], axis=-1)

    # And add a logistic regression (2 class - sigmoid) on top
    # used for backpropagating from the (pred, true) labels
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    siamese_net = Model([input_a, input_b], outputs=predictions)
    return siamese_net

siamese_model = build_siamese((K, M, 1))
siamese_model.compile(loss = keras.losses.BinaryCrossentropy(),
                      optimizer = keras.optimizers.Adam(),
                      metrics=['accuracy'])
siamese_model.summary()

training_generator = PairCmpDataGeneratorTrain(allPairsList_train, dataFolder=DATADIR+'train_input/')
siamese_model.fit_generator(generator=training_generator,
                            use_multiprocessing=True,
                            epochs=EPOCHS,
                            workers=4)
                            # validation_split=0.2,
                            # verbose=1)
# siamese_model.save_weights('/store/causalIR/model-aware-qpp/foo.weights')

test_generator = PairCmpDataGeneratorTest(allPairsList_test, dataFolder=DATADIR+'test_input/')
predictions = siamese_model.predict(test_generator)  # just to test, will rerank LM-scored docs
print('predict ::: ', predictions)
print('predict shape ::: ', predictions.shape)
with open(DATADIR + "26march.res", 'w') as outFile:     # (9)
    i = 0
    for entry in test_generator.paired_instances_ids:
        if predictions[i][0] >= 0.45:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '1\n')
        else:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '0\n')
        i += 1
outFile.close()

# measure accuracy
gt_file = np.genfromtxt(DATADIR + 'test_input/qid_ap.pairs.gt', delimiter='\t')    # (10)
actual = gt_file[:, 2:]
# print(type(actual))

predict_file = np.genfromtxt(DATADIR + '26march.res', delimiter='\t')
predict = predict_file[:, 3:]
# print(type(predict))

score = accuracy_score(actual, predict)
print('Accuracy : ', round(score, 4))




