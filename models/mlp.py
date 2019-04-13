from models.model import Model
import keras.models
from keras.layers import Input, Dropout, BatchNormalization, Dense, Activation
from player_profile import PlayerProfile
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf

#from tensorflow.python.keras.callbacks import TensorBoard
from time import time

class MLP(Model):
    def __init__(self, useRaceRatio=True, useRD=True, layers=2, width=200, layerArr=[], dropout=0.2, learning_rate= 1e-3,
                 weight_decay = 5e-4, batch_size=32, max_epochs=100000, modelSaveName="backups/mlp_best.h5", scalerSaveName="backups/mlp_scaler.joblib"):
        if len(layerArr) > 0:
            self.layersSpec = layerArr
        else:
            self.layersSpec = [width for i in range(layers)]

        self.dropout = dropout
        self.learningRate = learning_rate
        self.weightDecay = weight_decay
        self.useRaceRatio = useRaceRatio
        self.useRD = useRD

        p1 = PlayerProfile(name='temp1', race='Zerg', region='')
        p2 = PlayerProfile(name='temp2', race='Terran', region='')
        feats = self.getFeatures(p1, p2)

        self.featSize = len(feats)
        self.input, self.output, self.model = self.buildModel()
        self.saveName = modelSaveName
        self.scalerSaveName = scalerSaveName
        self.batchSize = batch_size
        self.max_epochs = max_epochs

        self.scaler = StandardScaler()
        self.Wsave = self.model.get_weights()

        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                print("Model tf session initialized")
                # for some reason in a flask app the graph/session needs to be used in the init

    def buildModel(self):
        input = Input(shape=(self.featSize,))
        bNorm = BatchNormalization()(input)
        layer = Dense(self.layersSpec[0], kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.weightDecay), bias_regularizer=l2(self.weightDecay))(bNorm)
        # Apply dropout before activation
        drop = Dropout(rate=self.dropout)(layer)
        act = Activation(activation='relu')(drop)
        for i in range(1, len(self.layersSpec)):
            bNorm = BatchNormalization()(act)
            layer = Dense(self.layersSpec[i], kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.weightDecay), bias_regularizer=l2(self.weightDecay))(bNorm)
            drop = Dropout(rate=self.dropout)(layer)
            act = Activation(activation='relu')(drop)
        out = Dense(1, activation='sigmoid')(act)

        model = keras.models.Model(inputs=input, outputs=out)
        adam = Adam(lr=self.learningRate, clipnorm=1.)
        model.compile(optimizer=adam, loss='binary_crossentropy')
        return input, out, model

    def updateRaw(self, features, matches, valFeatures, valMatches, full=False):
        with self.graph.as_default():
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())
                if not full:
                    transformedMatches = np.array([np.array([match[1]]) for match in matches])
                    transformedValMatches = np.array([np.array([match[1]]) for match in valMatches])
                    self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
                    mc = ModelCheckpoint(self.saveName, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
                    #tb = TensorBoard(log_dir="logs/{}".format(time()))
                    self.scaler.fit(features)
                    self.history = self.model.fit(self.scaler.transform(features), transformedMatches, validation_data=(self.scaler.transform(valFeatures), transformedValMatches), batch_size=self.batchSize,
                               epochs=self.max_epochs, verbose=1, callbacks=[self.es, mc])
                else:
                    self.model.set_weights(self.Wsave)
                    # Has already been trained once, retraining with full dataset
                    transformedMatches = np.array([np.array([match[1]]) for match in matches])
                    epochs = self.es.stopped_epoch
                    mc = ModelCheckpoint(self.saveName, monitor='loss', mode='min', save_best_only=True, verbose=1)
                    self.history = self.model.fit(self.scaler.fit_transform(features), transformedMatches, batch_size=self.batchSize,
                               epochs=epochs, verbose=1, callbacks=[mc])
                joblib.dump(self.scaler, self.scalerSaveName)

    def fitScaler(self, features):
        self.scaler.fit(features)
        joblib.dump(self.scaler, self.scalerSaveName)

    def loadBackup(self):
        with self.graph.as_default():
            with self.session.as_default():
                self.scaler = joblib.load(self.scalerSaveName)
                self.model = keras.models.load_model(self.saveName)

    def predict(self, profile1, profile2):
        with self.graph.as_default():
            with self.session.as_default():
                features1 = profile1.getFeatures(profile2, useRaceRatio=self.useRaceRatio, useRD=self.useRD)
                features2 = profile2.getFeatures(profile1, useRaceRatio=self.useRaceRatio, useRD=self.useRD)
                features = np.array(features1 + features2).reshape(1, -1)
                pred = self.model.predict(self.scaler.transform(features))[0]
        return np.array([1-pred, pred])

    def predictBatch(self, profiles):
        Xs = []
        for i in range(0, len(profiles)):
            features1 = profiles[i][0].getFeatures(profiles[i][1], useRaceRatio=self.useRaceRatio, useRD=self.useRD)
            features2 = profiles[i][1].getFeatures(profiles[i][0], useRaceRatio=self.useRaceRatio, useRD=self.useRD)

            # Update with profiles in both slots to prevent strange asymmetrical model
            Xs.append(features1 + features2)
        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model.predict(self.scaler.transform(Xs))
        transformedPreds = []
        for i in range(len(preds)):
            pred = preds[i]
            transformedPreds.append(np.array([1-pred, pred]))
        return np.array(transformedPreds)[:,:,0]
    def getFeatures(self, profile1, profile2):
        features1 = profile1.getFeatures(profile2, useRaceRatio=self.useRaceRatio, useRD=self.useRD)
        features2 = profile2.getFeatures(profile1, useRaceRatio=self.useRaceRatio, useRD=self.useRD)
        return features1 + features2

    def predictRaw(self, features):
        with self.graph.as_default():
            with self.session.as_default():
                features = np.array(features).reshape(1, -1)
                pred = self.model.predict(self.scaler.transform(features))[0]
        return np.array([1 - pred, pred])

    def predictBatchRaw(self, features):
        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model.predict(self.scaler.transform(features))
        transformedPreds = []
        for i in range(len(preds)):
            pred = preds[i]
            transformedPreds.append(np.array([1 - pred, pred]))
        return np.array(transformedPreds)[:,:,0]