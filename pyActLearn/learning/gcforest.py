#!usr/bin/env python
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class gcForest(object):
    max_acc = 0.0
    max_pred_layer = []

    def __init__(self, n_mgsRFtree=30, cascade_test_size=0.2, n_cascadeRF=2,
                 n_cascadeRFtree=101, cascade_layer=np.inf,
                 min_samples_cascade=0.05, tolerance=0.0):
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'cascade_layer', cascade_layer)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)

    def fit(self, X, y):
        _ = self.cascade_forest(X, y)

    def predict_proba(self, X):
        cascade_all_pred_prob = self.cascade_forest(X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)
        return predict_proba

    def predict(self, X):
        pred_proba = self.predict_proba(X=X)
        predictions = np.argmax(pred_proba, axis=1)
        return predictions

    def cascade_forest(self, X, y=None):
        if y is not None:
            setattr(self, 'n_layer', 0)
            test_size = getattr(self, 'cascade_test_size')
            max_layers = getattr(self, 'cascade_layer')
            tol = getattr(self, 'tolerance')
            # test_size = int(np.floor(X.shape[0] * test_size))
            # train_size = X.shape[0] - test_size
            # X_train = X[0:train_size, :]
            # y_train = y[0:train_size]
            # X_test = X[train_size:train_size + test_size, :]
            # y_test = y[train_size:train_size + test_size]
            # X_train, X_test, y_train, y_test = \
            #     train_test_split(X, y, test_size=test_size)
            X_train = X
            X_test = X
            y_train = y
            y_test = y
            self.n_layer += 1
            prf_pred_ref = self._cascade_layer(X_train, y_train)
            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            feat_arr = self._create_feat_arr(X_train, prf_pred_ref)

            self.n_layer += 1
            prf_pred_layer = self._cascade_layer(feat_arr, y_train)
            accuracy_layer = self._cascade_evaluation(X_test, y_test)
            max_acc = accuracy_ref
            max_pred_layer = prf_pred_layer

            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:
            #while accuracy_layer > (accuracy_ref - 0.000001) and \
            #    self.n_layer <= max_layers:
                if accuracy_layer > max_acc:
                    max_acc = accuracy_layer
                    max_pred_layer = prf_pred_layer
                    accuracy_ref = accuracy_layer
                    prf_pred_ref = prf_pred_layer
                    feat_arr = self._create_feat_arr(X_train, prf_pred_ref)
                    self.n_layer += 1
                    prf_pred_layer = self._cascade_layer(feat_arr, y_train)
                    accuracy_layer = self._cascade_evaluation(X_test, y_test)

                if accuracy_layer < accuracy_ref:
                    n_cascadeRF = getattr(self, 'n_cascadeRF')
                    for irf in range(n_cascadeRF):
                        delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                        delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                    self.n_layer -= 1

            print("layer %d - accuracy %f ref %f" % (self.n_layer, accuracy_layer, accuracy_ref))
        else:
            at_layer = 1
            prf_pred_ref = self._cascade_layer(X, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_pred_ref)
                prf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_pred_ref

    def _cascade_layer(self, X, y=None, layer=0):
        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_samples_cascade')

        prf = RandomForestClassifier(
            n_estimators=100, max_features=8,
            bootstrap=True, criterion="entropy", min_samples_split=20,
            max_depth=None, class_weight='balanced', oob_score=True)
        crf = ExtraTreesClassifier(
            n_estimators=100, max_depth=None,
            bootstrap=True, oob_score=True)

        prf_pred = []
        if y is not None:
            # print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)
                probas = prf.oob_decision_function_
                probas += crf.oob_decision_function_
                prf_pred.append(probas)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                probas = prf.predict_proba(X)
                probas += crf.predict_proba(X)
                prf_pred.append(probas)

        return prf_pred

    def _cascade_evaluation(self, X_test, y_test):
        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)
        #print('Layer validation accuracy = {}'.format(casc_accuracy))

        return casc_accuracy

    def _create_feat_arr(self, X, prf_pred):
        swap_pred = np.swapaxes(prf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)

        return feat_arr
