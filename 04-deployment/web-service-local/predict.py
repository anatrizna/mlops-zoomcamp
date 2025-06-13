import pickle


with open('lin_reg.bin', 'rb') as f_in:
    (dictionary_vectoriser, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features
#we need above because we did combined feature for pickup location and drop-off location during training
#df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']

def predict(features):
    X = dictionary_vectoriser.transform(features)
    preds = model.predict(X)
    return float(preds[0])