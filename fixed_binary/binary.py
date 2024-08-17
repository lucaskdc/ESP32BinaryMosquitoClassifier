
import csv
import glob
import os
import matplotlib.pyplot as plt

import librosa
import numpy as np
from keras import Sequential
from keras import Model
from keras.layers import MaxPooling2D, Conv2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as skm

from keras.models import model_from_json

def windows(data, window_size):
  start = 0
  while start < len(data):
    yield start, start + window_size
    start += (window_size // 2)

def extract_features( sub_dirs, file_ext="*.wav"):
  print("Extract features from {} list: {}", file_url, sub_dirs)
  window_size = hop_length * (frames - 1)
  log_specgrams = []
  labels = []



  for l, sub_dir in enumerate(sub_dirs):
    for fn in glob.glob(os.path.join(file_url, sub_dir, file_ext)):
      sound_clip, _ = librosa.load(fn, sr=sample_rate)
      print('Extracting features from: ' + fn)
      label = fn.split('/')[-2]
      for (start, end) in windows(sound_clip, window_size):
        if (len(sound_clip[start:end]) == window_size):
          signal = sound_clip[start:end]
          melspec = librosa.feature.melspectrogram(y=signal, n_mels=bands, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
          logspec = librosa.power_to_db(melspec, ref=np.max)
          logspec = logspec / 80 + 1
          #logspec = logspec.T.flatten()[:, np.newaxis].T
          logspec = logspec.flatten()[:, np.newaxis].T
          log_specgrams.append(logspec)
          labels.append(label)
  features = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
  np_labels = np.array(labels, dtype=np.int)
  unique, counts = np.unique(np_labels, return_counts=True) # to check the number of samples for each class

  return np.array(features), np_labels

def load_data():
  calculate_features_now = DEF_CALCULATE_FEATURES_NOW
  if calculate_features_now:
    tr_sub_dirs = ["0", "1"]
    #tr_sub_dirs = ["aedes_aegypti", "aedes_albopictus"]
    tr_features, tr_labels = extract_features(tr_sub_dirs)
    np.savez(file_url+'\\features.npz', tr_features, tr_labels)
    return tr_features, tr_labels
  else:
    npread = np.load(file_url+'\\features.npz')
    return npread['arr_0'], npread['arr_1']
    
def create_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(bands, frames, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(2, activation='sigmoid'))
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.summary()
  
  show_all_outputs = False
  if show_all_outputs:
    intermediate_model = Model(inputs=model.layers[0].input, \
                              outputs=[l.output for l in model.layers])
    intermediate_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    intermediate_model.summary()
  
    return intermediate_model

  return model

def train_and_evaluate_model(model, xtrain, ytrain, xval, yval):
  print(ytrain.shape)
  print(ytrain)

  fit_now = DEF_TRAIN_NOW
  if (fit_now):
    model.fit(xtrain, ytrain,  batch_size=32, epochs=10, verbose=2)
    with open(os.path.join(file_url,"model_binary.json"), "w") as json_file:
      model_json = model.to_json()
      json_file.write(model_json)
      model.save_weights(os.path.join(file_url,"model_binary.h5"))
  else: 
    with open(os.path.join(file_url,"model_binary.json"), 'r') as json_file:
      loaded_model_json = json_file.read()
      json_file.close()
      #model = model_from_json(loaded_model_json)
      model.load_weights(os.path.join(file_url,"model_binary.h5"))

  #y_predicted_classes = model.predict_classes(xval)
  #y_predicted_probability = model.predict(xval)

  all_outputs = model.predict(xval)
  if type(all_outputs) == list:
    y_predicted_probability = all_outputs[-1]
  else:
    y_predicted_probability = all_outputs
  y_predicted_classes = np.argmax(y_predicted_probability, axis=1)

  with open(file_url+'\\'+'summary.txt', 'w') as fsummary:
    fsummary.write('Feature shape: {}\n'.format(xval.shape[1:]))
    fsummary.write('Output shape: {}\n'.format(y_predicted_probability.shape[1:]))
    fsummary.write('qty: {}'.format(xval.shape[0]))

  file_x_all = open(file_url+'\\feature-result\\ALL'+str(xval.shape[0])+'-x.bin_input', 'wb')
  file_y_all = open(file_url+'\\feature-result\\ALL'+str(xval.shape[0])+'-y.bin_input', 'wb')
  for i in range(0,xval.shape[0]):
    tempx = xval[i].flatten(order='C')
    file_x_all.write(tempx.tobytes())
    with open(file_url+'\\feature-result\\'+str(i)+'-x.bin_input', 'wb') as filex:
      filex.write(tempx.tobytes())
      filex.close()

    if type(all_outputs) == list:
      for (iter_output, out) in enumerate(all_outputs[:-1]):
        if i == 0:
          print(out[0].shape)
        with open(file_url+'\\feature-result\\'+str(i)+'-'+str(iter_output+1)+'.bin_inter', 'wb') as fileint:
          temp = out[i].flatten(order='C')
          fileint.write(temp.tobytes())
          fileint.close()
    
    tempy = y_predicted_probability[i].flatten(order='C')
    file_y_all.write(tempy.tobytes())
    with open(file_url+'\\feature-result\\'+str(i)+'-y.bin_output', 'wb') as filey:
      filey.write(tempy.tobytes())
      filey.close()

  #with np.printoptions(threshold=np.inf):
  print("Y predicted: ")
  print("Type of y_predicted_classes: {}", type(y_predicted_probability))
  print("Type of y_predicted_classes[0]: {}", type(y_predicted_probability[0])) if len(y_predicted_probability) > 0 else 0
  print(y_predicted_probability)
  conf_matrix = skm.confusion_matrix(yval, y_predicted_classes, labels=[1, 0])
  print("Confusion matrix: \n"  + str(conf_matrix))
  score = model.evaluate(xval, yval, verbose=0)
  print("Accuracy: %.2f%%" % ( score[1] * 100))
  precision = skm.precision_score(yval, y_predicted_classes, pos_label=0)
  print("Precision: %.2f%%" % (precision*100))
  recall = skm.recall_score(yval, y_predicted_classes, pos_label=0)
  print("Recall: %.2f%%" % (recall*100))
  f1_score = skm.f1_score(yval, y_predicted_classes, pos_label=0)
  print("F1-score: %.4f" % f1_score)
  auroc = skm.roc_auc_score(yval, y_predicted_classes)
  print("AUROC: %.4f" % auroc)

  return score[1], precision, recall, f1_score

seed = 123
np.random.seed(seed)  # for reproducibility

file_url = '.'

DEF_CALCULATE_FEATURES_NOW = True
DEF_TRAIN_NOW = True
DEF_JUST_FIRST_FOLD = True

bands = 60
frames = 40
hop_length = 256
n_fft = 1024

sample_rate = 8000

n_folds = 10
#n_folds = 2
X, Y = load_data()

print("Len X: {}", len(X))
print("Len Y: {}", len(Y))
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for i, (train, test) in enumerate(skf.split(X, Y)):
  print("Running Fold", i + 1, "/", n_folds)
  model = None # Clearing the NN.
  model = create_model()

  # Generate batches from indices
  xtrain, xval = X[train], X[test]
  ytrain, yval = Y[train], Y[test]
  accuracy, precision, recall, f1 = train_and_evaluate_model(model, xtrain, ytrain, xval, yval)
  accuracy_scores.append(accuracy)
  precision_scores.append(precision)
  recall_scores.append(recall)
  f1_scores.append(f1)

  #STOP
  if DEF_JUST_FIRST_FOLD:
    break

print("Accuracy scores: " + str(accuracy_scores))
print("Precision scores: " + str(precision_scores))
print("Recall scores: " + str(recall_scores))
print("F1 scores: " + str(f1_scores))

csv_filename = "binary.csv"

with open(csv_filename, 'w', newline='') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter=',')
  csv_writer.writerow(['accuracy', 'precision', 'recall', 'f1_score'])

  for i in range(len(accuracy_scores)):
    csv_writer.writerow([accuracy_scores[i], precision_scores[i], recall_scores[i], f1_scores[i]])
