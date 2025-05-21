import numpy as np
import os
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


data_dir = 'clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(data_dir, category)):
        img_path = os.path.join(data_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

parameters = {
    'gamma': [0.01, 0.001, 0.0001],
    'C': [1, 10, 100, 1000],
}

model = GridSearchCV(SVC(), parameters, n_jobs=-1, verbose=1)
model.fit(x_train, y_train)

best = model.best_estimator_
y_predict = best.predict(x_test)
print(classification_report(y_test, y_predict))
pickle.dump(best, open('./mymodel.p', 'wb'))