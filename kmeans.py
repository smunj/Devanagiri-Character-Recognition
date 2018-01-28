import os
import cv2
import numpy as np
import tqdm
from sklearn.cluster import MiniBatchKMeans


k = 2
for f in tqdm.tqdm(os.listdir('train_images/')): 
  fname = os.path.join('train_images', f)
  sname = os.path.join('q_train_images', f)
  
  img = cv2.imread(fname)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  rows, cols = img.shape[:-1]

  fimg = img.reshape(rows * cols, -1)
  norm = [fimg[:, i].max() - fimg[:, i].min() for i in range(img.shape[-1])]

  fimg = fimg.astype(np.float32)
  fimg /= norm 

  kmeans = MiniBatchKMeans(n_clusters=k, 
                           verbose=False, 
                           max_no_improvement=20)
  labels = kmeans.fit_predict(fimg)
  out = kmeans.cluster_centers_[labels]
  out = out.reshape(rows, cols, -1)
  out *= norm
  out = out.astype(np.uint8)
  out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

  cv2.imwrite(sname, out)

