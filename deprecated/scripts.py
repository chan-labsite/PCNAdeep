import pandas as pd
import os
import numpy as np
from pcnaDeep.refiner import Refiner
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import f1_score

root = '/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/test/10A_cpd'
prefix = '10A_cpd'

#%% Use broken track to generate ground truth
dt = pd.read_csv(os.path.join(root, prefix+'_tracks.csv'))
r = Refiner(dt.copy(), mode='TRAIN')
mitosis_broken, mt_dic = r.doTrackRefine()
mitosis_broken.to_csv(os.path.join(root, prefix+'_mitosis_broken.csv'), index=0)

#%% Generate feature map
mt_lookup_array = np.array(pd.read_csv(os.path.join(root, prefix+'_mitosis_lookup.txt'), header=0))
X, y, sample_id = r.get_SVM_train(np.array(mt_lookup_array))
X = pd.DataFrame(np.array(X))
X['y'] = y

#%% Append sample ID for inspection
X['par'] = sample_id[:,0]
X['daug'] = sample_id[:,1]

#%% Merge with old feature map or direct output
X.to_csv(os.path.join(root, prefix+'_svm_train.txt'), index=False, header=False)

#%% Try out model fitting

table = pd.read_csv('/models/SVM_train.txt',
                    header=None)
table = np.array(table)
X = table[:,:table.shape[1]-1]
y = table[:,table.shape[1]-1]
# Oversample positive instances
smote = BorderlineSMOTE(random_state=1, kind='borderline-1')
X, y = smote.fit_resample(X, y)
# Normalize
s = RobustScaler()
X = s.fit_transform(X)
model = SVC(kernel='rbf', C=100, gamma=1, probability=True, class_weight='balanced')
model.fit(X, y)

#%% Evaluate the model
out = np.round(model.predict_proba(X))[:,1]
#print(out[np.where(out[:,1]>0.5),1])
#plt.hist(out[:,1])
f1_score(y, out)

#%% Visualize the training set
import matplotlib.pyplot as plt
sub = X[:,[0,1,2]]  # spatial and temporal distance and parent mitosis score, the first three features
plt.scatter(sub[:,0], sub[:,1], c=y, s=(- min(sub[:,2]) + sub[:,2])*1, alpha=0.2, cmap='coolwarm')
plt.show()
