##############################################################
#
# IMPORT MODULES
#
##############################################################

import pandas as pd
import numpy as np
from skrebate import ReliefF
#from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
from numpy import savetxt
import warnings



##############################################################
#
# IMPORT DATA
#
##############################################################

DATA = pd.read_csv("C:/Users/Inigo/Documents/PRECISESADS_DATA/WHOLE_BLOOD_ALL.tsv", sep='\t') # Load RNAseq raw counts database
LABELS = pd.read_csv("C:/Users/Inigo/Documents/PRECISESADS_DATA/CS.QC.Info.csv", sep=';') # Load metadata from precieseads



##############################################################
#
# QUALITY CONTROL IN DATA
#
##############################################################

# DROP ALL ZEROS GENES
DATA = DATA.set_index(DATA.columns[0]) # Sets the gene IDs as indexes and deletes that first string column
DATA.sum(axis=1) # Shows the sum of all genes in raw counts
DATA_QC = DATA.loc[(DATA!=0).any(axis=1)] # Drops the genes that have 0 presence in all samples
DATA_QC.sum(axis=1) # Show that only presence genes have been kept

# DROP MINIMUN PRESENCE GENES
groups = 8 # Sets the groups that we find in the metadata
presence = 4 # Sets a minimun threshold of presence in the samples of one group
DATA_dropped = DATA_QC[DATA_QC.sum(axis=1) >= (DATA_QC.shape[1]/groups*presence)] # Deletes genes that do not have a minimun of 4 counts in the samples of one group
DATA_dropped_ = DATA_QC[DATA_QC.var(axis=1) >= 1] # Deletes genes that do have a minimun of variance 1
print(DATA_dropped.sum(axis=1))

# DROP LOW VARIANCE GENES
Counts = DATA_QC.sum(axis=1)
Varience = DATA_QC.var(axis=1, numeric_only = True)



##############################################################
#
# CHECK FOR DATA AND METADATA INTERSECTION
#
##############################################################

samples_labeled = DATA_dropped.columns.intersection(list(map(str, LABELS["OMICID"].values))) # Intersection of samples between the data and the metadata
X_train = DATA_dropped[samples_labeled] # Data only with labeled samples
Y_train = LABELS.loc[LABELS["OMICID"].isin(map(int, samples_labeled))] # Metadata only with samples that have data
sum(X_train.columns == list(map(str, Y_train["OMICID"].values))) # Makes sure that the data and metadata samples are in the same order
Y_train = Y_train.set_index(Y_train["OMICID"]) # Sets the gene IDs as indexes and deletes that string column



##############################################################
#
# MACHINE LEARNING FORMAT
#
##############################################################

x_train = X_train.T.values.astype('float64') # Raw data with genes as features and float array format
y_train = Y_train["Diagnosis"].values # Diagnosis set as classification label



##############################################################
#
# RELIEF FEATURE SELECTION
#
##############################################################

reliefFS = ReliefF(n_neighbors=10, n_jobs = -1) # RelieF algorithm set to all variables and only 10 neigbours
warnings.filterwarnings("ignore") # Future warnings arise in RelieF application
reliefFS.fit(x_train,y_train) # Apply RelieF to data
relief_scores = reliefFS.feature_importances_ # Extract feature importance

# SAVE RELIEF SCORES
savetxt('C:/Users/Inigo/Documents/relief_scores.tsv', relief_scores, delimiter='\t')



##############################################################
#
# RELIEF FEATURE SUBSET SELECTION
#
##############################################################

#RelieF = pd.read_csv("C:/Users/Inigo/Documents/relief_scores.tsv", sep='\t') # Import RelieF feature scores saved
RelieF = relief_scores
reverse_array = np.sort(np.concatenate(RelieF.values))[::-1] # Order descending RelieF scores

# Maximun difference in importance selection
#np.diff(reverse_array) == min(np.diff(reverse_array))

# Low presence/low variance arbitrary cutoff
RelieF[RelieF>0.1155]
RelieF_named = np.vstack([RelieF, X_train.index.values]) # Labels the features
Relief_named_filtered = RelieF_named[:,relief_scores>0.1155] # Filters the features by cutoff

# SAVE RELIEF SUBSET
pd.DataFrame(Relief_named_filtered).to_csv("C:/Users/Inigo/Documents/relief_variables.csv")



