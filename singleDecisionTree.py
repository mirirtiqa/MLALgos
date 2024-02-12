import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

def computeEntropy(y):
  entropy = 0
  if len(y)!=0:
    p1 = len(y[y==1])/len(y)
    if p1 != 0 or p1 != 1:
      entropy = - p1  * np.log2(p1) - (1-p1)*np.log2(1-p1)
    else:
      entropy = 0
  return entropy

computeEntropy(y_train)

def splitDataset(X,nodes,feature):
  leftIndicies = []
  rightIndicies =[]
  for n in nodes:
    if X_train[n][feature]==1:
      leftIndicies.append(n)
    else:
      rightIndicies.append(n)
  return leftIndicies, rightIndicies



root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0
left_indices, right_indices = splitDataset(X_train, root_indices, feature)

def computeInformationGain(X,y, rootIndices,feature):
  leftIndices,rightIndices = splitDataset(X,rootIndices,feature)
  X_node,Y_node = X[rootIndices], y[rootIndices]
  X_left, Y_left = X[leftIndices], y[leftIndices]
  X_right,Y_right = X[rightIndices],y[rightIndices]

  entropyRoot = computeEntropy(Y_node)
  entropyLeft = computeEntropy(Y_left)
  entropyRight = computeEntropy(Y_right)

  wLeft = len(X_left)/len(X_node)
  wRight = len(X_right)/len(X_node)

  weightedEntropy = wLeft * entropyLeft + wRight*entropyRight

  InformationGain = entropyRoot - weightedEntropy

  return InformationGain 



info_gain0 = computeInformationGain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)
    
info_gain1 = computeInformationGain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = computeInformationGain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

def getBestSplit(X,y,node_indices):
  numF = X.shape[1]
  maxig = 0
  bestF = -1

  for f in range(numF):
    ig = computeInformationGain(X,y,node_indices,f)
    if ig> maxig:
      maxig = ig
      bestF = f

  return maxig,bestF


maxig,best_feature = getBestSplit(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

tree = []
def buildTree(X, y, node_indices, branch_name, max_depth, current_depth):
  if current_depth == max_depth:
    formatting = " "*current_depth + "-"*current_depth
    print(formatting, "%s leaf node with indices" % branch_name, node_indices)
    return
  

  maxig,best_feature = getBestSplit(X,y,node_indices)
  tree.append((best_feature,maxig,branch_name,node_indices,current_depth))

  formatting = "-"*current_depth
  print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

  leftIndices,rightIndices = splitDataset(X,node_indices,best_feature)

  buildTree(X,y,leftIndices,"Left",max_depth,current_depth+1)
  buildTree(X,y,rightIndices,"right",max_depth,current_depth+1)






buildTree(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)