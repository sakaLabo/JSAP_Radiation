import torch
import numpy as np
import matplotlib.pyplot as plt
import sampleLayers

model = sampleLayers.MyModel()
model.load_state_dict(torch.load('./model.pth'))

data = np.loadtxt('data.csv', delimiter=',')
input = data[:,0:2]
correct = data[:,2:4]

testMeshX, testMeshY = np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))
xTestNP = np.stack([testMeshX.reshape(-1), testMeshY.reshape(-1)], axis=1)

xTest = torch.from_numpy(xTestNP).float()
testLoader = torch.utils.data.DataLoader(xTest, batch_size = sampleLayers.BATCH_SIZE)

predicts = np.empty((0,sampleLayers.OUTPUT_NUM), float)

for dataTest in testLoader:
    predicted = model.forward(dataTest) #このモデルにデータを入力して出力を得る
    predicts = np.append(predicts, predicted.data.numpy(), axis=0)

import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

predictMesh0 = np.reshape(predicts[:,0], (testMeshX.shape[0], testMeshY.shape[1]))
ax0 = fig.add_subplot(121, projection='3d')
ax0.plot(input[:,0], input[:,1], correct[:,0], marker="o", linestyle='None', c='black', ms = 3, mew = 3)
ax0.plot_surface(testMeshX, testMeshY, predictMesh0, cmap = "jet", alpha=0.2)

predictMesh1 = np.reshape(predicts[:,1], (testMeshX.shape[0], testMeshY.shape[1]))
ax1 = fig.add_subplot(122, projection='3d', azim = 250)
ax1.plot(input[:,0], input[:,1], correct[:,1], marker="o", linestyle='None', c='black', ms = 3, mew = 3)
ax1.plot_surface(testMeshX, testMeshY, predictMesh1, cmap = "jet", alpha=0.2)

plt.show()