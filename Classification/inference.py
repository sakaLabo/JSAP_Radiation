import torch
import numpy as np
import matplotlib.pyplot as plt
import sampleLayers

model = sampleLayers.MyModel()
model.load_state_dict(torch.load('./model.pth'))

data = np.loadtxt('data.csv', delimiter=',')
input = data[:,0:2]
correct = data[:,2]
CLASS_COLOR = ['orange', 'blue']

fig = plt.figure()
for i in range(len(input)):
    plt.scatter(input[i][0], input[i][1], c = CLASS_COLOR[int(correct[i])])

def oncpaint(event):
    if event.button == 1:
        dataTest = torch.tensor([[float(event.xdata), float(event.ydata)]])

        labelPred = model.forward(dataTest) #このモデルにデータを入力して出力を得る
        maxIdx = torch.argmax(labelPred.detach().clone())

        dataTestNumpy = dataTest.to('cpu').detach().numpy().copy()
        plt.scatter(dataTestNumpy[:,0], dataTestNumpy[:,1], c = CLASS_COLOR[maxIdx], marker="x")
        fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', oncpaint)
plt.show()