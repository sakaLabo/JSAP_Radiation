import numpy as np
import torch
import sampleLayers

data = torch.from_numpy( np.loadtxt('data.csv', delimiter=',') ).float()
trainData, valData = torch.utils.data.random_split(data, [int(data.size(0) * 0.9), int(data.size(0) * 0.1)])

trainLoader = torch.utils.data.DataLoader(trainData, batch_size = sampleLayers.BATCH_SIZE, shuffle=True)
valLoader = torch.utils.data.DataLoader(valData, batch_size = sampleLayers.BATCH_SIZE)

model = sampleLayers.MyModel()
lossFunc = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.03)

epochNum = 300
lossHistory = np.zeros((2, epochNum))

for epoch in range(epochNum):
    for tData in trainLoader:
        train, correct = tData[:,0:2], tData[:,2:4]
        predicted = model(train)
        loss = lossFunc(predicted, correct)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossHistory[0, epoch] += loss.item()
            
    for vData in valLoader:
        val, correct = vData[:,0:2], vData[:,2:4]
        predicted = model(val)
        lossHistory[1, epoch] += lossFunc(predicted, correct).item()
                
    print(f'[Epoch{epoch:3d}]' f' trainLoss{ lossHistory[1, epoch]/len(trainLoader):.5f}'  f' valLoss{lossHistory[1, epoch]/len(valLoader):.5f}')

np.savetxt('trainLoss.csv', lossHistory, delimiter=',')
torch.save(model.state_dict(), './model.pth')