from model import *
from data import *
import tensorflow as tf



testfolder = input("Input test folder containing 'images' folder and 'labels' folder:\n")
trainedModel = input("Input the filename of the trained model weights to use:\n")
saveResultsFolder = input("Input the folder name where you want the results of predict stored, if empty doesn't save results:\n")


model = UNet(pretrained_weights=trainedModel, input_size=(608, 576, 1))


testImagePath = "data/" + testfolder + "/images"
numberOfImages = len(os.listdir(testImagePath))
evalGene = trainGenerator(1,'data/' + testfolder,'images','labels',{},save_to_dir = None)
metricDict = model.evaluate(evalGene, steps=numberOfImages)

# maybe threshold the results

print("LOSS AND METRICS:")
print(metricDict)
print()

if saveResultsFolder != "":
    testGene = testGenerator(testImagePath)
    results = model.predict(testGene,steps=len(os.listdir(testImagePath)),verbose=1)
    print("RESULT SHAPE:")
    print(results.shape)
    print()
    saveResult("data/" + saveResultsFolder,results)