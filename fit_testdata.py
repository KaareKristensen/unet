from model import *
from data import *
import tensorflow as tf



testfolder = input("Input test folder containing image folder:\n")
trainedModel = input("Input the filename of the trained model weights to use:\n")
saveResultsFolder = input("Input the folder name where you want the results of predict stored, if empty doesn't save results:\n")


model = UNet(pretrained_weights=trainedModel, input_size=(608, 576, 1))


testImagePath = "data/" + testfolder + "/test/images"
testGene = testGenerator(testImagePath)
metricDict = model.evaluate(testGene, steps=len(os.listdir(testImagePath)), return_dict=True, verbose=1)
testGene = testGenerator(testImagePath)
results = model.predict(testGene,steps=len(os.listdir(testImagePath)),verbose=1)

# maybe threshold the results

print("LOSS AND METRICS:")
print(metricDict)
print()


print("RESULT SHAPE:")
print(results.shape)
print()
if saveResult != "":
    saveResult("data/" + saveResultsFolder,results)