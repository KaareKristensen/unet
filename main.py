from model import *
from data import *
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Need to preprocess images since CHASEDB1 has size 584*565*3, but we need y32*x32*1
#We change to gray-scale and 608*576*1
trainfolder = input("Input training folder containing images and labels folders:\n")
testfolder = input("Input test folder containing image folder, or empty string for no tests:\n")

useDataAugmentation = input("Use data augmentation, either set to 'true' or 'false':\n")
saveModelFilename = input("Input filename for the model if you want it saved:\n")
if not useDataAugmentation in ['true', 'false']:
    print("useDataAugmentation must be either 'true' or 'false' and not " + useDataAugmentation)

data_gen_args = dict()

if useDataAugmentation == 'true':
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')


myGene = trainGenerator(2,'data/' + trainfolder +'/train','images','labels',data_gen_args,save_to_dir = None)

model = UNet(input_size=(608, 576, 1))
model_checkpoint = []
if saveModelFilename != "":
    model_checkpoint = [ModelCheckpoint(saveModelFilename + ".hdf5", monitor='loss',verbose=1, save_best_only=True)]

model.fit(myGene,steps_per_epoch=1000,epochs=10,callbacks=model_checkpoint)

if (testfolder != ""):
    testImagePath = "data/" + testfolder + "/test/images"
    testGene = testGenerator(testImagePath)
    results = model.predict(testGene,steps=len(os.listdir(testImagePath)),verbose=1)

    # maybe threshold the results

    # Calc error on result

    print("RESULT SHAPE:")
    print(results.shape)
    print()

    saveResult("data/" + testfolder + "/test/unet_result",results)