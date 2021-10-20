from model import *
from data import *
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4021)])
  except RuntimeError as e:
    print(e)

#Need to preprocess images since DRIVE has size 584*565*3, but we need y32*x32*1
#We change to gray-scale and 608*576*1


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/DRIVE_preprocessed/train','images','labels',data_gen_args,save_to_dir = None)

model = UNet(input_size=(608, 576, 1))
model_checkpoint = ModelCheckpoint('unet_DRIVE.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/DRIVE_preprocessed/test/images")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/DRIVE_preprocessed/test/unet_result",results)