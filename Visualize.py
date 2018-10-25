from preprocess import *
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

model = load_model('model.cnn')
print(model.summary)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print('Sample prediction')
# print(predict('./data/bed/004ae714_nohash_1.wav', model=model))
