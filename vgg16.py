from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation , Dense, Flatten

def vgg_16():
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3),input_shape = (224,224,3), padding = 'same',\
                     activation = 'relu',name = '1.1-conv' ))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu',name = '1.2-conv' ))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Conv2D(filters = 128,  kernel_size = (3,3), padding = 'same',\
                     activation = 'relu' , name = '2.1-conv'))
    model.add(Conv2D(filters= 128, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '2.2-conv'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '3.1-conv'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '3.2-conv'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '3.3-conv'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '4.1-conv'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '4.2-conv'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '4.3-conv'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '5.1-conv'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '5.2-conv'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3), padding = 'same',\
                     activation = 'relu', name = '5.3-conv'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(1000, activation = 'softmax'))
    
    model.summary()
    
    return model