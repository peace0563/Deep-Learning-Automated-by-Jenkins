import argparse
class Deeplearn:
    def methd(self,epochh,layerr):
        
        import struct
        import numpy as np

        def read_idx(filename):
            """Credit: https://gist.github.com/tylerneylon"""
            with open(filename, 'rb') as f:
                zero, data_type, dims = struct.unpack('>HBB', f.read(4))
                shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


        x_train = read_idx("./fashion_mnist/train-images-idx3-ubyte")
        y_train = read_idx("./fashion_mnist/train-labels-idx1-ubyte")
        x_test = read_idx("./fashion_mnist/t10k-images-idx3-ubyte")
        y_test = read_idx("./fashion_mnist/t10k-labels-idx1-ubyte")


        from keras.utils import np_utils
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
        from keras import backend as K

        # Training Parameters
        batch_size = 8
        epochs = epochh
        # Lets store the number of rows and columns
        img_rows = x_train[0].shape[0]
        img_cols = x_train[1].shape[0]

        # Getting our date in the right 'shape' needed for Keras
        # We need to add a 4th dimenion to our date thereby changing our
        # Our original image shape of (60000,28,28) to (60000,28,28,1)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # store the shape of a single image 
        input_shape = (img_rows, img_cols, 1)

        # change our image type to float32 data type
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Normalize our data by changing the range from (0 to 255) to (0 to 1)
        x_train /= 255
        x_test /= 255

        # Now we one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        num_classes = y_test.shape[1]
        num_pixels = x_train.shape[1] * x_train.shape[2]

        # create model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(BatchNormalization())
        for i in range(layerr-1):
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss = 'categorical_crossentropy',
                      optimizer = keras.optimizers.Adadelta(),
                      metrics = ['accuracy'])
        
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        model.save("mnist_model.h5")
        f = open("score","w")
        g = open("Actual_score.txt","w")
        f.write(str(int(score[1]*100)))
        g.write(str(score[1]*100))
        g.write("Layer: "+str(layerr))
        g.write("Epoch: "+str(epochh))
        g.close()
        f.close()


        
if __name__== '__main__':
    arg = argparse.ArgumentParser(description='Fashion Minist')

    arg.add_argument('-l', '--layers', type = int, default = 1, help = 'Convolutional Layers(Default=1)', choices= range(1,4))

    arg.add_argument('-e', '--epoch', type = int, default = 1, help = 'Epochs(Default=1)')

    args = arg.parse_args()
    deepLearn = Deeplearn()
    deepLearn.methd(args.epoch,args.layers)

