import keras
from SoundDataGenerator import SoundDataGenerator
from DataAug_misc import target_size_calc
import matplotlib.pyplot as plt

def build_model(input_shape):
    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(5, activation='softmax'))

    return model

def plot_history(history):
    fig = plt.figure()
    
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="test accuracy")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    
    plt.show()
    
    fig.savefig('OptRMS.png')

if __name__ == "__main__":

    class_dir = 'sig_noise_data/Signal/'

    #possible modes: 'mfcc', 'mel_spec', 'power_spec'
    output_mode = 'mfcc'
    input_shape = target_size_calc(output_mode)
    
    train_datagen = SoundDataGenerator(validation_split=0.2, output_mode=output_mode, noise_aug=True, same_class_aug=True,
                                       noise_dir='sig_noise_data/Noise', time_shift=True, pitch_shift=True)
    
    training_set = train_datagen.flow_from_directory(class_dir,
                                                     batch_size = 8,
                                                     class_mode = 'categorical',
                                                     subset = 'training', seed=10)
    
    valid_set = train_datagen.flow_from_directory(class_dir,
                                                  batch_size = 8,
                                                  class_mode = 'categorical',
                                                  subset = 'validation', seed = 10)

    model = build_model(input_shape)

    optimiser = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(training_set,epochs=10, validation_data=valid_set)
    
    model.save('modelOptAdam.h5')

    plot_history(history)


