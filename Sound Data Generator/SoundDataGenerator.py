import os
import glob
import threading
import multiprocessing.pool
import numpy as np

from keras_preprocessing import get_keras_submodule
from keras_preprocessing.image.utils import _list_valid_filenames_in_directory

try:
    IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
    IteratorType = object

from librosa import load
import DataAug_misc as da

class Iterator(IteratorType):

    white_list_formats = ('wav')

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        raise NotImplementedError

class BatchFromFilesMixin():
    
    def set_processing_attrs(self,
                             sound_data_generator,
                             target_size,
                             subset):
    
        self.sound_data_generator = sound_data_generator
        self.target_size = tuple(target_size)
        
        self.image_shape = self.target_size + (1,)

        if subset is not None:
            validation_split = self.sound_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)

        filepaths = self.filepaths
        noise_paths = glob.glob(os.path.join(self.sound_data_generator.noise_dir, "*.wav"))
        
        for i, j in enumerate(index_array):
            x, _ = load(filepaths[j])
            
            if self.sound_data_generator:
                if self.sound_data_generator.same_class_aug:
                    x = da.same_class_augmentation(x, os.path.dirname(filepaths[j]))
                if self.sound_data_generator.noise_aug:
                    x = da.noise_augmentation(x, noise_paths)
                
                x = self.sound_data_generator.wav_to_output(x)
            
            batch_x[i] = x
        
        if self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.    
        else:
            return batch_x
        
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    @property
    def filepaths(self):
        raise NotImplementedError
    @property
    def labels(self):
        raise NotImplementedError
    @property
    def sample_weight(self):
        raise NotImplementedError
        
class DirectoryIterator(BatchFromFilesMixin, Iterator):
   
    allowed_class_modes = {'categorical', 'binary', 'sparse', None}

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(DirectoryIterator, cls).__new__(cls)

    def __init__(self,
                 directory,
                 sound_data_generator,
                 target_size=(256, 256),
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        
        super(DirectoryIterator, self).set_processing_attrs(sound_data_generator,
                                                            target_size,
                                                            subset)
        
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype

        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        print('Found %d wav files belonging to %d classes.' %
              (self.samples, self.num_classes))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property 
    def sample_weight(self):
        return None

class SoundDataGenerator(object):
    def __init__(self,
                 output_mode='mfcc',
                 noise_dir="None",
                 validation_split=0.0,
                 time_shift=True,
                 pitch_shift=True,
                 same_class_aug=True,
                 noise_aug=True,
                 dtype='float32'):
        
        self.dtype = dtype
        self.time_shift = time_shift
        self.pitch_shift = pitch_shift
        self.output_mode = output_mode
        self.same_class_aug = same_class_aug
        self.noise_aug = noise_aug
        self.noise_dir = noise_dir
        self.data_format = "channels_last"

        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

    def flow_from_directory(self,
                            directory,
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            follow_links=False,
                            subset=None):

        target_size = da.target_size_calc(self.output_mode)[:2]

        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            follow_links=follow_links,
            subset=subset,
            dtype=self.dtype
        )

    def wav_to_output(self, x):

        if(self.output_mode == 'mfcc'):
            spec = da.wav_to_mel_util(x) 
            
            if(self.time_shift):
                spec = da.time_shift_spectrogram(spec)            
            if(self.pitch_shift):
                spec = da.pitch_shift_spectrogram(spec)
            
            spec = da.mel_to_mfcc(spec)
            return spec
        
        elif(self.output_mode == 'mel_spec'):
            spec = da.wav_to_mel(x)    
        
            if(self.time_shift):
                spec = da.time_shift_spectrogram(spec)            
            if(self.pitch_shift):
                spec = da.pitch_shift_spectrogram(spec)
        
            return spec
        
        else:
            spec = da.wav_to_power(x)
            
            if(self.time_shift):
                spec = da.time_shift_spectrogram(spec)            
            if(self.pitch_shift):
                spec = da.pitch_shift_spectrogram(spec)
        
            return spec
            

