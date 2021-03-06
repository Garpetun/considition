from keras.utils import Sequence
from skimage.io import imread
import os
import numpy as np
from sklearn.utils import shuffle
from albumentations import Resize

from augmentation import aug_with_crop

DATASET_DIR = os.path.join('.', 'data', 'consid')
IMAGE_FOLDER = 'Images'
MASKS_FOLDER = os.path.join('Masks', 'all')


def read_image(path):
    assumed_image = np.zeros((1024, 1024, 3), dtype=np.float32)
    actual_image = imread(path) / 255
    assumed_image[:actual_image.shape[0], :actual_image.shape[1], :] = actual_image
    return assumed_image


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir=r'data/val_test', image_folder='img/', mask_folder='masks/',
                 batch_size=1, image_size=768, nb_y_features=1,
                 augmentation=None,
                 suffle=True):
        self.image_filenames = listdir_fullpath(os.path.join(root_dir, image_folder))
        self.mask_names = listdir_fullpath(os.path.join(root_dir, mask_folder))
        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.suffle = suffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.suffle == True:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name):
        image = read_image(image_name)
        mask = np.zeros(image.shape, dtype=np.int8)
        for i, layer in enumerate(['road', 'building', 'water']):
            name = mask_name.replace('all', layer)
            if os.path.exists(name):
                try:
                    submask = (imread(name, as_gray=True) > 0).astype(np.int8)
                    if submask.shape[:2] == mask.shape[:2]:
                        mask[:, :, i] = submask
                    else:
                        print(name, image.shape, submask.shape)
                except:
                    print("FAAAAIL", name, mask_name)
        return image, mask

    def __getitem__(self, index):
        """
        Generate one batch of data

        """
        # Generate indexes of the batch
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        # Defining dataset
        X = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):

            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            # if augmentation is defined, we assume its a train set
            if self.augmentation is not None:

                # Augmentation code
                augmented = self.augmentation(self.image_size)(image=X_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
                X[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm

            # if augmentation isnt defined, we assume its a test set.
            # Because test images can have different sizes we resize it to be divisable by 32
            elif self.augmentation is None and self.batch_size == 1:
                X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i],
                                                          self.mask_names[index * 1 + i])
                augmented = Resize(height=(X_sample.shape[0] // 32) * 32, width=(X_sample.shape[1] // 32) * 32)(
                    image=X_sample, mask=y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32), \
                       y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

        return X, y


def get_train_generator(image_size=512):
    return DataGeneratorFolder(root_dir=os.path.join(DATASET_DIR, 'full'),
                               image_folder=IMAGE_FOLDER,
                               mask_folder=MASKS_FOLDER,
                               augmentation=aug_with_crop,
                               batch_size=4,
                               image_size=image_size,
                               nb_y_features=3)


def get_test_generator(image_size=1024, augment=False):
    augmentation = aug_with_crop if augment else None
    return DataGeneratorFolder(root_dir=os.path.join(DATASET_DIR, 'testing'),
                               image_folder=IMAGE_FOLDER,
                               mask_folder=MASKS_FOLDER,
                               batch_size=1,
                               nb_y_features=3,
                               image_size=image_size,
                               augmentation=augmentation)
