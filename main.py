import os

import matplotlib.pyplot as plt

from datagenerator import DataGeneratorFolder
from augmentation import aug_with_crop
from callback import get_callbacks

DATASET_DIR = os.path.join('.', 'data', 'consid')
IMAGE_FOLDER = 'Images'
MASKS_FOLDER = os.path.join('Masks', 'all')

if __name__ == "__main__":
    # # Benchmark dataset
    # test_generator = DataGeneratorFolder(root_dir = 'data/consid/',
    #                                      image_folder = 'Images/',
    #                                      mask_folder = 'Masks/',
    #                                      batch_size = 1,
    #                                      nb_y_features = 1, augmentation = aug_with_crop)

    # Consid dataset
    test_generator = DataGeneratorFolder(root_dir = DATASET_DIR,
                                         image_folder = IMAGE_FOLDER,
                                         mask_folder = MASKS_FOLDER,
                                         batch_size = 10,
                                         nb_y_features = 1, augmentation = aug_with_crop)
    Xtest, ytest = test_generator.__getitem__(0)
    plt.imshow(Xtest[0])
    plt.show()
    plt.imshow(ytest[0, :,:,0])
    plt.show()


    train_generator = DataGeneratorFolder(root_dir = DATASET_DIR, 
                                        image_folder = IMAGE_FOLDER, 
                                        mask_folder = MASKS_FOLDER, 
                                        augmentation = aug_with_crop,
                                        batch_size=4,
                                        image_size=512,
                                        nb_y_features = 1)

    callbacks = get_callbacks()
    
