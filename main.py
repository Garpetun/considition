import matplotlib.pyplot as plt

from datagenerator import DataGeneratorFolder
from augmentation import aug_with_crop

if __name__ == "__main__":
    # # Benchmark dataset
    # test_generator = DataGeneratorFolder(root_dir = 'data/road_segmentation/training',
    #                                      image_folder = 'input/',
    #                                      mask_folder = 'output/',
    #                                      batch_size = 1,
    #                                      nb_y_features = 1, augmentation = aug_with_crop)

    # Consid dataset
    test_generator = DataGeneratorFolder(root_dir = 'data/consid/Training_dataset',
                                         image_folder = 'Images/',
                                         mask_folder = 'Masks/all',
                                         batch_size = 10,
                                         nb_y_features = 1, augmentation = aug_with_crop)
    Xtest, ytest = test_generator.__getitem__(0)
    plt.imshow(Xtest[0])
    plt.show()
    plt.imshow(ytest[0, :,:,0])
    plt.show()
