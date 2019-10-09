import numpy as np

def count_percentages(image_matrix, treshold):
    percentages = [
        0, # Road
        0, # Building
        0  # Water
    ]
    total_pixels = 0

    for row in image_matrix:
        for channels in row:
            total_pixels += 1
            max_channel_value = max(channels[:3])
            if max_channel_value > treshold:
                percentages[np.argmax(channels)] += 1

    finallol = []
    for channel in percentages:
        finallol.append(channel/total_pixels)
    return finallol

def colors_to_things(image_matrix):
    for row in image_matrix:
        for channels in row:
            (r,g,b) = channels[:-1]
            if (r,g,b) == (1,0,1):
                channels[:-1] = (1,0,0) # road
            elif (r,g,b) == (1,1,0):
                channels[:-1] = (0,1,0) # building
            elif (r,g,b) == (1,0,0):
                channels[:-1] = (0,0,1) # water
            else:
                channels[:-1] = (0,0,0) # nothing

    return image_matrix

if __name__ == "__main__":
    import matplotlib.image as img
    image = img.imread('data/consid/training/Masks/all/cxb_11_31.png')

    image = colors_to_things(image)
    print(count_percentages(image, 0.1))
