import matplotlib.pyplot as plt
import numpy as np

import api
import solutionHelper
from model import load_model
from datagenerator import read_image

api_key = '1c66d89b-24c5-4270-95e8-4269f9905076'
image_folder_path = 'data/from_api/'


def main():
    model = load_model('consid.efficientnetb1imgsize')
    result = api.init_game(api_key)
    game_id = result["gameId"]
    rounds_left = result['numberOfRounds']
    print("Starting a new game with id: " + game_id)
    print("The game has {} rounds and {} images per round".format(rounds_left, result["imagesPerRound"]))
    while rounds_left > 0:
        print("Starting new round, {} rounds left".format(rounds_left))
        solutions = []
        zip_bytes = api.get_images(api_key)
        image_names = solutionHelper.save_images_to_disk(zip_bytes, image_folder_path)
        for name in image_names:
            path = image_folder_path + "/" + name
            image_solution = analyze_image(path, model)
            solutions.append({"ImageName": name,
                              "BuildingPercentage": image_solution["building_percentage"],
                              "RoadPercentage": image_solution["road_percentage"],
                              "WaterPercentage": image_solution["water_percentage"]})
        solution_response = api.score_solution(api_key, {"Solutions": solutions})
        solutionHelper.print_errors(solution_response)
        solutionHelper.print_scores(solution_response)
        rounds_left = solution_response['roundsLeft']

    solutionHelper.clean_images_from_folder(image_folder_path)


def plot_mask_image(mask, img):
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].imshow(mask, cmap="Blues")
    axs[1].imshow(img)
    plt.show()


def analyze_image(image_path, model):
    model_input = read_image(image_path).astype(np.float32)[None, :, :, :]
    predicted_mask = (model.predict(model_input)[0] > [0.3, 0.2, 0.5]).astype(np.float32)
    percentages = percentages_from_mask(predicted_mask)
    # return model[image_path] # TODO: This is out of wack
    result = {"building_percentage": percentages[1],
              "water_percentage": percentages[2],
              "road_percentage": percentages[0]}
    # print(result)
    # plot_mask_image(predicted_mask, model_input[0])
    return result


def percentages_from_mask(mask):
    percentages = []
    total_pixels = mask.shape[0] * mask.shape[1]
    for i in range(3):
        count_of_type = (mask[:, :, i]).sum()
        percentages.append(count_of_type / total_pixels * 100)
    return percentages


def read_annotated_mask(path):
    return np.zeros((1024, 1024, 3))


if __name__ == '__main__':
    main()
