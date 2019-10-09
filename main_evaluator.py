import api
import solutionHelper

api_key = '1c66d89b-24c5-4270-95e8-4269f9905076'
image_folder_path = 'data/from_api/'


def main():
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
            image_solution = analyze_image(path)
            solutions.append({"ImageName": name,
                              "BuildingPercentage": image_solution["building_percentage"],
                              "RoadPercentage": image_solution["road_percentage"],
                              "WaterPercentage": image_solution["water_percentage"]})
        solution_response = api.score_solution(api_key, {"Solutions": solutions})
        solutionHelper.print_errors(solution_response)
        solutionHelper.print_scores(solution_response)
        rounds_left = solution_response['roundsLeft']

    solutionHelper.clean_images_from_folder(image_folder_path)


def analyze_image(image_path):
    # return model[image_path] # TODO: This is out of wack
    return {"building_percentage": 13.37, "water_percentage": 13.37, "road_percentage": 13.37}


main()
