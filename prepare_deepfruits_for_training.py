import os
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

def create_folder_tree(new_dir):
    os.mkdir(Path(new_dir))
    os.mkdir(Path(new_dir) / "images")
    os.mkdir(Path(new_dir) / "images" / "train")
    os.mkdir(Path(new_dir) / "images" / "test")
    os.mkdir(Path(new_dir) / "labels")
    os.mkdir(Path(new_dir) / "labels" / "train")
    os.mkdir(Path(new_dir) / "labels" / "test")


def convert_copy_images(original_dir, new_dir):

    datasets_dir = original_dir + "datasets/"
    fruits_folder = next(os.walk(Path(datasets_dir)))[1]
    for fruit_folder in fruits_folder:
        
        # convert and copy test images into new_directory/images/test
        images_folder_path = Path(datasets_dir) / fruit_folder / "TEST_RGB"
        for png_fruit_image in next(os.walk(images_folder_path))[2]:
            if png_fruit_image.split(".")[1] != "png":
                continue
            fruit_image = Image.open(images_folder_path / png_fruit_image)
            jpg_fruit_image = png_fruit_image.split(".")[0] + ".jpg"
            # print(jpg_fruit_image)
            # convert RGBA to RGB
            rgb_fruit_image = fruit_image.convert('RGB')
            rgb_fruit_image.save(Path(new_dir) / "images" / "test" / jpg_fruit_image)

        # do the same thing for train images
        images_folder_path = Path(datasets_dir) / fruit_folder / "TRAIN_RGB"
        for png_fruit_image in next(os.walk(images_folder_path))[2]:
            if png_fruit_image.split(".")[1] != "png":
                continue
            fruit_image = Image.open(images_folder_path / png_fruit_image)
            jpg_fruit_image = png_fruit_image.split(".")[0] + ".jpg"
            # print(jpg_fruit_image)
            # convert RGBA to RGB
            rgb_fruit_image = fruit_image.convert('RGB')
            rgb_fruit_image.save(Path(new_dir) / "images" / "train" / jpg_fruit_image)


def create_labels(original_dir, new_dir):

    datasets_dir = original_dir + "datasets/"
    fruits_folder = next(os.walk(Path(datasets_dir)))[1]
    for fruit_folder in fruits_folder:

        # file test_RGB.txt is missnamed in the original dataset for the oranges
        if fruit_folder == "orange":
            test_rgb_file = "test_RGB.txt.txt"
        else:
            test_rgb_file = "test_RGB.txt"

        # get the test labels
        with open(Path(datasets_dir) / fruit_folder / test_rgb_file, "r") as labels_file:
            for line in labels_file.read().split("\n"):
                if line == "":
                    continue
                new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                # print(new_label_file_name)
                new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,6)
                lines_to_write = []
                for line_index in range(new_label_file_lines_array.shape[0]):
                    line_to_write = " ".join([
                        new_label_file_lines_array[line_index][4],
                        new_label_file_lines_array[line_index][0],
                        new_label_file_lines_array[line_index][1],
                        new_label_file_lines_array[line_index][2],
                        new_label_file_lines_array[line_index][3],
                        "\n"
                    ])
                    lines_to_write.append(line_to_write)
                    with open(Path(new_dir) / "labels" / "test" / new_label_file_name, "w") as new_labels_file:
                        new_labels_file.writelines(lines_to_write)

        # get the train labels
        if fruit_folder == "capsicum":
            # the file train_RGB for the capsicum doesn't follow the same pattern as the others, special casing
            with open(Path(datasets_dir) / fruit_folder / "train_RGB.txt", "r") as labels_file:
                for line in labels_file.read().split("\n"):
                    if line == "":
                        continue
                    new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                    # print(new_label_file_name)
                    new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,4)
                    lines_to_write = []
                    for line_index in range(new_label_file_lines_array.shape[0]):
                        line_to_write = " ".join([
                            "2",
                            new_label_file_lines_array[line_index][0],
                            new_label_file_lines_array[line_index][1],
                            new_label_file_lines_array[line_index][2],
                            new_label_file_lines_array[line_index][3],
                            "\n"
                        ])
                        lines_to_write.append(line_to_write)
                        with open(Path(new_dir) / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
                            new_labels_file.writelines(lines_to_write)

        else:
            with open(Path(datasets_dir) / fruit_folder / "train_RGB.txt", "r") as labels_file:
                for line in labels_file.read().split("\n"):
                    if line == "":
                        continue
                    new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                    # print(new_label_file_name)
                    new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,6)
                    lines_to_write = []
                    for line_index in range(new_label_file_lines_array.shape[0]):
                        line_to_write = " ".join([
                            new_label_file_lines_array[line_index][4],
                            new_label_file_lines_array[line_index][0],
                            new_label_file_lines_array[line_index][1],
                            new_label_file_lines_array[line_index][2],
                            new_label_file_lines_array[line_index][3],
                            "\n"
                        ])
                        lines_to_write.append(line_to_write)
                        with open(Path(new_dir) / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
                            new_labels_file.writelines(lines_to_write)


def perpare_dataset(original_dir, new_dir):
    """
    Function that transforms the deepFruit dataset into input that Yolov3 can understand,
    and copy the new dataset into the new_dir directory.
    """
    
    # create the new directories
    print("Creating folder ...")
    create_folder_tree(new_dir)
    print("Done.")

    # convert all images from the original dataset into .jpg and copy them in the new dataset
    print("Converting and copying images ...")
    convert_copy_images(original_dir, new_dir)
    print("Done.")

    # create the new labels, but keep x1, y1, x2, y2 for now
    print("Creating labels ...")
    create_labels(original_dir, new_dir)
    print("Done.")

    # delete labels without images and images without labels

    # get the images height and width, and compute the new values for the label files
    # from (object_class, x1, y1, x2, y2) to (object_class, center_x, center_y, width, height)
    # normalized between 0 and 1

    # create new classes for the different fruits
    # 1: capsicum, 2: rockmelon, 3: apple, 4: avocado, 5: mango, 6: orange, 7: strawberry
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='./deepFruits_dataset/', help='original deepFruits dataset directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./deepFruits_for_training/', help='new directory name for the prepared deepFruits dataset')

    args = parser.parse_args()

    # verify input
    if args.input_dir[-1] != "/":
        args.input_dir += "/"
    if args.output_dir[-1] != "/":
        args.output_dir += "/"

    if os.path.exists(Path(args.output_dir)):
        print("DeepFruits dataset already prepared for training.")
    else:
        print("preparing dataset ...")
        perpare_dataset(args.input_dir, args.output_dir)
        print("DeepFruits dataset prepared for training.")
