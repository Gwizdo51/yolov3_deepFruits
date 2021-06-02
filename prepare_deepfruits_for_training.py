import os
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import time


def print_erase(sentence, last_print):
    print("\r" + (" " * len(last_print)), end="\r")
    print(sentence, end="", flush=True)


def create_folder_tree(new_dir):
    os.mkdir(Path(new_dir))
    os.mkdir(Path(new_dir) / "images")
    os.mkdir(Path(new_dir) / "images" / "train")
    os.mkdir(Path(new_dir) / "images" / "test")
    os.mkdir(Path(new_dir) / "labels")
    os.mkdir(Path(new_dir) / "labels" / "train")
    os.mkdir(Path(new_dir) / "labels" / "test")


def convert_copy_images(original_dir, new_dir, verbose):

    datasets_dir_path = Path(original_dir) / "datasets"
    fruits_folder = next(os.walk(datasets_dir_path))[1]
    last_print = ""
    for fruit_folder in fruits_folder:
        
        # convert and copy test images into new_directory/images/test
        images_folder_path = datasets_dir_path / fruit_folder / "TEST_RGB"
        for png_fruit_image in next(os.walk(images_folder_path))[2]:
            if png_fruit_image.split(".")[1] != "png":
                continue
            fruit_image = Image.open(images_folder_path / png_fruit_image)
            jpg_fruit_image = png_fruit_image.split(".")[0] + ".jpg"
            # print(jpg_fruit_image)
            print_erase(jpg_fruit_image, last_print) if verbose else ...
            last_print = jpg_fruit_image
            # convert RGBA to RGB
            rgb_fruit_image = fruit_image.convert('RGB')
            rgb_fruit_image.save(Path(new_dir) / "images" / "test" / jpg_fruit_image)

        # do the same thing for train images
        images_folder_path = datasets_dir_path / fruit_folder / "TRAIN_RGB"
        for png_fruit_image in next(os.walk(images_folder_path))[2]:
            if png_fruit_image.split(".")[1] != "png":
                continue
            fruit_image = Image.open(images_folder_path / png_fruit_image)
            jpg_fruit_image = png_fruit_image.split(".")[0] + ".jpg"
            # print(jpg_fruit_image)
            print_erase(jpg_fruit_image, last_print) if verbose else ...
            last_print = jpg_fruit_image
            # convert RGBA to RGB
            rgb_fruit_image = fruit_image.convert('RGB')
            rgb_fruit_image.save(Path(new_dir) / "images" / "train" / jpg_fruit_image)

    # resume normal prints:
    print() if verbose else ...


def create_labels(original_dir, new_dir, verbose):

    # 1: capsicum, 2: rockmelon, 3: apple, 4: avocado, 5: mango, 6: orange, 7: strawberry
    fruits_classes = {
        'apple': "3",
        'avocado': "4",
        'capsicum': "1",
        'mango': "5",
        'orange': "6",
        'rockmelon': "2",
        'strawberry': "7"
    }

    datasets_dir_path = Path(original_dir) / "datasets"
    fruits_folder = next(os.walk(Path(datasets_dir_path)))[1]
    last_print = ""
    for fruit_folder in fruits_folder:

        # file test_RGB.txt is missnamed in the original dataset for the oranges
        if fruit_folder == "orange":
            test_rgb_file = "test_RGB.txt.txt"
        else:
            test_rgb_file = "test_RGB.txt"

        # get the test labels
        with open(datasets_dir_path / fruit_folder / test_rgb_file, "r") as labels_file:
            for line in labels_file.read().split("\n"):
                if line == "":
                    continue
                new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                # print(new_label_file_name)
                print_erase(new_label_file_name, last_print) if verbose else ...
                last_print = new_label_file_name
                new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,6)
                lines_to_write = []
                for line_index in range(new_label_file_lines_array.shape[0]):
                    line_to_write = " ".join([
                        fruits_classes[fruit_folder],
                        new_label_file_lines_array[line_index][0],
                        new_label_file_lines_array[line_index][1],
                        new_label_file_lines_array[line_index][2],
                        new_label_file_lines_array[line_index][3]
                    ]) + "\n"
                    lines_to_write.append(line_to_write)
                with open(Path(new_dir) / "labels" / "test" / new_label_file_name, "w") as new_labels_file:
                    new_labels_file.writelines(lines_to_write)

        # get the train labels
        if fruit_folder == "capsicum":
            # the file train_RGB for the capsicum doesn't follow the same pattern as the others, special casing
            with open(datasets_dir_path / fruit_folder / "train_RGB.txt", "r") as labels_file:
                for line in labels_file.read().split("\n"):
                    if line == "":
                        continue
                    new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                    # print(new_label_file_name)
                    print_erase(new_label_file_name, last_print) if verbose else ...
                    last_print = new_label_file_name
                    new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,4)
                    lines_to_write = []
                    for line_index in range(new_label_file_lines_array.shape[0]):
                        line_to_write = " ".join([
                            fruits_classes[fruit_folder],
                            new_label_file_lines_array[line_index][0],
                            new_label_file_lines_array[line_index][1],
                            new_label_file_lines_array[line_index][2],
                            new_label_file_lines_array[line_index][3]
                        ]) + "\n"
                        lines_to_write.append(line_to_write)
                    with open(Path(new_dir) / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
                        new_labels_file.writelines(lines_to_write)

        else:
            with open(datasets_dir_path / fruit_folder / "train_RGB.txt", "r") as labels_file:
                for line in labels_file.read().split("\n"):
                    if line == "":
                        continue
                    new_label_file_name = line.split()[0].split("/")[1].split(".")[0] + ".txt"
                    # print(new_label_file_name)
                    print_erase(new_label_file_name, last_print) if verbose else ...
                    last_print = new_label_file_name
                    new_label_file_lines_array = np.array(line.split()[2:]).reshape(-1,6)
                    lines_to_write = []
                    for line_index in range(new_label_file_lines_array.shape[0]):
                        line_to_write = " ".join([
                            fruits_classes[fruit_folder],
                            new_label_file_lines_array[line_index][0],
                            new_label_file_lines_array[line_index][1],
                            new_label_file_lines_array[line_index][2],
                            new_label_file_lines_array[line_index][3]
                        ]) + "\n"
                        lines_to_write.append(line_to_write)
                    with open(Path(new_dir) / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
                        new_labels_file.writelines(lines_to_write)
    
    # resume normal prints
    print() if verbose else ...


def delete_useless_data(new_dir, verbose):

    images_test_files = [name.split(".")[0] for name in next(os.walk(Path(new_dir) / "images" / "test"))[2]]
    images_train_files = [name.split(".")[0] for name in next(os.walk(Path(new_dir) / "images" / "train"))[2]]
    labels_test_files = [name.split(".")[0] for name in next(os.walk(Path(new_dir) / "labels" / "test"))[2]]
    labels_train_files = [name.split(".")[0] for name in next(os.walk(Path(new_dir) / "labels" / "train"))[2]]

    images_test_files_no_labels = [image_file for image_file in images_test_files if image_file not in labels_test_files]
    images_train_files_no_labels = [image_file for image_file in images_train_files if image_file not in labels_train_files]
    labels_test_files_no_images = [label_file for label_file in labels_test_files if label_file not in images_test_files]
    labels_train_files_no_images = [label_file for label_file in labels_train_files if label_file not in images_train_files]

    if verbose:
        print("Test images without labels :", images_test_files_no_labels)
        print("Train images without labels :", images_train_files_no_labels)
        print("Test labels without images :", labels_test_files_no_images)
        print("Train labels without images :", labels_train_files_no_images)

    for image_file in images_test_files_no_labels:
        image_file += ".jpg"
        os.remove(Path(new_dir) / "images" / "test" / image_file)
    for image_file in images_train_files_no_labels:
        image_file += ".jpg"
        os.remove(Path(new_dir) / "images" / "train" / image_file)
    for label_file in labels_test_files_no_images:
        label_file += ".txt"
        os.remove(Path(new_dir) / "labels" / "test" / label_file)
    for label_file in labels_train_files_no_images:
        label_file += ".txt"
        os.remove(Path(new_dir) / "labels" / "train" / label_file)


def compute_labels(new_dir, verbose):

    last_print = ""
    steps = ["test", "train"]
    for step in steps:
    
        test_file_names = [name.split(".")[0] for name in next(os.walk(Path(new_dir) / "images" / step))[2]]

        for file_name in test_file_names:

            print_erase(file_name, last_print) if verbose else ...
            last_print = file_name

            lines_to_write = []
            image_file = Image.open(Path(new_dir) / "images" / step / (file_name + ".jpg"))
            width_px, height_px = image_file.size

            with open(Path(new_dir) / "labels" / step / (file_name + ".txt"), "r") as label_file:
                for line in label_file.read().split("\n")[:-1]:

                    [fruit_class, x1, y1, x2, y2] = [int(item) for item in line.split()]
                    center_x = round((x1 + x2) / (2 * width_px), 6)
                    center_y = round((y1 + y2) / (2 * height_px), 6)
                    width = round((x2 - x1) / width_px, 6)
                    height = round((y2 - y1) / height_px, 6)

                    line_to_write = " ".join([
                        str(fruit_class),
                        str(center_x),
                        str(center_y),
                        str(width),
                        str(height)
                    ]) + "\n"
                    lines_to_write.append(line_to_write)
            
            with open(Path(new_dir) / "labels" / step / (file_name + ".txt"), "w") as label_file:
                label_file.writelines(lines_to_write)

    # resume normal print
    print() if verbose else ...


def perpare_dataset(original_dir, new_dir, verbose):
    """
    Function that transforms the deepFruit dataset into input that Yolov3 can understand,
    and copy the new dataset into the new_dir directory.
    """
    
    # create the new directories
    print("Creating folder ...") if verbose else ...
    create_folder_tree(new_dir)
    print("Done.") if verbose else ...

    # convert all images from the original dataset into .jpg and copy them in the new dataset
    print("Converting and copying images ...") if verbose else ...
    convert_copy_images(original_dir, new_dir, verbose)
    print("Done.") if verbose else ...

    # create the new labels, but keep x1, y1, x2, y2 for now
    print("Creating labels ...") if verbose else ...
    create_labels(original_dir, new_dir, verbose)
    print("Done.") if verbose else ...

    # delete labels without images and images without labels
    print("Deleting useless data ...") if verbose else ...
    delete_useless_data(new_dir, verbose)
    print("Done.") if verbose else ...

    # get the images height and width, and compute the new values for the label files
    # from (object_class, x1, y1, x2, y2) to (object_class, center_x, center_y, width, height)
    # normalized between 0 and 1
    print("Recomputing labels with image size ...") if verbose else ...
    compute_labels(new_dir, verbose)
    print("Done.") if verbose else ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='deepFruits_dataset', help='original deepFruits dataset directory (defaults to "deepruits_dataset")')
    parser.add_argument('--output_dir', '-o', type=str, default='deepFruits_for_training', help='new directory name for the prepared deepFruits dataset (defaults to "deepFruits_for_training")')
    parser.add_argument('--verbose', '-v', action='store_true', help='print advancement')

    args = parser.parse_args()

    if os.path.exists(Path(args.output_dir)):
        print("DeepFruits dataset already prepared for training.")
    else:
        print("preparing dataset ...")
        perpare_dataset(args.input_dir, args.output_dir, args.verbose)
        print("DeepFruits dataset prepared for training.")