import os
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import time


def print_erase(sentence, last_print):
    print("\r" + (" " * len(last_print)), end="\r")
    print(sentence, end="", flush=True)


def create_folder_tree(new_dir_path):
    os.mkdir(new_dir_path)
    os.mkdir(new_dir_path / "images")
    os.mkdir(new_dir_path / "images" / "train")
    os.mkdir(new_dir_path / "images" / "test")
    os.mkdir(new_dir_path / "labels")
    os.mkdir(new_dir_path / "labels" / "train")
    os.mkdir(new_dir_path / "labels" / "test")


def convert_copy_images(original_dir_path, new_dir_path, verbose):

    datasets_dir_path = original_dir_path / "datasets"
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
            rgb_fruit_image.save(new_dir_path / "images" / "test" / jpg_fruit_image)

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
            rgb_fruit_image.save(new_dir_path / "images" / "train" / jpg_fruit_image)

    # resume normal prints:
    print() if verbose else ...


def create_labels(original_dir_path, new_dir_path, verbose):

    # 1: capsicum, 2: rockmelon, 3: apple, 4: avocado, 5: mango, 6: orange, 7: strawberry
    fruits_classes = {
        'apple': "2",
        'avocado': "3",
        'capsicum': "0",
        'mango': "4",
        'orange': "5",
        'rockmelon': "1",
        'strawberry': "6"
    }

    datasets_dir_path = original_dir_path / "datasets"
    fruits_folder = next(os.walk(datasets_dir_path))[1]
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
                with open(new_dir_path / "labels" / "test" / new_label_file_name, "w") as new_labels_file:
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
                    with open(new_dir_path / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
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
                    with open(new_dir_path / "labels" / "train" / new_label_file_name, "w") as new_labels_file:
                        new_labels_file.writelines(lines_to_write)
    
    # resume normal prints
    print() if verbose else ...


def delete_useless_data(new_dir_path, verbose):

    images_test_files = [name.split(".")[0] for name in next(os.walk(new_dir_path / "images" / "test"))[2]]
    images_train_files = [name.split(".")[0] for name in next(os.walk(new_dir_path / "images" / "train"))[2]]
    labels_test_files = [name.split(".")[0] for name in next(os.walk(new_dir_path / "labels" / "test"))[2]]
    labels_train_files = [name.split(".")[0] for name in next(os.walk(new_dir_path / "labels" / "train"))[2]]

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
        os.remove(new_dir_path / "images" / "test" / image_file)
    for image_file in images_train_files_no_labels:
        image_file += ".jpg"
        os.remove(new_dir_path / "images" / "train" / image_file)
    for label_file in labels_test_files_no_images:
        label_file += ".txt"
        os.remove(new_dir_path / "labels" / "test" / label_file)
    for label_file in labels_train_files_no_images:
        label_file += ".txt"
        os.remove(new_dir_path / "labels" / "train" / label_file)


def compute_labels(new_dir_path, verbose):

    last_print = ""
    steps = ["test", "train"]
    for step in steps:
    
        file_names = [name.split(".")[0] for name in next(os.walk(new_dir_path / "images" / step))[2]]

        for file_name in file_names:

            print_erase(file_name, last_print) if verbose else ...
            last_print = file_name

            lines_to_write = []
            image_file = Image.open(new_dir_path / "images" / step / (file_name + ".jpg"))
            width_px, height_px = image_file.size

            with open(new_dir_path / "labels" / step / (file_name + ".txt"), "r") as label_file:
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
            
            with open(new_dir_path / "labels" / step / (file_name + ".txt"), "w") as label_file:
                label_file.writelines(lines_to_write)

    # resume normal print
    print() if verbose else ...


def resize_images(new_dir_path, max_size, verbose):

    last_print = ""
    steps = ["test", "train"]
    for step in steps:

        for image_name in next(os.walk(new_dir_path / "images" / step))[2]:

            image_file = Image.open(new_dir_path / "images" / step / image_name)
            
            if image_file.width > max_size or image_file.height > max_size:

                print_erase(image_name, last_print) if verbose else ...
                last_print = image_name

                if image_file.width > image_file.height:
                    new_width = max_size
                    new_height = round(max_size * image_file.height / image_file.width)
                elif image_file.width < image_file.height:
                    new_width = round(max_size * image_file.width / image_file.height)
                    new_height = max_size
                else:
                    new_width = max_size
                    new_height = max_size
                image_file.resize((new_width, new_height)).save(new_dir_path / "images" / step / image_name)

    # resume normal print
    print() if verbose else ...


def perpare_dataset(original_dir_path, new_dir_path, max_size, verbose):
    """
    Function that transforms the deepFruit dataset into input that Yolov3 can understand,
    and copy the new dataset into the new_dir directory.
    """
    
    # create the new directories
    print("Creating folder ...") if verbose else ...
    create_folder_tree(new_dir_path)
    print("Done.") if verbose else ...

    # convert all images from the original dataset into .jpg and copy them in the new dataset
    print("Converting and copying images ...") if verbose else ...
    convert_copy_images(original_dir_path, new_dir_path, verbose)
    print("Done.") if verbose else ...

    # create the new labels, but keep x1, y1, x2, y2 for now
    print("Creating labels ...") if verbose else ...
    create_labels(original_dir_path, new_dir_path, verbose)
    print("Done.") if verbose else ...

    # delete labels without images and images without labels
    print("Deleting useless data ...") if verbose else ...
    delete_useless_data(new_dir_path, verbose)
    print("Done.") if verbose else ...

    # get the images height and width, and compute the new values for the label files
    # from (object_class, x1, y1, x2, y2) to (object_class, center_x, center_y, width, height)
    # normalized between 0 and 1
    print("Recomputing labels with image size ...") if verbose else ...
    compute_labels(new_dir_path, verbose)
    print("Done.") if verbose else ...

    # resize each image to 640 pixels maximum
    print("Resizing images ...") if verbose else ...
    resize_images(new_dir_path, max_size, verbose)
    print("Done.") if verbose else ...

    # add a .yaml file


def create_yaml(yaml_path):

    yaml_content = "\n".join([
        "# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]",
        "train: ../deepFruits_for_training/images/train/",
        "val: ../deepFruits_for_training/images/test/",
        "",
        "# number of classes",
        "nc: 7",
        "",
        "# class names",
        "names: [ 'capsicum', 'rockmelon', 'apple', 'avocado', 'mango', 'orange', 'strawberry' ]"
    ])

    with open(yaml_path, "w", encoding="UTF-8") as f:
        f.write(yaml_content)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default='deepFruits_dataset', help='original deepFruits dataset directory (defaults to "deepruits_dataset")')
    parser.add_argument('--output_dir', '-o', type=str, default='./', help='directory in which to store the prepared dataset and the YAML file (defaults to current directory)')
    parser.add_argument('--max_size', type=int, default=640, help='maximum height/width of the output images in pixels (defaults to 640)')
    parser.add_argument('--verbose', '-v', action='store_true', help='print advancement')

    args = parser.parse_args()
    new_dir_path = Path(args.output_dir) / "deepFruits_for_training"
    yaml_path = Path(args.output_dir) / "deepFruits.yaml"

    if os.path.exists(new_dir_path):
        print("DeepFruits dataset already prepared for training.")
    else:
        print("Preparing dataset ...")
        perpare_dataset(Path(args.input_dir), new_dir_path, args.max_size, args.verbose)
        print("DeepFruits dataset prepared for training.")

    if os.path.exists(yaml_path):
        print("YAML file already made.")
    else:
        print("Making YAML file ...")
        create_yaml(yaml_path)
        print("YAML file ready for training.")