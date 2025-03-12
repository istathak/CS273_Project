from PIL import Image
import os

def rotate_images(filepath):

    for filename in os.listdir(filepath):
        img = Image.open(os.path.join(filepath, filename))
        if "mask" in filename:
            continue
        else:
            ben_mal = "benign" if "benign" in filename else "malignant"

            for angle in [0, 90, 180, 270]:
                new_name = filename[:-4] + "_rot_" + str(angle) + ".png"
                img.rotate(angle, expand = True).save(os.path.join("rotated_set", ben_mal, new_name))

#rotate_images("training_set/malignant")

