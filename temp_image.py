from pathlib import Path

from cv2 import imread, imshow, waitKey, destroyAllWindows, imwrite


if __name__ == "__main__":
    
    for img_file in Path("drops").iterdir():

        img_size = 450
        x_start = 360
        z_start = 310

        if "EtOH" not in img_file.name:
            
            img = imread(str(img_file))

            img = img[z_start:z_start + img_size, x_start:x_start + img_size]

            imshow("dev", img)
            waitKey(0)
            destroyAllWindows()

            imwrite(str(Path("temp_drops") / img_file.name), img)
