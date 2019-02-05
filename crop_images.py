
import os
import cv2
import dlib
from matplotlib import pyplot as plt
import numpy as np
import config

detector = dlib.get_frontal_face_detector()


def crop_faces(plot_images=False, max_images_to_plot=5):
    bad_crop_count = 0
    if not os.path.exists(config.CROPPED_IMGS_DIR):
        os.makedirs(config.CROPPED_IMGS_DIR)
    print 'Cropping faces and saving to %s' % config.CROPPED_IMGS_DIR
    good_cropped_images = []
    good_cropped_img_file_names = []
    detected_cropped_images = []
    original_images_detected = []
    for file_name in sorted(os.listdir(config.ORIGINAL_IMGS_DIR)):
        np_img = cv2.imread(os.path.join(config.ORIGINAL_IMGS_DIR,file_name))
        detected = detector(np_img, 1)
        img_h, img_w, _ = np.shape(np_img)
        original_images_detected.append(np_img)

        if len(detected) != 1:
            bad_crop_count += 1
            continue

        d = detected[0]
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = int(x1 - config.MARGIN * w)
        yw1 = int(y1 - config.MARGIN * h)
        xw2 = int(x2 + config.MARGIN * w)
        yw2 = int(y2 + config.MARGIN * h)
        cropped_img = crop_image(np_img, xw1, yw1, xw2, yw2)
        norm_file_path = '%s/%s' % (config.CROPPED_IMGS_DIR, file_name)
        cv2.imwrite(norm_file_path, cropped_img)

        good_cropped_img_file_names.append(file_name)

    # save info of good cropped images
    with open(config.ORIGINAL_IMGS_INFO_FILE, 'r') as f:
        column_headers = f.read().splitlines()[0]
        all_imgs_info = f.read().splitlines()[1:]
    cropped_imgs_info = [l for l in all_imgs_info if l.split(',')[-1] in good_cropped_img_file_names]

    with open(config.CROPPED_IMGS_INFO_FILE, 'w') as f:
        f.write('%s\n' % column_headers)
        for l in cropped_imgs_info:
            f.write('%s\n' % l)

    print 'Cropped %d images and saved in %s - info in %s' % (len(original_images_detected), config.CROPPED_IMGS_DIR, config.CROPPED_IMGS_INFO_FILE)
    print 'Error detecting face in %d images - info in Data/unnormalized.txt' % bad_crop_count

    if plot_images:
        print 'Plotting images...'
        img_index = 0
        plot_index = 1
        plot_n_cols = 3
        plot_n_rows = len(original_images_detected) if len(original_images_detected) < max_images_to_plot else max_images_to_plot
        for row in range(plot_n_rows):
            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(original_images_detected[img_index].astype('uint8'))
            plot_index += 1

            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(detected_cropped_images[img_index])
            plot_index += 1

            plt.subplot(plot_n_rows,plot_n_cols,plot_index)
            plt.imshow(good_cropped_images[img_index])
            plot_index += 1

            img_index += 1
    plt.show()
    return good_cropped_images



# image cropping method taken from:
# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

if __name__ == '__main__':
    crop_faces()
