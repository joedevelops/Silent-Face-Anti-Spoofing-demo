# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import dlib

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# Because the aspect ratio of the video stream obtained by the Android APK is 3:4, 
# in order to be consistent with it, the aspect ratio is limited to 3:4.

def check_image(image):
    height, width, channel = image.shape
    # print("Image dimensions:\nheight: "+ str(height)+"\nwidth: "+str(width))
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)
    if len(faces) > 0:
        # Return the bounding box of the first detected face
        return faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
    else:
        return None

def crop_to_desired_aspect_ratio(image, bbox):
    desired_height = int(image.shape[1] * 3 / 4)  # Calculate the desired height based on the aspect ratio
    if desired_height <= image.shape[0]:
        # Calculate the center of the detected face region
        face_center_x = (bbox[0] + bbox[2]) // 2
        face_center_y = (bbox[1] + bbox[3]) // 2
        # Calculate the new cropping region centered around the face
        crop_start_y = max(0, face_center_y - (desired_height // 2))
        crop_end_y = min(image.shape[0], face_center_y + (desired_height // 2))
        cropped_image = image[crop_start_y:crop_end_y, :]
        print("Image succesfully cropped!")
        return cropped_image
    else:
        return None

def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        # Detect face in the original image
        face_bbox = detect_face(image)
        if face_bbox is None:
            print("No face detected in the image.")
            return

        cropped_image = crop_to_desired_aspect_ratio(image, face_bbox)

        if cropped_image is None:
            print("Cropped image does not meet the aspect ratio requirement.")
            return
        # Save the cropped image with a new name
        format_ = os.path.splitext(image_name)[-1]
        cropped_image_name = image_name.replace(format_, "_cropped" + format_)
        cv2.imwrite(SAMPLE_IMAGE_PATH + cropped_image_name, cropped_image)

        # Read the saved cropped image
        saved_cropped_image = cv2.imread(SAMPLE_IMAGE_PATH + cropped_image_name)

        # Update the image_bbox to reflect the new dimensions
        image_bbox = model_test.get_bbox(saved_cropped_image)
    else:
        image_bbox = model_test.get_bbox(image)

    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        if cropped_image is not None: 
            image = saved_cropped_image
        
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))

    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
