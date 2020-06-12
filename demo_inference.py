#!/usr/bin/env python
# coding: utf-8

# ## 0. Setup
# First, we import the libraries required to load images and setup the SSD head detector for inference. This model has been trained using [Pierluigi Ferrari's SSD implementation](https://github.com/pierluigiferrari/ssd_keras/), of which we include the required code in this repository. We refer you to the original implementation in case of an issue with that code.

# In[1]:


from platform import python_version

# import subprocess
import os

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

import cv2
from PIL import Image

# get_ipython().run_line_magic('matplotlib', 'inline')


# Since there are differences in the object serialization methods used between Python versions previous to version 3.6, we provide two different versions of our model. In order to determine which version should be used, please run the following code:

# In[2]:


py_version = "".join(python_version().split("."))
model_version = "3.6"
if int(py_version) < 360:
    model_version = "3.5"
print("Using model version for Python %s." % model_version)


# This model uses a 512x512 input size, let's set our image size to that resolution size:

# In[3]:


# Set the image size.
img_height = 512
img_width = 512


# There are two different inference modes: the mode `inference` uses the original implementation's algorithm, and the `inference_fast` mode is faster but might produce different results.

# In[4]:


# Set the model's inference mode
model_mode = "inference"


# Finally, we set the model's confidence threshold value, i.e. the model will only output predictions above or equal to this value. Set a value in range [0, 1], where a higher value decreases the number of detections produced by the model and increases its speed during inference.
#
# We choose to set a low value (1%) for the model confidence threshold and then use a higher value at a later stage.

# In[5]:


# Set the desired confidence threshold
conf_thresh = 0.01


# ## 1. Load a trained SSD model
# In order to run the head detection model first you must download the `.h5` file that contains the model and the trained weights. To do so, run the code in [section 1.1](#download_cell).
#
# Once downloaded, you can either build a new model and load the trained weights into it ([section 1.2](#build_cell)) or just load the ready to use model ([section 1.3](#load_cell)). The first option allows you to configure some model parameters, while the latter loads the original model's configuration.

# <a id='download_cell'></a>
# ### 1.1 Download the model's .h5 file
# First, we download the model's file from the server to this repository's `data/models` directory. We provide a simple script which will do this task for you, which you can execute by running the next cell. NOTE: this script requires `curl`to be installed in your system.
#
# In case you prefer to download the model manually, you can download it using the following links:
#  - [Python versions <= 3.5](https://drive.google.com/open?id=12cqKTPtQBAu780219hEbST7VwQuf6xDH).
#  - [Python versions >= 3.6](https://drive.google.com/open?id=1vlmKOBtaT7eAd4_WcAv5MLBn7q_SWXoh).

# In[6]:


# Determine which script to use
script_path = "download_model_py%s.sh" % model_version
# Download the model file into the "data" directory
os.chdir("./data/")
if script_path in os.listdir(os.getcwd()):
    print("Downloading model...")
    #     subprocess.call('./' + script_path)
    print("Download finished.")
# Restore the working directory path
os.chdir("..")


# <a id='build_cell'></a>
# ### 1.2 Build the model and load the trained weights into it
# You can build a new model and load the provided weights into it, which allows some degree of personalization.

# In[7]:


# 1: Build the Keras model
K.clear_session()  # Clear previous models from memory.
model = ssd_512(
    image_size=(img_height, img_width, 3),
    n_classes=1,
    mode=model_mode,
    l2_regularization=0.0005,
    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],  # PASCAL VOC
    aspect_ratios_per_layer=[
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5],
    ],
    two_boxes_for_ar1=True,
    steps=[8, 16, 32, 64, 128, 256, 512],
    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    clip_boxes=False,
    variances=[0.1, 0.1, 0.2, 0.2],
    normalize_coords=True,
    subtract_mean=[123, 117, 104],
    swap_channels=[2, 1, 0],
    confidence_thresh=conf_thresh,
    iou_threshold=0.45,
    top_k=200,
    nms_max_output_size=400,
)

# 2: Load the trained weights into the model. Make sure the path correctly points to the model's .h5 file
weights_path = (
    "./data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py%s.h5"
    % model_version
)
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 4: OPTIONAL Save compiled model in inference mode so we can use option 1.3 in future executions
# model.save('./data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py%s.h5' % model_version)


# Or

# <a id='load_cell'></a>
# ### 1.3 Load the model as-is
# This option loads the downloaded model as it was compiled.

# In[8]:


# Make sure the path correctly points to the model's .h5 file
weights_path = (
    "./data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py%s.h5"
    % model_version
)

# Create an SSDLoss object in order to pass that to the model loader
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

# Clear previous models from memory.
K.clear_session()

# Configure the decode detections layer based on the model mode
if model_mode == "inference":
    decode_layer = DecodeDetections(
        img_height=img_height,
        img_width=img_width,
        confidence_thresh=conf_thresh,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
    )
if model_mode == "inference_fast":
    decode_layer = DecodeDetectionsFast(
        img_height=img_height,
        img_width=img_width,
        confidence_thresh=conf_thresh,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
    )

# Finally load the model
model = load_model(
    weights_path,
    custom_objects={
        "AnchorBoxes": AnchorBoxes,
        "L2Normalization": L2Normalization,
        "DecodeDetections": decode_layer,
        "compute_loss": ssd_loss.compute_loss,
    },
)


# ## 2. Load example images
#
# Load some images for which you'd like the model to make predictions. There are some demo images available in this repository's `examples`directory.

# In[9]:


def cv2pil(image):
    """ OpenCV型 -> PIL型 """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


fig = plt.figure(figsize=(20, 12))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise "camera not open!"
    _ret, frame = cap.read()
    cap.release()

    frame = cv2pil(frame)

    # Original images array
    orig_images = []
    # Resized images array
    input_images = []

    # We'll only load one image in this example.
    img_paths = []
    # '/home/shogo/Pictures/ウェブカム/2020-05-28-161738.jpg'
    from pathlib import Path

    # base_path = Path("/home/shogo/Nextcloud/sail/tanaka/me")
    base_path = Path("/home/shogo/Pictures/ウェブカム")
    for path in base_path.iterdir():
        if path.is_file():
            img_paths.append(str(path))
    # img_path = 'examples/fish_bike.jpg'
    # img_path = 'examples/rugby_players.jpg'

    # Load the original image (used to display results)
    # for imp in img_paths:
    #     orig_images.append(image.load_img(imp))
    # # Load the image resized to the model's input size
    # for imp in img_paths:
    #     img = image.load_img(imp, target_size=(img_height, img_width))
    #     img = image.img_to_array(img)
    #     input_images.append(img)

    orig_images.append(frame)
    img = frame
    img = img.resize((img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)

    input_images = np.array(input_images)

    # ## 3. Make predictions
    # Now we feed the example images into the network to obtain the predictions.

    # In[10]:

    y_pred = model.predict(input_images)
    y_pred

    # `y_pred` contains a fixed number of predictions per batch item (200 if you use the original model configuration), many of which are low-confidence predictions or dummy entries. We therefore need to apply a confidence threshold to filter out the bad predictions. Set this confidence threshold value how you see fit.

    # In[19]:

    confidence_threshold = 0.25

    # Perform confidence thresholding.
    y_pred_thresh = [
        y_pred[k][y_pred[k, :, 1] > confidence_threshold]
        for k in range(y_pred.shape[0])
    ]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print("   class   conf xmin   ymin   xmax   ymax")
    print(y_pred_thresh[0])

    # ## 4. Visualize the predictions
    #
    # We'd like to visualize the predictions on the image in its original size, so we transform the coordinates of the predicted boxes accordingly.

    # In[34]:

    N = 0

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ["background", "head"]

    # Configure plot and disable axis

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(orig_images[N])
    current_axis = plt.gca()

    # Display the image and draw the predicted boxes onto it.
    for box in y_pred_thresh[N]:
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = box[2] * np.array(orig_images[0]).shape[1] / img_width
        ymin = box[3] * np.array(orig_images[0]).shape[0] / img_height
        xmax = box[4] * np.array(orig_images[0]).shape[1] / img_width
        ymax = box[5] * np.array(orig_images[0]).shape[0] / img_height
        color = colors[int(box[0])]
        label = "{}: {:.2f}".format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                color=color,
                fill=False,
                linewidth=2,
            )
        )
        current_axis.text(
            xmin,
            ymin,
            label,
            size="x-large",
            color="white",
            bbox={"facecolor": color, "alpha": 1.0},
        )

    # plt.plot()
    plt.pause(0.01)


while True:
    main()
