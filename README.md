
### 1. Install dependencies

Install the Python requirements for this code pattern. Run:
```
pip install -r requirements.txt
```

> **Note**: If you have an Nvidia GPU and want to use it in training, then you will need to
install `tensorflow-gpu` instead of `tensorflow`. Details for installation can be found
[here](https://www.tensorflow.org/install/gpu).

> **TIP** :bulb: To terminate the virtual environment use the `deactivate` command.

### 2. Generate Image Data

In order to train a decent model, having copious amounts of data is necessary.
However, getting a large enough dataset of actual handwritten Korean characters
is challenging to find and cumbersome to create.

One way to deal with this data issue is to programmatically generate the data
yourself, taking advantage of the abundance of Korean font files found online.
So, that is exactly what we will be doing.

Provided in the tools directory of this repo is
[hangul-image-generator.py](./tools/hangul-image-generator.py).
This script will use fonts found in the fonts directory to create several images
for each character provided
in the given labels file. The default labels file is
[2350-common-hangul.txt](./labels/2350-common-hangul.txt)
which contains 2350 frequent characters derived from the
[KS X 1001 encoding](https://en.wikipedia.org/wiki/KS_X_1001). Other label files
are [256-common-hangul.txt](./labels/256-common-hangul.txt) and
[512-common-hangul.txt](./labels/512-common-hangul.txt). These were adapted from
the top 6000 Korean words compiled by the National Institute of Korean Language
listed [here](https://www.topikguide.com/download/6000_korean_words.htm).
If you don't have a powerful machine to train on, using a smaller label set can
help reduce the amount of model training time later on.

The [fonts](./fonts) folder is currently empty, so before you can generate the
Hangul dataset, you must first download
several font files as described in the fonts directory [README](./fonts/README.md).
For my dataset, I used around 40 different font files, but more can always be
used to improve your dataset, especially if you get several uniquely stylized ones.
Once your fonts directory is populated, then you can proceed with the actual
image generation with [hangul-image-generator.py](./tools/hangul-image-generator.py).

Optional flags for this are:

* `--label-file` for specifying a different label file (perhaps with less characters).
  Default is _./labels/2350-common-hangul.txt_.
* `--font-dir` for specifying a different fonts directory. Default is _./fonts_.
* `--output-dir` for specifying the output directory to store generated images.
  Default is _./image-data_.

Now run it, specifying your chosen label file:

```
python ./tools/hangul-image-generator.py --label-file <your label file path>
```

Depending on how many labels and fonts there are, this script may take a while
to complete. In order to bolster the dataset, three random elastic distortions are
also performed on each generated character image. An example is shown below, with
the original character displayed first, followed by the elastic distortions.

![Normal Image](doc/source/images/hangul_normal.jpeg "Normal font character image")
![Distorted Image 1](doc/source/images/hangul_distorted1.jpeg "Distorted font character image")
![Distorted Image 2](doc/source/images/hangul_distorted2.jpeg "Distorted font character image")
![Distorted Image 3](doc/source/images/hangul_distorted3.jpeg "Distorted font character image")

Once the script is done, the output directory will contain a _hangul-images_ folder
which will hold all the 64x64 JPEG images. The output directory will also contain a
_labels-map.csv_ file which will map all the image paths to their corresponding
labels.

### 3. Convert Images to TFRecords

The TensorFlow standard input format is
[TFRecords](https://github.com/tensorflow/docs/tree/master/site/en/api_guides/python#tfrecords_format_details),
which is a binary format that we can use to store raw image data and their labels
in one place. In order to better feed in data to a TensorFlow model, let's first create
several TFRecords files from our images. A [script](./tools/convert-to-tfrecords.py)
is provided that will do this for us.

This script will first read in all the image and label data based on the
_labels-map.csv_ file that was generated above. Then it will partition the data
so that we have a training set and also a testing set (15% testing, 85% training).
By default, the training set will be saved into multiple files/shards
(three) so as not to end up with one gigantic file, but this can be configured
with a CLI argument, _--num-shards-train_, depending on your data set size.

Optional flags for this script are:

* `--image-label-csv` for specifying the CSV file that maps image paths to labels.
  Default is _./image-data/labels-map.csv_
* `--label-file` for specifying the labels that correspond to your training set.
  This is used by the script to determine the number of classes.
  Default is _./labels/2350-common-hangul.txt_.
* `--output-dir` for specifying the output directory to store TFRecords files.
  Default is _./tfrecords-output_.
* `--num-shards-train` for specifying the number of shards to divide training set
  TFRecords into. Default is _3_.
* `--num-shards-test` for specifying the number of shards to divide testing set
  TFRecords into. Default is _1_.

To run the script, you can simply do:

```
python ./tools/convert-to-tfrecords.py --label-file <your label file path>
```

Once this script has completed, you should have sharded TFRecords files in the
output directory _./tfrecords-output_.

```
$ ls ./tfrecords-output
test1.tfrecords    train1.tfrecords    train2.tfrecords    train3.tfrecords
```

### 4. Train the Model

Now that we have a lot of data, it is time to actually use it. In the root of
the project is [hangul_model.py](./hangul_model.py). This script will handle
creating an input pipeline for reading in TFRecords files and producing random
batches of images and labels. Next, a convolutional neural network (CNN) is
defined, and training is performed. The training process will continuously feed
in batches of images and labels to the CNN to find the optimal weight and biases
for correctly classifying each character. After training, the model is exported
so that it can be used in our Android application.

The model here is similar to the MNIST model described on the TensorFlow
[website](https://www.tensorflow.org/tutorials/). A third
convolutional layer is added to extract more features to help classify for the
much greater number of classes.

Optional flags for this script are:

* `--label-file` for specifying the labels that correspond to your training set.
  This is used by the script to determine the number of classes to classify for.
  Default is _./labels/2350-common-hangul.txt_.
* `--tfrecords-dir` for specifying the directory containing the TFRecords shards.
  Default is _./tfrecords-output_.
* `--output-dir` for specifying the output directory to store model checkpoints,
   graphs, and Protocol Buffer files. Default is _./saved-model_.
* `--num-train-epochs` for specifying the number of epochs to train for.
  This is the number of complete passes through the training dataset.
  Definitely try tuning this parameter to improve model performance on your dataset.
  Default is 15 epochs.

To run the training, simply do the following from the root of the project:

```
python ./hangul_model.py --label-file <your label file path> --num-train-epochs <num>
```

Depending on how many images you have, this will likely take a long time to
train (several hours to maybe even a day), especially if only training on a laptop.
If you have access to GPUs, these will definitely help speed things up, and you
should certainly install the TensorFlow version with GPU support (supported on
[Ubuntu](https://www.tensorflow.org/install/) and
[Windows](https://www.tensorflow.org/install/) only).

One alternative is to use a reduced label set (i.e. 256 vs 2350 Hangul
characters) which can reduce the computational complexity quite a bit.

As the script runs, you should hopefully see the printed training accuracies
grow towards 1.0, and you should also see a respectable testing accuracy after
the training. When the script completes, the exported model we should use will
be saved, by default, as `./saved-model/optimized_hangul_tensorflow.pb`. This is
a [Protocol Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) file
which represents a serialized version of our model with all the learned weights
and biases. This specific one is optimized for inference-only usage.

### 5. Try Out the Model

Before we jump into making an Android application with our newly saved model,
let's first try it out. Provided is a [script](./tools/classify-hangul.py) that
will load your model and use it for inference on a given image. Try it out on
images of your own, or download some of the sample images below. Just make sure
each image is 64x64 pixels with a black background and white character color.

Optional flags for this are:

* `--label-file` for specifying a different label file. This is used to map indices
  in the one-hot label representations to actual characters.
  Default is _./labels/2350-common-hangul.txt_.
* `--graph-file` for specifying your saved model file.
  Default is _./saved-model/optimized_hangul_tensorflow.pb_.

Run it like so:

```
python ./tools/classify-hangul.py <Image Path> --label-file <your label file path>
```

***Sample Images:***

![Sample Image 1](doc/source/images/hangul_sample1.jpeg "Sample image")
![Sample Image 2](doc/source/images/hangul_sample2.jpeg "Sample image")
![Sample Image 3](doc/source/images/hangul_sample3.jpeg "Sample image")
![Sample Image 4](doc/source/images/hangul_sample4.jpeg "Sample image")
![Sample Image 5](doc/source/images/hangul_sample5.jpeg "Sample image")

After running the script, you should see the top five predictions and their
corresponding scores. Hopefully the top prediction matches what your character
actually is.

### 6. Create the Android Application

With the saved model, a simple Android application can be created that will be
able to classify handwritten Hangul that a user has drawn. A completed application
has already been included in [./hangul-tensordroid](./hangul-tensordroid).

#### Set up the project

The easiest way to try the app out yourself is to use
[Android Studio](https://developer.android.com/studio/). This
will take care of a lot of the Android dependencies right inside the IDE.

After downloading and installing Android Studio, perform the following steps:

1) Launch Android Studio
2) A **Welcome to Android Studio** window should appear, so here, click on
  **Open an existing Android Studio project**. If this window does not appear,
  then just go to **File > Open...** in the top menu.
3) In the file browser, navigate to and click on the _./hangul-tensordroid_ directory
   of this project, and then press **OK**.

After building and initializing, the project should now be usable from within
Android Studio. When Gradle builds the project for the first time, you might
find that there are some dependency issues, but these are easily resolvable in
Android Studio by clicking on the error prompt links to install the dependencies.

In Android Studio, you can easily see the project structure from the side menu.

![Android Project Structure](doc/source/images/android-project-structure.png "Project Structure")

The java folder contains all the java source code for the app. Expanding this
shows that we have just four java files:

1) **[MainActivity.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/MainActivity.java)**
   is the main launch point of the application and will
   handle the setup and button pressing logic.
2) **[PaintView.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/PaintView.java)**
   is the class that enables the user to draw Korean characters in a BitMap on
   the screen.
3) **[HangulClassifier.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/HangulClassifier.java)**
   handles loading our pre-trained model and connecting it with the TensorFlow
   Inference Interface which we can use to pass in images for classification.
4) **[HangulTranslator.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/HangulTranslator.java)**
   interfaces with the Watson Language Translator API to get English translations for our text.

In its current state, the provided Android application uses the _2350-common-hangul.txt_
label files and already has a pre-trained model trained on about 320,000 images
from 40 fonts. These are located in the _assets_ folder of the project,
_./hangul-tensordroid/app/src/main/assets/_.
If you want to switch out the model or labels file, simply place them in this directory.
You must then specify the names of these files in
[MainActivity.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/MainActivity.java),
_./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/MainActivity.java_, by
simply changing the values of the constants `LABEL_FILE` and `MODEL_FILE` located
at the top of the class.

#### Run the application

When you are ready to build and run the application, click on the green arrow
button at the top of Android Studio.

![Android Studio Run Button](doc/source/images/android-studio-play-button.png "Run Button")

This should prompt a window to **Select Deployment Target**. If you have an actual
Android device, feel free to plug it into your computer using USB. More info can be
found [here](https://developer.android.com/studio/run/device). If you do not
have an Android device, you can alternatively use an emulator. In the
**Select Deployment Target** window, click on **Create New Virtual Device**. Then
just follow the wizard, selecting a device definition and image (preferably an
image with API level 21 or above). After the virtual device has been created,
you can now select it when running the application.

After selecting a device, the application will automatically build, install, and
then launch on the device.

Try drawing in the application to see how well the model recognizes your Hangul
writing.


