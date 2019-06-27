[//]: # (Image References)

[Loss-Graph]: https://github.com/safdark/SDC-Semantic-Segmentation/blob/master/docs/images/loss_graph.png "Loss Graph"
[Welcome]: https://github.com/safdark/SDC-Semantic-Segmentation/blob/master/docs/images/welcome.png "Welcome"

# Semantic Segmentation

![Welcome][Welcome]

This project is about teaching a DNN to classify pixels in an image as belonging to one of multiple classes.

The initial cut of the project uses 2 classes:
- Road
- Non-Road

## Training

### How to run

To run training, the following command-line utility exists:

```text

usage: trainer.py [-h] [-r] [-o MODEL_FOLDER] -e NUM_EPOCHS

Trainer

optional arguments:
  -h, --help       show this help message and exit
  -r               Will load the model from disk if available (default:
                   false).
  -o MODEL_FOLDER  Location of model on disk
  -e NUM_EPOCHS    Number of epochs to run
```

For example:
```text
> python trainer.py -r -o my_model -e 60
```

This will create a model that can be used for inference (infer_images.py), at the following location:
```text
/<Project-Root>
    /models
        /my_model
```

### Architecture Highlights

I've utilized transfer learning from a pretrained VGG16 network, and a fully-convolutional decoder that deconvolves the features extracted by the encoder, through 4 deconvolutional layers, to produce the expected output.

#### Skip Connections

Skip connections exist that allow deconvolutional layers to use spatial information from the features extracted by the encoder earlier in the pipeline.

I have achieved this by *concatenating* those layers with the corresponding layers in the decoder, as follows:
- Deconv Layer 4 <=concat=> VGG Layer 4 output
- Deconv Layer 3 <=concat=> VGG Layer 3 output

An alternative was to just add the tensor elements together, but I felt that that was not as effective at preserving the spatial information as concatenation was.

#### Activation Layers

I've used ReLU activations for the 2 Deconv layers: 3 and 4. This non-linearity improves the learning capacity of the network's additional layers.

#### Regularization: Data Augmentation

While generating training inputs, the system probabilistically augments the images by:
- Randomly playing with the color levels in the image
- Randomly flipping the image horizontally
- Randomly cropping the image

This regularization technique helps the network generalize better to the test data. 

#### Training loss curve

A training run was performed with 80 epochs, and produced the following training losses. 

![Loss-Graph][Loss-Graph]

The command used to generate this was:
```text
> python trainer.py -r -o model1 -e 80
```

## Image Inference

### How to run

Once training has been done (using trainer.py, above), the following command-line utility performs inference on the test images located at:
```text
/<Project-Root>
    /data
        /data_road
            /testing
                /images
                    /...
                /gt_images
                    /...
```

```text
usage: infer_images.py [-h] [-o MODEL_FOLDER] [-n COUNT]

Image Segmentation Inference

optional arguments:
  -h, --help       show this help message and exit
  -o MODEL_FOLDER  Location of model on disk
  -n COUNT         Number of images to segment (default: All)
```

For example:
```text
> python infer_images.py -o my_model
```

The output of the inference run will produce a folder, as below:
```text
/<Project-Root>
    /runs
        /<timestamp>
            /mask-<image-name>.png  <======== The pure segmentation output (road/non-road)
            /<image-name>.png       <======== The image of the segmented road overlayed on top of the original image, for illustration
```

### Sample Output

Sample output after running inference on the test images is located in the following zip file:
```text
/<Project-Root>
    /runs
        /inferences.zip
```

Please unzip using the following command, to view the files:
```text
> tar -xzvf inferences.zip
```

Please ignore the "mask-" prefixed file names ... those are merely the segmentation masks. Look at the files that do not have the "mask-" prefix for the actual image with the road sections colored in green.
