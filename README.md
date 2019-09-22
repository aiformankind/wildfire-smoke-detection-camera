# wildfire-smoke-detection-hpwren

In this tutorial, we will show you how to build an object detector for a wildfire smoke dataset using wildfire smoke images captured by HPWREN cameras http://hpwren.ucsd.edu/cameras/ . You can find the custom dataset in the input/images folder. The goal is to detect wildfire smokes in the images.

<p float="left">
<img src=docs/wildfire-detect.png />
</p>

We adapted the Dockerfile provided by Tensorflow and prepared a docker container preloaded with all the necessary libraries to get you started quickly to build your own object detection using custom wildfire smoke dataset. We modified the scripts provided by Tensorflow and from other excellent online tutorials eg. https://github.com/bourdakos1/Custom-Object-Detection to give an easy to follow step by step tutorial.

Install Docker:

https://docs.docker.com/v17.12/docker-for-mac/install/#install-and-run-docker-for-mac

Clone the repository (https://github.com/aiformankind/wildfire-smoke-detection):
```
git clone https://github.com/aiformankind/wildfire-smoke-detection.git
```

Go to the repository directory that you just clone:
```
cd wildfire-smoke-detection
```

Build the Tensorflow docker (this job will pull the latest tensorflow images and set up the environment) :
```
docker build -t aiformankind/wildfiredetection:0.0.1 .
```

Start the Tensorflow container (this job will spin up the objectdetection container):
```
docker run -it -p 8888:8888 -p 6006:6006 --name=wildfiredetection aiformankind/wildfiredetection:0.0.1
```
Train model:
You can stop the training by pressing CTRL-C after iteration step ~200.
```
python models/research/object_detection/legacy/train.py --logtostderr --train_dir=/tensorflow/train/ --pipeline_config_path=/tensorflow/train/faster_rcnn_resnet101.config
```

Create a frozen inference graph:
Replace model.ckpt-XXX with the largest(latest) checkpoint file number. Look it up in the /tensorflow/train folder

```
python models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /tensorflow/train/faster_rcnn_resnet101.config --trained_checkpoint_prefix /tensorflow/train/model.ckpt-XXX --output_directory /tensorflow/inference_graph
```

Start jupyter notebook
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

You can access the jupyter notebook on browser. Find the object_detection_wildfire.ipynb in /tensorflow/models/research/object-detection folder.

Run the notebook to see the our model in action. In the last cell, we use the model to detect and identify wildfire smokes in the test images.

### Prepare Custom Dataset
First, you have to annotate your images by building bounding boxes around the objects you want to detect. There are many tools you can use. Among them are [labelImg](https://github.com/tzutalin/labelImg), [RectLabel](https://rectlabel.com/), and [Labelbox](https://labelbox.com/).

You can find the custom images for this tutorial in the /tensorflow/input/images folder. The annotations of bounding boxes are described using XML format. These xmls are stored in /tensorflow/input/annotations/xmls folder. See <bndbox> XML element which describes the bounding box in the xml below. All the annotation tools mentioned above provide easy to use UI interface to draw the bounding boxes(annotations) and have the export option to create these XMLs automatically from your annotated images. 

```
<annotation>
    <folder>tmpenqgsts7</folder>
    <filename>cjpqohplrgdvq0873x9oy8cmn.jpeg</filename>
    <path>/tmp/tmpenqgsts7/cjpqohplrgdvq0873x9oy8cmn.jpeg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>Smoke</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
                <xmin>86</xmin>
                <ymin>189</ymin>
                <xmax>124</xmax>
                <ymax>219</ymax>
            </bndbox>
        </object>
    </annotation>
```        
