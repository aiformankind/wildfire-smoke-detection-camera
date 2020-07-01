# wildfire-smoke-detection-hpwren

In this tutorial provided by [AI For Mankind](https://aiformankind.org/), we show you how to build an object detector for a wildfire smoke dataset using wildfire smoke images captured by HPWREN cameras http://hpwren.ucsd.edu/cameras/ . You can find the custom dataset in the input/images folder. The goal is to detect wildfire smokes in the images.

[Join AI For Mankind's Lets Stop Wildfires Hackathon 2.0](https://aiformankind.org/lets-stop-wildfires-hackathon-2.0/)

<p float="left">
<img src=docs/wildfire-detect.png />
</p>

We adapted the Dockerfile provided by Tensorflow and prepared a docker container preloaded with all the necessary libraries to get you started quickly to build your own object detection using custom wildfire smoke dataset. We modified the scripts provided by Tensorflow and from other excellent online tutorials eg. https://github.com/bourdakos1/Custom-Object-Detection to give an easy to follow step by step tutorial.

Install Docker:

https://docs.docker.com/v17.12/docker-for-mac/install/#install-and-run-docker-for-mac

Clone the repository (https://github.com/aiformankind/wildfire-smoke-detection-camera):
```
git clone https://github.com/aiformankind/wildfire-smoke-detection-camera.git
```

Go to the repository directory that you just clone:
```
cd wildfire-smoke-detection-camera
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

### Experimental Results

AI For Mankind had annotated these 744 wildfire smoke HPWREN images in 2019 and built a wildfire smoke detector aka "The Super Duper" using the annotated images. It has shown promising results running against wildfire images captured by HPWREN in 2020. See below.

1. 2020-02-05 fire captured by HPWREN camera hp-w-mobo-c, our detector detected the smoke *13 minutes* after fire ignition.

    <br/>
    <a href="https://youtu.be/CcbXdcMEQvs"><img src="docs/smoke_detector_in_action_13mins_detected.png" width="500" ></a>
    <br clear="right"/>


2. 2020-03-06 fire captured by HPWREN camera mlo-n-mobo-c, our detector detected the smoke *3 minutes* after fire ignition. See video below.

    <br/>
    <a href="https://youtu.be/X_QvjA1-Nb4"><img src="docs/smoke_detector_in_action_3mins_detected.png" width="500" ></a>

    <br clear="right"/>

    <br/>


3. 2020-05-21 fire captured by HPWREN camera VEGMGMT ml-w-mobo-c, our detector detected the smoke *16 minutes* after fire ignition.
<img src="docs/20200521_VEGMGMT_ml-w-mobo-c_1590081617_+00960_smoke_detected.png" alt="Wildfire Smoke Detector in Action" width="450"/>

### Other Results

#### True Positives
We ran AI For Mankind's wildfire smoke detector aka "The Super Duper" against past HPWREN images of several fires, here are the promising results obtained.

- 20191006: Detected smoke ~6 mins after fire ignition (HPWREN ml w mobo c) https://youtu.be/wt1sQyRjoCI
- 20191006: Detected smoke ~7 mins after fire ignition (HPWREN lp n mobo c) https://youtu.be/dViR_XGQ8Oo
- 20191006: Detected smoke ~10 mins after fire ignition (HPWREN lp s mobo c) https://youtu.be/XVvZVnxHv4A
- 20191006: Detected smoke ~3 mins after fire ignition (HPWREN pi s mobo c) https://youtu.be/e9T_8coM20M
- 20191007: Detected smoke ~10 mins after fire ignition (HPWREN sm s mobo c) https://youtu.be/LqAxrY-Xa4w
- 20200206: Detected smoke ~4 mins after fire ignition (HPWREN ml s mobo c) https://youtu.be/Y3tal1-nk1Y
- 20200306: Detected smoke ~3 mins after fire ignition (HPWREN mlo n mobo c) https://youtu.be/X_QvjA1-Nb4
- 20200205: Detected smoke ~13 mins after fire ignition (HPWREN hp w mobo c) https://youtu.be/CcbXdcMEQvs

**See the complete playlist:** [https://www.youtube.com/playlist?list=PLB-XesK9mcaCCvSbogU9SFBlp1LEAjgT2](https://www.youtube.com/playlist?list=PLB-XesK9mcaCCvSbogU9SFBlp1LEAjgT2)

<img src="docs/truePositive1.png" alt="detected smoke" width="400"/><img src="docs/truePositive2.png" alt="detected smoke" width="400"/>

<img src="docs/truePositive5.png" alt="detected smoke" width="400"/>

#### False Positives

<img src="docs/falsePositive1.png" alt="not smoke" width="400"/>

### Model Performance
AI For Mankind had developed 3 models: SuperDuper-v1, SuperDuper-v2, and SuperDuper-edge. One of them, SuperDuper-edge is optimized for edge device. The table shows the performance of our models.


| Name            | AveragePrecison 0.5IOU/smoke |
|-----------------|-----------------|
| SuperDuper-v1   | 0.7506           |
| SuperDuper-v2   | 0.8669           |
| SuperDuper-edge | 0.6822           |


### False Positive Rate
We tested our model against different time periods eg. during sunrise, sunset, or foggy condition and shared our false positive rates below. We will continue to curate, test, and share the results and datasets.

| False Positive Study |               |              |                        |                     |                                                                           |
|----------------------|---------------|--------------|------------------------|---------------------|---------------------------------------------------------------------------|
|                      | Model         | Total Images | Num of False Positives | False Positive Rate | Link to Dataset                                                           |
| Sunrise              | SuperDuper-v1 | 181          | 2                      | 0.011       | https://www.dropbox.com/sh/71jdkv7tdtmmif8/AACdd51AH4BNX84bJSrGWrssa?dl=0 |
| Fog                  | SuperDuper-v1 | 181          | 72                     | 0.398        | https://www.dropbox.com/sh/iw40v0yrkkimhha/AAANC4cxJR90cp8cfXF5kYHaa?dl=0 |

On average, false positive rate ~0.0860

### Join Our Slack Channel
- [AI For Mankind Slack Channel](https://tinyurl.com/vch2z68)

### Join Our Meetup Group
- [Join AI For Mankind Meetup Group](https://www.meetup.com/AI-for-Mankind/) to receive event announcement.

### AI For Mankind's Youtube Channel
- [Presentations](https://www.youtube.com/channel/UCyosmgDqG3jfZ4_aYsN94Lw)


### Donate to Support Us
[Donate](https://donorbox.org/support-the-evaluation-and-deployment-of-wildfire-smoke-detector) to support our efforts. Your donation is tax deductible. 
