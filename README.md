# Mass Deepfake Video Generation Using DeepFaceLab Tool

This repository is an adaptation of the [DeepFaceLab](https://github.com/iperov/DeepFaceLab) GitHub project in order to mass generate deepfake videos instead of one.

This project is not finished and might cause some error during the process of video generation.

The purpose is to generate deepfake between each video with each other.

## Getting Started

### Prerequisites

- Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Environment creation :

```bash
$ conda create -n deepfacelab python=3.7

$ conda activate deepfacelab

# Install your GPU lib Nvidia or AMD
$ conda install -c conda-forge cudatookit cudnn

# Either
$ pip install -r requirements.txt

# Or
$ conda install -c conda-forge --file requirements.txt
```

- Define a temporary folder to place your raw video, a folder for the different processes output (subjects) and finally a folder for the *SAEHD* model to use for training:

```bash
$ TMP_RAW_VIDEOS="/path/to/tmp/videos/folder"
$ SUBJECTS_DIR="/path/to/your/output/subjects/folder"
$ MODEL_DIR="/path/to/your/models/dir"
```

### First step: Workspace creation

Organize your video into subject folder tree :

```bash
# Usage
$ python auto_main.py to_subject -h

$ python auto_main.py to_subject \
--videos_dir $TMP_RAW_VIDEOS \
--subjects_dir $SUBEJCTS_DIR
```

At the end of this scrip you should have a number of subject equivalent of the number of raw videos. Moreover, the original videos inside the subject's folder should be renamed **output.***.

If you want to remove subjects you should use this scrip: ```$ python auto_main.py update_wrk -h```, or add subjects (never tested) you should re-use the script above.

You can also clean the **workspace** by removing the unused frames during the process using the following script:

```bash
# Usage
$ python auto_main.py clean -h

# ONE AT THE TIME
# Remove the merged frames
$ python auto_main.py clean --subjects_dir $SUBJECTS_DIR --redo_merge

# Remove aligned and original frames
$ python auto_main.py clean --subjects_dir $SUBJECTS_DIR --redo_original

# Remove extracted faces (different frames than the aligned one)
$ python auto_main.py clean --subjects_dir $SUBJECTS_DIR --redo_face
```

### Second step: Extract Frames and Alignments

Make frames from the *original videos* and extract the *alignment* :

```bash
# Usage
$ python auto_main.py extract -h

$ python auto_main.py extract \
--subjects_dir $SUBJECTS_DIR \
--dim_output_faces 1024 \
--png_quality 90
```

The better the dimension is the better the deepfake quality will be, same for the png quality.

### Third step: Model Choice and Pretrain

To generate *deepfake* it is better to use a pretrained model on internet (speed-up process, better face generalization). You can find model on the [***DeepfakeVFX***](https://www.deepfakevfx.com/) website. It will help you choose your model parameters according to your computer resources and other useful **tools**.

You can **pretrain** your model on your face set to hopefully have better and faster results (**mandatory** if you create your own model from scratch) by either **monitoring the loss** and quit the programs when you are satisfied or set a **maximum number iteration** (should be higher than the current reached iteration by the model in the case you are using a pretrained model from internet) 

**Pack** the faces :

```bash
# Usage
$ python auto_main.py pack -h

$ python auto_main.py pack --subjects_dir $SUBJECTS_DIR
```

**Pretrain** your model (Follow the instructions at the beginning of the script) :

```bash
# Usage
$ python auto_main.py pretrain -h

# The program will open a new terminal to setup the model argument, remember to modify the 
# iteration goal (higher than the current reached iteration) and the pretrain value (set it to true).
$ python auto_main.py pretrain \
--subjects_dir $SUBJECTS_DIR \
--model_dir $MODEL_DIR \
--model_dir_backup $MODEL_DIR + "my-backup"
```

*OPTIONAL*: You can also create a benchmark on different models to choose the best one that suit your need :

```bash
# Usage
$ python auto_main.py face_swap_benchmark -h
```

The *benchmark* will also help you to choose which interation should be enough to generate the videos in the next part.

## Fourth step: Train on Subjects and Merge the Frames

Two processes are available to execute this step:

- auto training and auto merging.
- auto training and manual merging.

1. Auto training with Auto merging :

This method will **automatically** train on 2 subjects and merge the predicted face into frames and a video. The **chose parameters** can be adapted by modifying the code **line 213**, for instance, in the file [MergerConfig.py](merger/MergerConfig.py).

Usage example :
```bash
# Usage
$ python auto_main.py swap_auto -h

# The argument iteration_goal does not work and could be a nice feature to add.
# The program will open a new terminal where you will be ask to choose the parameters, remember to change the 
# iteration goal and pretrain value to false !!
$ python auto_main.py swap_auto \
--subjects_dir $SUBJECTS_DIR \
--model_dir $MODEL_DIR
```

2. Auto training with Manual merging :

This method will **automatically** train all the necessary model to predict faces and save them (it can easily reach a hundred of GB), and then will ask you to **manually** merge the different subjects.

Usage example :

```bash
# Usage to train models
$ python auto_main.py swap_flexible_train -h

# The argument iteration_goal does not work and could be a nice feature to add.
# The program will open a new terminal where you will be ask to choose the parameters, remember to change the 
# iteration goal and pretrain value to false !!
$ python auto_main.py swap_flexible_train \
--subjects_dir $SUBJECTS_DIR \
--model_dir $MODEL_DIR
```

```bash
# Usage to merge faces
$ python auto_main.py swap_flexible_merge -h

# The program will automatically select the remaining subject to merge and will open a new 
# terminal for you te execute the merge process, then you will have to close the terminal when the 
# merging status reach 100% because it does not properly close (that is why a new terminal is needed to
# avoid restarting the program for every subjects)
# Remember to set the interactive merger to true every time a new terminal is open.
$ python auto_main.py swap_flexible_merge \
--subjects_dir $SUBJECTS_DIR \
--model_dir $MODEL_DIR
```

*OPTIONAL*: to clean all the trained models you can do : ```$ python auto_main.py clean_flexible_train -h```

You can also test the merged frames to see if they bypass a face recognition algorithm :

```bash
# Usage
$ python auto_main.py frames_generated_benchmark -h
```

this algorithm uses the [YuNet](YuNet_model_face_recognition) onxx model. You can modify the face extraction / face recognition algorithm if you want.

### Fifth step: DataFrame creation

It is useful to create a DataFrame to use this dataset for another project in order to make a frame referencing.

***First*** you need to extract the faces of the merged and original frames (same remark: you cna modify the face extraction [method](scripts/extract/face))

```bash
# Usage
$ python auto_main.py extract_face_from_subject -h 

# You can also put other videos and non face extracted frame (real or fake) in the random_data_augmentation folder to
# add more data to the dataset

# Usage
$ python auto_main.py extract_face_from_video_data_augmentation -h
```

**Then** you can create the DataFrame:

```bash
# Usage
$ python auto_main.py dataframe -h
```

This algorithm will not add two identical images (SHA256 verification), it can also be stopped or enhanced.

DataFrame main structure: 

| Index                      | video                                           | original                                                                  | label                    | SHA256              | top                                                                                                  | bottom              | left              | right              |
|:---------------------------|:------------------------------------------------|:--------------------------------------------------------------------------|:-------------------------|:--------------------|:-----------------------------------------------------------------------------------------------------|:--------------------|:------------------|:-------------------|
| relative path to the frame | relative path of the video related to the frame | relative path of the original video related to fake frame (possible None) | True (Fake) False (Real) | SHA256 of the frame | top corner of the image (useful if you did not extracted the face but just inform the extracted box) | same but for bottom | same but for left | same but for right |

## Possible Upgrade

- Change the [Subject](scripts/Subject.py) class for better organization of images;
- Fix the possibility to select the iteration goal as script arg;
- Add to possibility to automatically select the pretrain mode --> avoid terminal opening to select intern parameters;
- Find a better way to merge the predicted faces.
