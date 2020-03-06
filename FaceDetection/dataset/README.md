# Dataset

## Wider Face Dataset

학습을 위한 Face Data는 실제 학습 후 성능이 좋았던 Wider Face라는 데이터셋을 사용합니다.

<a href='http://shuoyang1213.me/WIDERFACE/'>이곳</a>의 Download에서 다음의 세가지 파일을 다운받고 압축을 해제합니다.

- WIDER Face Training Images
- WIDER Face Validation Images
- Face annotations

## image, annotation 파일 분류하기

편의를 위해 세 파일의 압축을 풀고 나온 디렉토리들을 하나의 디렉토리에 옮깁니다. (여기서는 datasets라는 디렉토리를 생성하고 그곳에 옮겼다고 가정하고 진행하겠습니다)

```bash
user@ubuntu:~/datasets$ ls
wider_face_split  WIDER_train  WIDER_val
```

wider_face_split에는 각 이미지의 label에 관한 정보가 있고, WIDER_train, WIDER_val에는 이미지들이 있으며 장소별로 분류되어있습니다.

wider_face_split에서는 wider_face_train_bbx_gt.txt, wider_face_val_bbx_gt.txt을 통해 label 정보를 얻을 것입니다.

WIDER_train, WIDER_val에서는 이미지 파일들을 얻을 것입니다. 하지만 이미지 파일들은 각 장소별로 분류될 필요가 없습니다.

따라서, 다음과 같은 구조로 wider_face_split에서는 필요한 두 파일만을 남기고, WIDER_train, WIDER_val의 장소별로 분류된 이미지들을 한곳에다 모을 것입니다.

/datasets  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/wider_face_split  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/wider_face_train_bbx_gt.txt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/wider_face_val_bbx_gt.txt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/WIDER_train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/0.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/WIDER_val  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/0.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/...

다음의 과정을 통해 위의 구조로 만들어줍니다.

```bash
# wider_face_split
user@ubuntu:~/datasets$ ls
wider_face_split  WIDER_train  WIDER_val
user@ubuntu:~/datasets$ cd ./wider_face_split
user@ubuntu:~/datasets/wider_face_split$ ls
readme.txt                    wider_face_train_bbx_gt.txt  wider_face_val.mat
wider_face_test_filelist.txt  wider_face_train.mat
wider_face_test.mat           wider_face_val_bbx_gt.txt
user@ubuntu:~/datasets/wider_face_split$ for i in `ls | grep -v "bbx"`; do rm ${i}; done
user@ubuntu:~/datasets/wider_face_split$ ls
wider_face_train_bbx_gt.txt  wider_face_val_bbx_gt.txt

# WIDER_val, WIDER_train
user@ubuntu:~/datasets$ ls
wider_face_split  WIDER_train  WIDER_val
user@ubuntu:~/datasets$ cd ./WIDER_val/images
user@ubuntu:~/datasets/WIDER_val/images$ for i in `ls`; do mv ${i}/* ..; done
user@ubuntu:~/datasets/WIDER_val/images$ cd ..
user@ubuntu:~/datasets/WIDER_val$ rm -rf ./images
user@ubuntu:~/datasets/WIDER_val$ cd ../WIDER_train/images
user@ubuntu:~/datasets/WIDER_train/images$ for i in `ls`; do mv ${i}/* ..; done
user@ubuntu:~/datasets/WIDER_train$ rm -rf ./images
```

## label 파일(\*.csv)과 tfrecord 파일(\*.record)

label 파일이란 각 이미지의 어떤 위치에 어떤 Object가 있는지를 기록한 파일이며 주로 \*.csv 파일을 통해 기록합니다.

tfrecord 파일이란 텐서플로우로 딥러닝 학습을 하는데 필요한 데이터들을 보관하기 위한 데이터 포맷을 뜻하며, tfrecord 파일을 만들기 위해서는 이미지 파일과 label 파일이 필요합니다. 또한, 여기서 사용할 Tensorflow Object Detection API에도 tfrecord 파일을 통해 학습시키므로 이를 만들어주어야 합니다.

따라서, 원래는 각 image마다 <a href="https://github.com/tzutalin/labelImg">labelImg</a>와 같은 도구를 이용해 직접 label 파일(\*.csv)을 만들어 준 후 이미지들과 함께 tfrecord 파일로 변환시켜주는 과정이 필요합니다.

하지만 Wider Face Dataset에서 제공된 wider_face_train_bbx_gt.txt, wider_face_val_bbx_gt.txt에 label들에 대한 정보들이 이미 있으므로 여기서는 파이썬을 통해 그 정보들을 parsing해서 \*.csv파일을 만들고 tfrecord 파일로 변환시켜 사용하겠습니다.

## label (\*.csv) 파일 만들기

make_csv.py를 내려받아 datasets 디렉토리에서 실행시키면 wider_face_train_bbx_gt.txt, wider_face_val_bbx_gt.txt의 정보를 통해 train.csv, val.csv 파일이 생성됩니다.

(실행 중 출력되는 4개의 파일은 label 갯수가 0인 파일들인데 어떻게 해야할지 몰라 이들은 그냥 지웠습니다...)

```bash
user@ubuntu:~/datasets$ ls
make_csv.py  wider_face_split  WIDER_train  WIDER_val
user@ubuntu:~/datasets$ python ./make_csv.py
/home/user/datasets/WIDER_train/0_Parade_Parade_0_452.jpg
/home/user/datasets/WIDER_train/2_Demonstration_Political_Rally_2_444.jpg
/home/user/datasets/WIDER_train/39_Ice_Skating_iceskiing_39_380.jpg
/home/user/datasets/WIDER_train/46_Jockey_Jockey_46_576.jpg
user@ubuntu:~/datasets$ ls
make_csv.py  train.csv  val.csv  wider_face_split  WIDER_train  WIDER_val
```

## tfrecord 파일 만들기 (\*.record)

generate_tfrecord.py를 내려받은 후 해당 파일의 제일 윗 주석 부분(Usage)의 script를 datasets 디렉토리에서 다음과 같이 실행시키면 앞에서 생성한 label 파일과 이미지 파일을 통해 tfrecord 파일이 생성됩니다.

(여기서 사용한 generate_tfrecord.py는 https://github.com/datitran/raccoon_dataset 의 generate_tfrecord.py의 제일 윗줄 주석 부분(Usage)과 30~35번째 줄을 저희 데이터셋에 맞게 수정한 것입니다)

```bash
user@ubuntu:~/datasets$ ls
generate_tfrecord.py  make_csv.py  train.csv  val.csv  wider_face_split  WIDER_train  WIDER_val
user@ubuntu:~/datasets$ python generate_tfrecord.py --csv_input=./train.csv --image_dir=./WIDER_train --output_path=./train.record
WARNING: Logging before flag parsing goes to stderr.
W0905 15:08:58.575685 140687487387456 deprecation_wrapper.py:119] From generate_tfrecord.py:100: The name tf.app.run is deprecated. Please use tf.compat.v1.app.run instead.

W0905 15:08:58.576304 140687487387456 deprecation_wrapper.py:119] From generate_tfrecord.py:86: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

W0905 15:09:03.905350 140687487387456 deprecation_wrapper.py:119] From generate_tfrecord.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

Successfully created the TFRecords: /home/user/datasets/train.record
user@ubuntu:~/datasets$ python generate_tfrecord.py --csv_input=./val.csv --image_dir=./WIDER_val --output_path=./val.record
WARNING: Logging before flag parsing goes to stderr.
W0905 15:09:52.593293 140525197961024 deprecation_wrapper.py:119] From generate_tfrecord.py:100: The name tf.app.run is deprecated. Please use tf.compat.v1.app.run instead.

W0905 15:09:52.593990 140525197961024 deprecation_wrapper.py:119] From generate_tfrecord.py:86: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

W0905 15:09:54.012392 140525197961024 deprecation_wrapper.py:119] From generate_tfrecord.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

Successfully created the TFRecords: /home/user/datasets/val.record
user@ubuntu:~/datasets$ ls
generate_tfrecord.py  train.csv     val.csv     wider_face_split  WIDER_val
make_csv.py           train.record  val.record  WIDER_train
```
