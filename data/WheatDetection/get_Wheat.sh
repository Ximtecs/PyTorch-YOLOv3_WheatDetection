kaggle competitions download global-wheat-detection
unzip global-wheat-detection.zip
rm global-wheat-detection.zip
mv train images
mkdir labels
python Yolo_setup.py
mv test ../samples