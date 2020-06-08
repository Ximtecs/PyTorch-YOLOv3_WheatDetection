rm -rf images/ labels/ train.csv sample_submission.csv train.txt valid.txt
kaggle competitions download global-wheat-detection
unzip global-wheat-detection.zip
rm global-wheat-detection.zip
mv train images
mkdir labels
python Yolo_setup.py
rm -r ../samples/*
mv test ../samples
