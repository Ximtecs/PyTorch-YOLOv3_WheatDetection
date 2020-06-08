VAR="experiments/tiny_parameters/yolov3-Wheat-tiny_"
DIR="experiments/tiny_parameters/"

for type in "SGD"
do
    for i in {8..9}
    do
        python train.py --model_def $VAR$i'.cfg' --optimizer $type --epochs 20
        rm -rf $DIR'/'$type$i;
        mkdir $DIR'/'$type$i
        mv checkpoints/*.pth $DIR'/'$type$i
        mv 'logs/scores.txt' $DIR'/'$type$i'/scores_'$type'_'$i'.txt' 
    done
done