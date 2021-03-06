"""
Create anchors using Kmeans clustering on dataset.  Creates a text file
to use during training.

Run convert_labels.py first on the data/obj folder to get an intermediate file
to use with this script.
"""

import numpy as np
import argparse
import json


class YOLO_Kmeans:
    def __init__(self, cluster_number, annot_file, out_file):
        self.cluster_number = cluster_number
        self.annot_file = annot_file
        self.out_file = out_file

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(self.out_file, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self, files):
        img_size = 416
        dataSet = []
        for file in files:
            f = open(file,'r')
            for line in f:
                infos = line.split(" ")
                width = int(float(infos[3]) * img_size)
                height = int(float(infos[4]) * img_size)
                dataSet.append([width, height])
            f.close()
        result = np.array(dataSet)
        return result

    def txt2clusters(self, files):
        all_boxes = self.txt2boxes(files)
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        result_print = json.dumps(result.tolist())
        print("K anchors:\n {}".format(result_print))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))

    def txt2filenames(self):
        f = open(self.annot_file, 'r')
        default_path = "data/WheatDetection/labels/"
        files = []
        for line in f:
            infos = line.split("images/")
            name =  infos[1].strip()[:-4]
            files.append(f'{default_path}{name}.txt')
        return files





if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Command line options
    parser.add_argument(
        '--num_clusters', type=int,
        default=9,
        help='Number of desired clusters or anchors'
    )

    # Command line options
    parser.add_argument(
        '--annot_file', type=str,
        help='Annotations file name (list of images and bboxes)'
    )

    parser.add_argument(
        '--out_file', type=str,
        default="yolo_anchors_custom.txt",
        help='Output file name'
    )

    args = parser.parse_args()

    cluster_number = args.num_clusters
    annot_file = args.annot_file
    out_file = args.out_file
    kmeans = YOLO_Kmeans(cluster_number, annot_file, out_file)
    files = kmeans.txt2filenames()
    #res = kmeans.txt2boxes(files)
    kmeans.txt2clusters(files)