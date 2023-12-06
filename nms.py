import numpy as np 
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


#The provided Python code defines a function named non_max_suppression_fast that implements a fast and efficient algorithm for non-maximum suppression (NMS) on a set of bounding boxes.
# NMS is commonly used in object detection to filter out redundant bounding boxes and retain only the most relevant ones.
# The function takes two parameters: boxes, representing the input bounding boxes as a NumPy array, and overlapThresh, which determines the overlap threshold for considering two bounding boxes as duplicates.
# The algorithm iteratively selects the bounding box with the highest y-coordinate, adds it to the list of picked boxes, and removes any boxes with significant overlap.
# The process continues until all boxes are examined. The code utilizes NumPy for array manipulations and efficiently handles different data types for bounding box coordinates.
# Additionally, it includes exception handling to capture and print any exceptions that might occur during execution.
# Overall, this code provides a streamlined implementation of the non-maximum suppression algorithm for bounding box post-processing in object detection pipelines.