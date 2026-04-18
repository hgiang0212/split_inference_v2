from ultralytics.utils import ops, nms
import os



        for i, pred in enumerate(preds):
            square_size = max(h, w)
            pred[:, :4] = ops.scale_boxes(img, pred[:, :4], (square_size, square_size))
                for box in pred:
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h