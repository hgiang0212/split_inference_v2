from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops, nms
import os


class Predictions(DetectionPredictor):
    def get_file_predictions(self, preds, img=(640, 640), frame_idx=0, orig_img_shape=(),OUTPUT_DIR="dataset/predictions" ):
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # tự tạo thư mục nếu chưa có

        (h, w) = orig_img_shape
        preds = nms.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)
        frame_id = frame_idx
        for i, pred in enumerate(preds):
            output_file = os.path.join(OUTPUT_DIR, f"frame_{frame_id + i:06d}.txt")
            square_size = max(h, w)
            pred[:, :4] = ops.scale_boxes(img, pred[:, :4], (square_size, square_size))
            if h > w:
                pred[:, 0].clamp_(0, w)  # x1
                pred[:, 2].clamp_(0, w)  # x2
            else:
                pred[:, 1].clamp_(0, h)  # y1
                pred[:, 3].clamp_(0, h)  # y2
            with open(output_file, "w") as f:
                for box in pred:
                    x1, y1, x2, y2, conf, classes = box
                    classes = int(classes)
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{classes} {cx} {cy} {bw} {bh} {conf}\n")