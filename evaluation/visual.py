import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MAPVisualizer:

    def __init__(self, calculator):
        """
        Args:
            calculator: DirectoryMAPCalculator đã load data
        """
        self.calc = calculator

    def _get_classes(self):
        return sorted(set(
            [g["class_id"] for g in self.calc.ground_truths] +
            [p["class_id"] for p in self.calc.predictions]
        ))

    def _calc_pr_curve(self, class_id, iou_threshold):
        preds = [p for p in self.calc.predictions if p["class_id"] == class_id]
        gts   = [g for g in self.calc.ground_truths if g["class_id"] == class_id]

        if not gts:
            return [], [], 0.0

        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        gt_pool = {}
        for g in gts:
            gt_pool.setdefault(g["frame_id"], []).append({"box": g["box"], "matched": False})

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            fid      = pred["frame_id"]
            best_iou = 0
            best_idx = -1

            for j, gt in enumerate(gt_pool.get(fid, [])):
                iou = self.calc.calculate_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0 and not gt_pool[fid][best_idx]["matched"]:
                TP[i] = 1
                gt_pool[fid][best_idx]["matched"] = True
            else:
                FP[i] = 1

        acc_TP = np.cumsum(TP)
        acc_FP = np.cumsum(FP)

        recalls    = acc_TP / len(gts)
        precisions = acc_TP / (acc_TP + acc_FP + 1e-9)

        mrec = np.concatenate(([0], recalls, [1]))
        mpre = np.concatenate(([1], precisions, [0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ap = 0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

        return mrec, mpre, ap

    def plot(self, save_path=None):
        thresholds = np.arange(0.5, 1.0, 0.05)
        map_per_thresh = [self.calc.compute_map(t) for t in thresholds]
        map5095 = self.calc.compute_coco_map()
        map50 = self.calc.compute_map(0.5)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(thresholds, map_per_thresh, marker="o", color="#378ADD", linewidth=2, markersize=5)
        ax.fill_between(thresholds, map_per_thresh, alpha=0.15, color="#378ADD")
        ax.axhline(map5095, color="red", linestyle="--", linewidth=1, label=f"mAP@0.5:0.95 = {map5095:.4f}")
        ax.axhline(map50, color="green", linestyle="--", linewidth=1, label=f"mAP@0.5 = {map50:.4f}")

        ax.set_title("mAP across IoU thresholds", fontsize=12, fontweight="bold")
        ax.set_xlabel("IoU threshold")
        ax.set_ylabel("mAP")
        ax.set_xlim(0.48, 0.97)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"saved: {save_path}")

        plt.show()
    def save_report(self, save_path="res/map_report.txt"):
        classes = self._get_classes()
        thresholds = np.arange(0.5, 1.0, 0.05)
        map_per_thresh = [self.calc.compute_map(t) for t in thresholds]

        aps = {}
        for c in classes:
            _, _, ap = self._calc_pr_curve(c, 0.5)
            aps[c] = ap

        map50 = np.mean(list(aps.values()))
        map5095 = self.calc.compute_coco_map()
        with open(save_path, "w") as f:

            f.write("=" * 50 + "\n")
            f.write("mAP REPORT\n")
            f.write("=" * 50 + "\n\n")

            # tổng quan
            f.write("[summary]\n")
            f.write(f"  mAP@0.5          : {map50:.6f}\n")
            f.write(f"  mAP@0.5:0.95     : {map5095:.6f}\n")
            f.write(f"  total predictions: {len(self.calc.predictions)}\n")
            f.write(f"  total ground truth: {len(self.calc.ground_truths)}\n")
            f.write(f"  classes          : {len(classes)}\n\n")

            # AP từng class
            f.write("[AP per class @ IoU=0.5]\n")
            f.write(f"  {'class':<12} {'AP':>10}\n")
            f.write("  " + "-" * 24 + "\n")
            for c in classes:
                f.write(f"  class {c:<6}   {aps[c]:>10.6f}\n")
            f.write("\n")

            # mAP từng threshold
            f.write("[mAP across IoU thresholds]\n")
            f.write(f"  {'threshold':<12} {'mAP':>10}\n")
            f.write("  " + "-" * 24 + "\n")
            for t, score in zip(thresholds, map_per_thresh):
                f.write(f"  {t:<12.2f} {score:>10.6f}\n")
            f.write("\n")

            # PR curve từng class (recall / precision tại mỗi điểm)
            f.write("[PR curve per class @ IoU=0.5]\n")
            for c in classes:
                mrec, mpre, ap = self._calc_pr_curve(c, 0.5)
                f.write(f"\n  class {c}  AP={ap:.6f}\n")
                f.write(f"  {'recall':<12} {'precision':>12}\n")
                f.write("  " + "-" * 26 + "\n")
                for r, p in zip(mrec, mpre):
                    f.write(f"  {r:<12.4f} {p:>12.4f}\n")

            f.write("\n" + "=" * 50 + "\n")

        print(f"saved: {save_path}")