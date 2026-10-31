import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import box_iou, calculate_detection_metrics

def test_iou_identical_boxes():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

def test_metrics():
    result = calculate_detection_metrics(tp=8, fp=2, fn=2)
    assert round(result["precision"], 2) == 0.80
    assert round(result["recall"], 2) == 0.80
    assert round(result["f1"], 2) == 0.80

if __name__ == "__main__":
    test_iou_identical_boxes()
    test_metrics()
    print("Evaluation tests passed.")
