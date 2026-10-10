from __future__ import annotations

import unittest

import numpy as np

from backend.detector import detect_objects


class FakeTensor:
    def __init__(self, values: list | np.ndarray) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def astype(self, dtype: type) -> np.ndarray:
        return self.values.astype(dtype)


class FakeBoxes:
    def __init__(
        self,
        coordinates: list[list[int]],
        confidences: list[float],
        class_ids: list[int],
    ) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(class_ids)

    def __len__(self) -> int:
        return len(self.conf.values)


class FakeModel:
    names = {0: "fire", 1: "smoke", 2: "other"}

    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes
        self.predict_arguments: dict = {}

    def predict(self, **kwargs: object) -> list:
        self.predict_arguments = kwargs
        return [type("Prediction", (), {"boxes": self.boxes})()]


class DetectObjectsRegressionTests(unittest.TestCase):
    def test_preserves_inference_options_and_detection_outputs(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        model = FakeModel(
            FakeBoxes(
                coordinates=[
                    [10, 10, 30, 30],
                    [35, 35, 95, 95],
                    [0, 0, 20, 20],
                    [70, 0, 90, 20],
                    [0, 70, 10, 80],
                ],
                confidences=[0.91, 0.88, 0.76, 0.99, 0.19],
                class_ids=[0, 0, 1, 2, 0],
            )
        )

        annotated, stats = detect_objects(
            frame=frame,
            model=model,
            conf_threshold=0.2,
            iou_threshold=0.4,
            image_size=512,
            max_detections=4,
            device="cpu",
        )

        expected_options = {
            "conf": 0.2,
            "iou": 0.4,
            "imgsz": 512,
            "max_det": 4,
            "device": "cpu",
            "verbose": False,
        }
        self.assertEqual(
            expected_options,
            {key: model.predict_arguments[key] for key in expected_options},
        )
        self.assertIs(model.predict_arguments["source"], frame)
        self.assertEqual(
            ["Live Flame", "Fire", "Smoke"],
            [detection["class_name"] for detection in stats["detections"]],
        )
        self.assertEqual(
            [0.91, 0.88, 0.76],
            [detection["confidence"] for detection in stats["detections"]],
        )
        self.assertEqual(
            {"Live Flame": 1, "Fire": 1, "Smoke": 1},
            stats["class_counts"],
        )
        self.assertEqual("CRITICAL", stats["hazard_level"])
        self.assertEqual(3, stats["total_detections"])
        self.assertEqual((100, 100, 3), annotated.shape)
        self.assertTrue(np.any(annotated != frame))

    def test_no_detections_returns_clear_telemetry(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        model = FakeModel(FakeBoxes([], [], []))

        annotated, stats = detect_objects(frame=frame, model=model)

        self.assertEqual("CLEAR", stats["hazard_level"])
        self.assertEqual(0, stats["total_detections"])
        self.assertEqual([], stats["detections"])
        self.assertEqual({}, stats["class_counts"])
        self.assertEqual(0.0, stats["highest_confidence"])
        np.testing.assert_array_equal(frame, annotated)


if __name__ == "__main__":
    unittest.main()
