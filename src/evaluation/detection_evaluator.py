"""
Detection Evaluator for CV Pipeline.

Computes object detection metrics:
- mAP (mean Average Precision)
- mAP@50, mAP@75
- Precision, Recall at various IoU thresholds
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_evaluator import BaseEvaluator


class DetectionEvaluator(BaseEvaluator):
    """
    Evaluator for object detection tasks.

    Computes COCO-style metrics:
    - mAP (IoU 0.5:0.95)
    - mAP@50 (IoU 0.5)
    - mAP@75 (IoU 0.75)
    - AR (Average Recall)

    Supports both YOLO and torchvision detection outputs.

    Example:
        evaluator = DetectionEvaluator(
            model=yolo_model,
            test_loader=test_loader,
            class_names=['person', 'car', 'dog']
        )
        metrics = evaluator.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
        conf_threshold: float = 0.25,
    ):
        super().__init__(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            class_names=class_names
        )

        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.conf_threshold = conf_threshold

        # Detection-specific storage
        self.all_predictions: List[Dict] = []
        self.all_targets: List[Dict] = []

        # Check if model is YOLO
        self._is_yolo = self._check_is_yolo(model)

    def _check_is_yolo(self, model: nn.Module) -> bool:
        """Check if the model is a YOLO model."""
        model_class = model.__class__.__name__
        if 'YOLO' in model_class or hasattr(model, 'model'):
            if hasattr(model, 'predict'):
                return True
        return False

    def _process_outputs(
        self,
        outputs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process detection outputs - not used directly."""
        # Detection outputs are handled differently
        return torch.tensor([]), None

    def _prepare_batch(
        self,
        batch: Tuple
    ) -> Tuple[Any, List[Dict[str, torch.Tensor]]]:
        """Prepare detection batch."""
        images, targets = batch

        if isinstance(images, torch.Tensor):
            images = images.to(self.device)
        else:
            images = [img.to(self.device) for img in images]

        # Move targets to device
        if isinstance(targets, list):
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        return images, targets

    def _collect_predictions(self) -> None:
        """Collect all detection predictions and targets."""
        self.all_predictions = []
        self.all_targets = []

        with torch.no_grad():
            for batch in self.test_loader:
                images, targets = self._prepare_batch(batch)

                if self._is_yolo:
                    # YOLO inference
                    results = self.model.model(images, conf=self.conf_threshold)
                    preds = self._parse_yolo_results(results)
                else:
                    # torchvision inference
                    self.model.eval()
                    preds = self.model(images)

                self.all_predictions.extend(preds)
                self.all_targets.extend(targets)

    def _parse_yolo_results(self, results) -> List[Dict]:
        """Parse YOLO results to standard format."""
        parsed = []
        for result in results:
            boxes = result.boxes
            pred_dict = {
                'boxes': boxes.xyxy.cpu(),
                'scores': boxes.conf.cpu(),
                'labels': boxes.cls.cpu().long()
            }
            parsed.append(pred_dict)
        return parsed

    def evaluate(self) -> Dict[str, float]:
        """
        Run detection evaluation.

        Returns:
            Dictionary with mAP metrics
        """
        self.logger.info("Running detection evaluation...")

        # Collect predictions
        self._collect_predictions()

        if not self.all_predictions or not self.all_targets:
            self.logger.warning("No predictions or targets found")
            return {}

        # Compute mAP metrics
        self.metrics = self._compute_map_metrics()

        self.logger.info("Evaluation complete")
        self.print_summary()

        return self.metrics

    def _compute_map_metrics(self) -> Dict[str, float]:
        """Compute mAP metrics."""
        metrics = {}

        # Compute AP at different IoU thresholds
        aps_per_iou = []

        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap_at_iou(iou_thresh)
            aps_per_iou.append(ap)

        # mAP@0.5:0.95
        metrics['mAP'] = np.mean(aps_per_iou) if aps_per_iou else 0.0

        # mAP@0.5
        metrics['mAP50'] = self._compute_ap_at_iou(0.5)

        # mAP@0.75
        metrics['mAP75'] = self._compute_ap_at_iou(0.75)

        # Compute precision and recall at IoU 0.5
        precision, recall = self._compute_precision_recall(0.5)
        metrics['precision_50'] = precision
        metrics['recall_50'] = recall

        return metrics

    def _compute_ap_at_iou(self, iou_threshold: float) -> float:
        """Compute Average Precision at a specific IoU threshold."""
        all_scores = []
        all_matches = []

        for pred, target in zip(self.all_predictions, self.all_targets):
            pred_boxes = pred.get('boxes', torch.tensor([]))
            pred_scores = pred.get('scores', torch.tensor([]))
            pred_labels = pred.get('labels', torch.tensor([]))

            target_boxes = target.get('boxes', torch.tensor([]))
            target_labels = target.get('labels', torch.tensor([]))

            if len(pred_boxes) == 0:
                continue

            # Match predictions to targets
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                all_scores.append(score.item())

                # Find matching target
                matched = False
                if len(target_boxes) > 0:
                    ious = self._compute_iou(box.unsqueeze(0), target_boxes)
                    max_iou, max_idx = ious.max(dim=1)

                    if max_iou.item() >= iou_threshold:
                        if target_labels[max_idx].item() == label.item():
                            matched = True

                all_matches.append(matched)

        if not all_scores:
            return 0.0

        # Sort by scores
        sorted_indices = np.argsort(all_scores)[::-1]
        all_matches = np.array(all_matches)[sorted_indices]

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(all_matches)
        fp_cumsum = np.cumsum(~all_matches)

        recalls = tp_cumsum / max(len(self.all_targets), 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            prec_at_recall = precisions[recalls >= r]
            if len(prec_at_recall) > 0:
                ap += prec_at_recall.max() / 11

        return ap

    def _compute_precision_recall(self, iou_threshold: float) -> Tuple[float, float]:
        """Compute precision and recall at a specific IoU threshold."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(self.all_predictions, self.all_targets):
            pred_boxes = pred.get('boxes', torch.tensor([]))
            target_boxes = target.get('boxes', torch.tensor([]))

            if len(target_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            if len(pred_boxes) == 0:
                total_fn += len(target_boxes)
                continue

            # Match predictions to targets
            matched_targets = set()

            for box in pred_boxes:
                ious = self._compute_iou(box.unsqueeze(0), target_boxes)
                max_iou, max_idx = ious.max(dim=1)

                if max_iou.item() >= iou_threshold and max_idx.item() not in matched_targets:
                    total_tp += 1
                    matched_targets.add(max_idx.item())
                else:
                    total_fp += 1

            total_fn += len(target_boxes) - len(matched_targets)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return precision, recall

    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        # boxes format: [x1, y1, x2, y2]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Intersection
        inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        union_area = area1.unsqueeze(1) + area2 - inter_area

        return inter_area / (union_area + 1e-6)

    def generate_visualizations(
        self,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Generate detection visualizations."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path("./evaluation_results")
            output_dir.mkdir(parents=True, exist_ok=True)

        self.visualization_paths = []

        # Save metrics summary
        metrics_path = output_dir / "detection_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Detection Evaluation Metrics\n")
            f.write("=" * 40 + "\n")
            for key, value in self.metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        self.visualization_paths.append(str(metrics_path))

        self.logger.info(f"Generated {len(self.visualization_paths)} visualizations")
        return self.visualization_paths
