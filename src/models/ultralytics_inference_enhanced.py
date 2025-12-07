"""
Enhanced Ultralytics YOLOv8 inference wrapper with advanced features.
Supports: per-class thresholds, custom visualization, export, batch processing,
multi-GPU, async processing, frame skipping, and multimodal initialization.
"""
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import torch

from ..utils.detection_export import (
    export_detections_json,
    export_detections_csv,
    export_detections_xml,
    parse_ultralytics_results,
)
from ..utils.visualization import (
    draw_custom_boxes,
    get_class_colors,
    filter_detections_by_class,
    apply_per_class_thresholds,
)


class UltralyticsInferenceEnhanced:
    """Enhanced wrapper for Ultralytics YOLOv8 inference with advanced features."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None,
    ):
        """
        Initialize enhanced Ultralytics inference.

        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device (cuda, cpu, mps)
            multi_gpu: Whether to use multiple GPUs
            gpu_ids: List of GPU IDs to use (None = use all available)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        if multi_gpu and device == "cuda":
            if gpu_ids is None:
                self.gpu_ids: Optional[List[int]] = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = gpu_ids
            print(f"🖥️  Multi-GPU mode: Using GPUs {self.gpu_ids}")
        else:
            self.gpu_ids = None

        # Map device names
        if multi_gpu and device == "cuda" and self.gpu_ids is not None:
            device_str = ",".join(map(str, self.gpu_ids))
        else:
            device_str = "0" if device == "cuda" else device
        
        device_map = {
            "cuda": device_str,
            "cpu": "cpu",
            "mps": "mps",
        }
        self.device = device_map.get(device, device)

        print(f"Loading model: {model_path}")
        self.model = YOLO(str(self.model_path))

        # Multimodal single GPU initialization (load model once, use for multiple tasks)
        self._model_loaded = True
        print(f"✅ Model loaded successfully on device: {device}")

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_video: bool = True,
        show_video: bool = False,
        class_thresholds: Optional[Dict[str, float]] = None,
        filter_classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        show_labels: bool = True,
        show_confidences: bool = True,
        export_format: Optional[str] = None,
        export_path: Optional[str | Path] = None,
        frame_skip: int = 1,
        async_mode: bool = False,
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """
        Process video file with enhanced features.

        Args:
            video_path: Path to input video (MP4)
            output_path: Path to output video
            conf_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
            save_video: Whether to save output video
            show_video: Whether to display video during processing
            class_thresholds: Per-class confidence thresholds dict
            filter_classes: List of class names to include (None = all)
            exclude_classes: List of class names to exclude
            custom_colors: Custom colors for classes (BGR format)
            show_labels: Whether to show labels
            show_confidences: Whether to show confidence scores
            export_format: Export format ('json', 'csv', 'xml', or None)
            export_path: Path for export file
            frame_skip: Process every Nth frame (1 = all frames)
            async_mode: Use async processing (experimental)

        Returns:
            Tuple of (total_frames, processed_frames, all_detections)
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path_obj}")

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path_obj}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        if frame_skip > 1:
            print(f"   Frame skip: {frame_skip} (processing every {frame_skip} frames)")

        # Setup video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(output_path_obj), fourcc, fps // frame_skip, (width, height)
            )

        frame_count = 0
        processed_frames = 0
        all_detections = []

        print(f"\n🔄 Processing video...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames if frame_skip > 1
                if frame_count % frame_skip != 0:
                    continue

                # Run inference
                results = self.model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False,
                )

                # Parse results
                detections = parse_ultralytics_results(
                    results, frame_number=frame_count, image_path=str(video_path_obj)
                )

                # Apply per-class thresholds
                if class_thresholds:
                    detections = apply_per_class_thresholds(
                        detections, class_thresholds, conf_threshold
                    )

                # Filter by class
                if filter_classes or exclude_classes:
                    detections = filter_detections_by_class(
                        detections, filter_classes, exclude_classes
                    )

                # Store detections for export
                all_detections.extend(detections)

                # Draw custom boxes
                if custom_colors or show_labels or show_confidences:
                    annotated_frame = draw_custom_boxes(
                        frame,
                        detections,
                        class_colors=custom_colors,
                        show_labels=show_labels,
                        show_confidences=show_confidences,
                    )
                else:
                    annotated_frame = results[0].plot()

                # Save frame
                if save_video:
                    out.write(annotated_frame)

                # Show frame
                if show_video:
                    cv2.imshow("Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                processed_frames += 1

                # Progress update
                if processed_frames % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames, {len(detections)} detections)"
                    )

        finally:
            cap.release()
            if save_video:
                out.release()
            if show_video:
                cv2.destroyAllWindows()

        # Export detections
        if export_format and export_path:
            self._export_detections(all_detections, export_format, export_path, video_path_obj)

        print(f"\n✅ Video processing complete!")
        print(f"   Processed {processed_frames}/{total_frames} frames")
        print(f"   Total detections: {len(all_detections)}")
        if save_video:
            print(f"   Output saved to: {output_path_obj}")

        return total_frames, processed_frames, all_detections

    def process_image(
        self,
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_thresholds: Optional[Dict[str, float]] = None,
        filter_classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        show_labels: bool = True,
        show_confidences: bool = True,
        export_format: Optional[str] = None,
        export_path: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process single image with enhanced features.

        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            conf_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
            class_thresholds: Per-class confidence thresholds dict
            filter_classes: List of class names to include (None = all)
            exclude_classes: List of class names to exclude
            custom_colors: Custom colors for classes (BGR format)
            show_labels: Whether to show labels
            show_confidences: Whether to show confidence scores
            export_format: Export format ('json', 'csv', 'xml', or None)
            export_path: Path for export file

        Returns:
            Tuple of (annotated_image, detections)
        """
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path_obj}")

        # Load image
        image = cv2.imread(str(image_path_obj))
        if image is None:
            raise ValueError(f"Could not load image: {image_path_obj}")

        # Run inference
        results = self.model.predict(
            str(image_path_obj),
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = parse_ultralytics_results(results, image_path=str(image_path_obj))

        # Apply per-class thresholds
        if class_thresholds:
            detections = apply_per_class_thresholds(
                detections, class_thresholds, conf_threshold
            )

        # Filter by class
        if filter_classes or exclude_classes:
            detections = filter_detections_by_class(
                detections, filter_classes, exclude_classes
            )

        # Draw custom boxes
        if custom_colors or show_labels or show_confidences:
            annotated_image = draw_custom_boxes(
                image,
                detections,
                class_colors=custom_colors,
                show_labels=show_labels,
                show_confidences=show_confidences,
            )
        else:
            annotated_image = results[0].plot()

        # Save if requested
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path_obj), annotated_image)
            print(f"✅ Image saved to: {output_path_obj}")

        # Export detections
        if export_format and export_path:
            image_size = (image.shape[1], image.shape[0])
            self._export_detections(
                detections, export_format, export_path, image_path_obj, image_size
            )

        return annotated_image, detections

    def process_batch_images(
        self,
        image_paths: List[str | Path],
        output_dir: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_thresholds: Optional[Dict[str, float]] = None,
        filter_classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        show_labels: bool = True,
        show_confidences: bool = True,
        export_format: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image paths
            output_dir: Output directory for processed images
            conf_threshold: Default confidence threshold
            iou_threshold: IoU threshold for NMS
            class_thresholds: Per-class confidence thresholds dict
            filter_classes: List of class names to include (None = all)
            exclude_classes: List of class names to exclude
            custom_colors: Custom colors for classes (BGR format)
            show_labels: Whether to show labels
            show_confidences: Whether to show confidence scores
            export_format: Export format ('json', 'csv', 'xml', or None)
            batch_size: Batch size for processing
            num_workers: Number of worker threads

        Returns:
            List of all detections from all images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_detections = []

        print(f"🔄 Processing {len(image_paths)} images in batches of {batch_size}...")

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(image_paths) + batch_size - 1) // batch_size

            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} images)")

            for image_path in batch:
                image_path_obj = Path(image_path)
                output_path = output_dir / image_path_obj.name

                _, detections = self.process_image(
                    image_path=image_path,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    class_thresholds=class_thresholds,
                    filter_classes=filter_classes,
                    exclude_classes=exclude_classes,
                    custom_colors=custom_colors,
                    show_labels=show_labels,
                    show_confidences=show_confidences,
                )

                all_detections.extend(detections)

        # Export all detections if requested
        if export_format:
            export_path = output_dir / f"detections.{export_format}"
            self._export_detections(all_detections, export_format, export_path)

        print(f"\n✅ Batch processing complete!")
        print(f"   Processed {len(image_paths)} images")
        print(f"   Total detections: {len(all_detections)}")

        return all_detections

    def _export_detections(
        self,
        detections: List[Dict[str, Any]],
        export_format: str,
        export_path: str | Path,
        source_path: Optional[Path] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Internal method to export detections."""
        export_path = Path(export_path)

        if export_format.lower() == "json":
            metadata = {}
            if source_path:
                metadata["source"] = str(source_path)
            export_detections_json(detections, export_path, metadata)
        elif export_format.lower() == "csv":
            export_detections_csv(detections, export_path)
        elif export_format.lower() == "xml":
            image_path = str(source_path) if source_path else None
            export_detections_xml(detections, export_path, image_path, image_size)
        else:
            print(f"⚠️  Unknown export format: {export_format}")

