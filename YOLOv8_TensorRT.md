# YOLOv8 TensorRT

The detector loads `model/yolov8x.engine` and expects a YOLOv8 detection output
with shape `[1, 84, N]` or `[1, N, 84]` for a COCO model. The first four values
are `cx, cy, width, height`; the remaining values are class scores.

Export the engine with the same TensorRT version used to build this project:

```bash
pip install ultralytics
yolo export model=yolov8x.pt format=engine imgsz=640 half=True device=0
mkdir -p model
cp yolov8x.engine model/yolov8x.engine
cp coco.names model/coco.names
```

The old `yolov5x.engine` cannot be reused. YOLOv8 does not use the YOLOv5
anchor/plugin output format, so the engine must be regenerated from a YOLOv8
checkpoint. For a custom model, replace `yolov8x.pt` and ensure `coco.names`
contains the matching class order.