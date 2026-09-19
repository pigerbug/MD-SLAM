# ManhattanSLAM with YOLOv8 TensorRT

ManhattanSLAM is a real-time RGB-D SLAM system built on ORB-SLAM2 and extended with planar constraints, dense surfel reconstruction, and dynamic-object filtering using a YOLOv8 TensorRT detector.

This repository keeps the original ManhattanSLAM mapping and tracking pipeline, while replacing the legacy YOLOv5 TensorRT integration with a modern YOLOv8 inference path. The detector is used to generate dynamic object masks and suppress moving regions during tracking and mapping.

## Features

- RGB-D SLAM with ORB features, line features, plane features, and Manhattan-frame constraints
- Dense surfel-based 3D reconstruction
- Dynamic object detection using YOLOv8
- TensorRT deployment for low-latency inference
- Support for dynamic region masking in tracking
- Example configs for TUM, ICL, and TAMU datasets

## Related Publication

Raza Yunus, Yanyan Li, and Federico Tombari. "ManhattanSLAM: Robust Planar Tracking and Mapping Leveraging Mixture of Manhattan Frames." ICRA 2021.

Paper: https://arxiv.org/pdf/2103.15068.pdf

## License

This project is released under the GPLv3 license.

See:
- LICENSE.txt
- License-gpl.txt
- Dependencies.md

If you use this project in academic work, please cite the original paper listed above.

## Prerequisites

The project was originally developed and tested under Ubuntu 16.04 / 20.04 environments. A typical modern build environment is:

- Ubuntu 18.04 or 20.04
- CUDA 11.x
- TensorRT 8.x
- OpenCV 3.x or 4.x
- Eigen3
- PCL 1.7+ / 1.10+
- Pangolin
- tinyply
- CMake
- GCC / g++ with C++14 support

The exact include and link paths in the top-level CMakeLists.txt may need to be adjusted to your local CUDA/TensorRT installation.

## Repository Layout

- `include/` : core SLAM headers
- `src/` : SLAM implementations
- `Thirdparty/` : DBoW2, g2o and external dependencies
- `Example/` : dataset config and example runner
- `Vocabulary/` : ORB vocabulary files
- `model/` : expected YOLOv8 engine and class list
- `build.sh` : project build script

## Build

Clone the project:

```bash
git clone <your-repo-url>
cd MD-SLAM
```

Build the dependencies and project:

```bash
chmod +x build.sh
./build.sh
```

If your environment does not use the default CUDA/TensorRT locations, edit the relevant include and link directories in `CMakeLists.txt` before building.

## YOLOv8 TensorRT Setup

The detector now expects a YOLOv8 TensorRT engine named:

```text
model/yolov8x.engine
```

and a class list file:

```text
model/coco.names
```

### Export a YOLOv8 engine

Use the same TensorRT version as the project build environment:

```bash
pip install ultralytics

yolo export model=yolov8x.pt format=engine imgsz=640 half=True device=0
mkdir -p model
cp yolov8x.engine model/yolov8x.engine
cp coco.names model/coco.names
```

Important:
- Old YOLOv5 engine files are not compatible with the current detector.
- The engine must be generated from a YOLOv8 model.
- If you use a custom class set, make sure the class ordering matches the training labels.

A more detailed note is available in `YOLOv8_TensorRT.md`.

## Run the SLAM System

Prepare a dataset sequence and a corresponding association file.

For TUM RGB-D:

```bash
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```

Then run:

```bash
./Example/manhattan_slam Vocabulary/ORBvoc.txt Example/TUM1.yaml PATH_TO_SEQUENCE_FOLDER associations.txt
```

Replace the config file according to the dataset:

- `Example/TUM1.yaml`
- `Example/TUM2.yaml`
- `Example/TUM3.yaml`
- `Example/ICL.yaml`
- `Example/TAMU.yaml`

## Notes

- The YOLOv8 TensorRT path expects a COCO-style output layout with bounding box attributes in the first part of the detection vector.
- Dynamic detection is mainly used for person and other moving objects to reduce tracking drift.
- If you run on a different CUDA/TensorRT installation, update the include and library paths at the top of `CMakeLists.txt`.

## Troubleshooting

Common issues:

- `TensorRT not found` : verify the TensorRT include and lib directories in `CMakeLists.txt`
- `engine not found` : ensure `model/yolov8x.engine` exists
- `class name mismatch` : rebuild or update `model/coco.names` to match the model
- `CUDA runtime error` : check the GPU driver and TensorRT/CUDA compatibility

## Citation

```bibtex
@inproceedings{yunus2021manhattanslam,
    author = {R. Yunus and Y. Li and F. Tombari},
    title = {ManhattanSLAM: Robust Planar Tracking and Mapping Leveraging Mixture of Manhattan Frames},
    year = {2021},
    booktitle = {2021 IEEE International Conference on Robotics and Automation (ICRA)}
}
```
