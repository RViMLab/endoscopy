# Endoscopy
Utilities for processing of endoscopic images.

## Installation
To install, simply run
```shell
pip install endoscopy
```

## Example
To run an example on a single image, try
```shell
python img_main.py
```

## Credits
If you use this work as part of your project, please consider citing our [paper](https://arxiv.org/abs/2109.15098)
```bibtex
@article{huber2021deep,
  title={Deep Homography Estimation in Dynamic Surgical Scenes for Laparoscopic Camera Motion Extraction},
  author={Huber, Martin and Ourselin, S{\'e}bastien and Bergeles, Christos and Vercauteren, Tom},
  journal={arXiv preprint arXiv:2109.15098},
  year={2021}
}
```

## Notes
 - SVD doesnt work for incomplete data
 - Simple least squares doesn't work if not at center and angle not zero https://jekel.me/2020/Least-Squares-Ellipsoid-Fit/
 - Linearize?
