## Real-time Finger-Detection using Neural Networks (SSD) on Tensorflow.

Real Time Hand Detection using SSD and processing the detected hand to identify the fingers in the hand.

For Hand Detection in real time, I am using the published work by Victor Dibia, where they have used a Single Shot Multi-Box Detector (SSD), and I am using their pretrained model on MobileNet (v1).

For Finger Detection, I am using image processing techniques like Gaussian Blur as well as finding Convex Hull and Convexity Defects to find points of finger tips and points betwwen the fingers.

### Citation

Victor Dibia, Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow, (2017), GitHub repository, https://github.com/victordibia/handtracking
```bib
@misc{Dibia2017,
  author = {Victor, Dibia},
  title = {Real-time Hand Tracking Using SSD on Tensorflow },
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/victordibia/handtracking}},
  commit = {b523a27393ea1ee34f31451fad656849915c8f42}
}
```

