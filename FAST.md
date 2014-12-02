### Определение углов перехода, используя FAST алгоритм ###
    C++: void FAST(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression=true )
    C++: void FASTX(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression, int type)
**Аргументы функциий:**
* **image** - черно-белое изображение (в виде массива пикселей).
* **keypoints** - в вектор будут занесены позиции ключевых точек (углов).
* **threshold** - порог, задающий разницу между интенсивностью центрального пикселя и окружающих его пикселей.
* **nonmaxSuppression** - если этот параметр установлен в true, то будет применяться [_подавления немаксимумов_](https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80_%D0%9A%D1%8D%D0%BD%D0%BD%D0%B8)
* **type** - может принимать одно из трех значений: *FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12, FastFeatureDetector::TYPE_5_8.

***

#### MSER ####

```c++
class MSER : public CvMSERParams
{
public:
    // default constructor
    MSER();
    // constructor that initializes all the algorithm parameters
    MSER( int _delta, int _min_area, int _max_area,
          float _max_variation, float _min_diversity,
          int _max_evolution, double _area_threshold,
          double _min_margin, int _edge_blur_size );
    // runs the extractor on the specified image; returns the MSERs,
    // each encoded as a contour (vector<Point>, see findContours)
    // the optional mask marks the area where MSERs are searched for
    void operator()( const Mat& image, vector<vector<Point> >& msers, const Mat& mask ) const;
};
```
```python
import numpy as np
import cv2
import video

if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = video.create_capture(video_src)
    mser = cv2.MSER()
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detect(gray, None)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))

        cv2.imshow('img', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
```