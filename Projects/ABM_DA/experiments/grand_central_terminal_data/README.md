# Grand Central Terminal (New York City) data

The data used here have been inferred from a video sequence recorded at GCT. 

## Original video 
The [original video](http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html) is 33:20-minutes  long  at  25fps  with  a  resolution  of 720×480.


## Trajectories
The  trajectories  were determined by a [Kanade-Lucas-Tomasi (KLT) keypoint tracker](http://cygnus-x1.cs.duke.edu/courses/spring06/cps296.1/handouts/lucas_kanade.pdf).

## Perspective correction
<a href="https://www.codecogs.com/eqnedit.php?latex=(x,&space;y)&space;=&space;(x',&space;y')[1&space;&plus;&space;\delta]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x,&space;y)&space;=&space;(x',&space;y')[1&space;&plus;&space;\delta]" title="(x, y) = (x', y')[1 + \delta]" /></a>

where:

* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\delta=y'/h'" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\delta=y'/h'" title="\delta=y'/h'" /></a>;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(x',&space;y')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;(x',&space;y')" title="(x', y')" /></a>: pedestrian (horizontal, vertical) position before transformation;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(x,&space;y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;(x,&space;y)" title="(x, y)" /></a>: pedestrian (horizontal, vertical) position after transformation;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h'" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;h'" title="h'" /></a>: height of the unprocessed image;


## Trajectory reconstruction process
A interval of the Grand Central Terminal (New York City) data was selected and processed.

### Video interval
From frame 20000 to frame 23000

### Folder: [GCT_final_real_data](./GCT_final_real_data)
* Trajectories organized by frames;
* Activation file with information about each pedestrian (pedestrian ID, activation time, entance gate, exit gate);

## Notebooks

* [GCT-data:](GCT-data.ipynb) Notebook with a data overview;
* [Real_Data_correct_trails:](Real_Data_correct_trails.ipynb) Notebook with the trajectories reconstruction process;
