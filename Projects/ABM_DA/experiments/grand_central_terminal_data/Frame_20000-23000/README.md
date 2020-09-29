# Trajectory reconstruction process
A interval of the Grand Central Terminal (New York City) data was selected and processed.

## Video interval
From frame 20000 to frame 23000

## Perspective correction
<a href="https://www.codecogs.com/eqnedit.php?latex=(x,&space;y)&space;=&space;(x',&space;y')[1&space;&plus;&space;\delta]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(x,&space;y)&space;=&space;(x',&space;y')[1&space;&plus;&space;\delta]" title="(x, y) = (x', y')[1 + \delta]" /></a>

where:

* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\delta=y'/h'" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\delta=y'/h'" title="\delta=y'/h'" /></a>;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(x',&space;y')" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;(x',&space;y')" title="(x', y')" /></a>: pedestrian (horizontal, vertical) position before transformation;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(x,&space;y)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;(x,&space;y)" title="(x, y)" /></a>: pedestrian (horizontal, vertical) position after transformation;
* <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h'" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;h'" title="h'" /></a>: height of the unprocessed image;

### Folder: GCT_final_real_data
* Trajectories organized by frames;
* Activation file with informations about each pedestrian (pedestrian ID, activation time, entance gate, exit gate);

### Folder: GCT_pedestrian_data
* Trajectories organized by pedestrians;
* Time needs to be shifted by -19000 frames.
