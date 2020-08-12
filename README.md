Analysis repo for preprint: Mole et al. (2020), _Drivers use active gaze to monitor waypoints during automated driving_

This repo contains all the files needed to process the raw data, hosted on OSF (https://osf.io/f2n4c/), through to the figures in the manuscript.

You could begin from the raw video files using main_pipeline.py. The code is very inefficient so the processing is incredibly slow. Or you could begin from the processed video files (_ProcessedData.zip_ on OSF), just extract them to the Data folder.

Key files are listed below.

### Processing

Python scripts for processing pupil-labs videos. 

To run, install anaconda and create an environment with the spec-file.txt
e.g. conda create --name pupil --file spec-file.txt

- main_pipeline.py (and imported files. In this script you can choose which analysis stage to begin at, instructions as comments).
- plot_perspective_view.py (annotated track; appendix fig)
- plot_gaze_density_perspective.py (density overlaid on screenshots).
- gaze_through_midline_densities.py (heatmap track)
- linmix.py (fitting example)
- linmix_inference.py (post-cluster individual means).


### Post-Processing

Folder contains R scripts for aggregating frame by frame data into bespoke measures, visualisation, and statistical analyis.

- manuscript_overall_th_diffs.rmd (pre-cluster inferences)
- manuscript_figs_postcluster.rmd (post-cluster means and contrasts) 
- manuscript_sawtooth_analysis.rmd (waypoint inferences and analysis).
