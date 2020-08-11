
There is now a time-synced json file (zmqlog) stored on the Venlab comp. The intended workflow is as follows:


1. Create pldata from zmqlog. Here you could recalibrate.
2. Run `extract_markers_pldata.py` to correct for the distortion in the camera and correct gaze. This will save 
`gaze_corrected.pldata` and `gaze_corrected_timestamps.npy` files. Few things to note here:
   * Use the correct undistortion. 
   * ...
3. Run `offline_surface_gaze-master/main.py` to extract the gaze csvs for post-processing. This will output `gaze_on_surface_Corrected.csv`.
4. Run `analyse_gazeonsrf_withSteering.py` to stitch gaze and steering together and project gaze through into world. The _Data Dictionary_ can be found in the folder _Data_. Few things to note:
   * Attention needed for the `StitchGazeAndSteering` function since the timestamps will already be synced. 
   * Could alter the `SurfaceToGazeAngle` function to use the camera projection rather than hard-coded horizon values to combat for any mis-measurement. 
