# Data Dictionary

The Data is held on the OSF: https://osf.io/f2n4c/.

The scripts in the *Processing* folder produce a csv of stitched gaze and steering data. This csv is named something like: 'GazeAndSteering_longformat_XXXX.csv'. 

For analysis in R the original csv has underwent further grooming with the *Post-processing* script 'tidy_data_for_analysis.R'. This saves the files 'steergazedata_fulltrials' as a csv or a rds (R binary file, much quicker to deal with than csv if you are using R). You will want to use the steergazedata files for the majority of your analysis. 


### STEERGAZEDATA 

**_Columns not necessarily in order_**


1) **Unnamed**: Monotically increasing index. Does not start from zero as some data is cut when syncing gaze and steering time-ranges

2) **ID**: participant id used for testing

3) **Age**: Participant age

4) **Gender**: 99=not responded, 1=Female, 2=Male, 3=Other, 4=Prefer not to say. *In 100918 dataset 0=not responded*

5) **Vision**: Qu "Wearing contacts or glasses for testing". 99=not responded, 1:Yes, 2:No

6) **LicenseMonths**: Months holding driving license.

7) **frameidx**: Monotically increasing index from i->len(trial) within a specific trial. May not start at zero if frames have been cropped during processing.

8) **trackindex**: Closest point on the midline of the track to the vehicle's position

9) **currtime**: Experiment running time from experiment begin to experiment end.

10) **SWA**: Steering wheel angle.

11) **posx**: X-axis position of observer in world

12) **posz**: Z-axis position of observer in world

13) **yaw**: World orientation, from 0->360.

14) **yawrate**: Rate of change of yaw.

15) **autoflag**: Binary flag indicating whether automation was switched on. 1=Automated, 0=Manual.

16) **steeringbias**: Rightward Steering Bias. Signed distance from the closest point on the midline. Negative = towards the left side of the track. Positive = towards the right side of the track.

17) **sectionorder**: order of section, regardless of type, in overall sequence.

18) **drivingmode**: (called *sectiontype* in the GazeAndSteering csv). section index. 0 = Manual, 1 = Playback, 2 = Stock, 3 = Backup (only exists if errors are made during active manual trials), 4=PID control, 5=Interpolation period.

19) **obstaclecolour**: 0=Blue, attract, 1=Red, avoid.

20) **obstacleoffset**: deflection of targets from midline.

21) **vangle**: Vertical angular declination of gaze relative to the horizon. Negative = Below Horizon. Positive = Above Horizon.

22) **hangle**: Horizontal angular difference relative to the middle of the screen. Negative = Leftwards, Positive = Rightwards.

23) **lookahead**: Pythagorean distance, in metres, from observer to gaze landing point.

24) **gazebias**: Same as steeringbias above but using the gaze landing point.

25) **xpog**: X-axis position in world of gaze

26) **zpog**: Z-axis position in world of gaze

27) **trialcode**: Unique string identifying trial. EXPBLOCK_PP_drivingmode_condition_count

28) **count**: trialnumber within each condition, from 0->(Ntrials-1). +100 for active trials that are not replaying. *For 100918 piloting datasets, Only one trial per condition was used*

29) **condition**: condition index, from 0->(NCndts-1). *For 100918 piloting datasets, 0 = Attract_Narrow, 1=Attract_Medium, 2=Attract_Wide, 3=Avoid_Narrow, 4=Avoid_Medium, 5=Avoid_Wide*

30) **block**: Experimental Block (before or after midway break). 1 or 2.

31) **sample_class**: Corresponding eye-movement class parsed by nlmm parser.

*the following trials are added with the script 'tidy_data_for_analysis.R'*

32) **currtimezero**: Time of trial starting from zero at the start of the trial.

33) **f**: frame index starting from 1 for each trial.

34) **startingposition**: 0 or 1 depending on whether the trial starts on the opposite side of the oval track.

35) **posx_mirror**: x vehicle position has been mirrored so that all trials start from the same point

36) **posx_mirror**: z vehicle position has been mirrored so that all trials start from the same point

37) **xpog_mirror**: x gaze position has been mirrored so that all trials start from the same point

38) **zpog_mirror**: z gaze position has been mirrored so that all trials start from the same point

39) **yaw_mirror**: yaw has been mirrored so that all trials start from the same point

40) **roadsection**: categorised as belonging to Straight (=0), Bend (=1), or the slalom (=2). The cutoff is 1 s before the geometric change. So the bend starts at Z = 60 but the roadsection categorising the bend starting at Z = 52 m.

41) **on_road**: flag as to whether absolute gaze bias was <1.5m or >1.5m (landing on the road or not).

42) **trial_match**: code that is a concatenation of ID, condition, block, count. Can be used to pick matched manual and replay trials as they will have the same code. NA for Stock trials.


### Segmentation Data csv

**_Columns in order_**

1) **Unnamed**: Zero-based index. Each row is one segment.

2) **seg_class**: Segmented class. 1 = Fixations. 2= Saccades, 3= Post-Saccadic Oscillations, 4= Smooth Pursuit.

the following variables are the same as Gaze and Steering dataframe.\
3. **pp_id** or **ID**\
4. **trialcode**\
5. **condition**\
6. **count**\
7. **sectiontype**\
8. **sectionorder**\
9. **block**

the following variables correspond to segments start and end data of each segment.\
10. **t1**: Start time\
11. **t2**: End time\
12. **v1**: Start vertical angle\
13. **v2**: End vertical angle\
14. **h1**: Start horizontal angle\
15. **h2**: End horizontal angle

16) **yawrate**: Avg yawrate for segment. Useful for comparing against horizonal velocity (see Itkonen et al., 2015).


### Target Positions csv

**_Columns in order_**

1) **Unnamed**: Zero-based index.

2) **targetindex**: Identifier for specific target, from 0->5. There are three on each straight.

3) **condition**: corresponding condition index

4) **xcentre**: world x-axis position of target centre

5) **zcentre**: world z-axis position of target centre

6) **centre_radius**: radius of centre grey circle

7) **target_radius**: radius of entire coloured target.

