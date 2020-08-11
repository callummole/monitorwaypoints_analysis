import numpy as np

def TrackMaker(sectionsize):
	
	"""adds oval track with double straight. Returns 4 variables: midline, origin, section breaks, track details"""
	#at the moment each straight or bend is a separate section. So we can alter the colour if needed. But more efficient to just create one line per edge.
#    code copied from vizard trackmaker.

	"""
       ________
	 /   _B__   \ 
	/	/    \   \ 
	| A |    | C |
   _|   |    |   |_
   _| H |    | D |_
	|	|	 |   |
	| G	|	 | E |
	\	\___ /   /
     \ ____F___ /


	A = Empty Straight 
	B = Constant curvature Bend
	C = Straight with Targets.
	D = Interp period (Length = StraightLength / 4.0)
	E = Empty Straight
	F = Constant curvature bend
	G = Straight with Targets
	H = Interp period (Length = StraightLength / 4.0)

	TrackOrigin, centre of track = 0,0. Will be half-way in the interp period.

	"""



	#Start at beginning of 1st straight.
	StraightLength = 40.0 #in metres. 
	InterpProportion = 1.0 #Length of interpolation section relative to the straight sections
	InterpLength = StraightLength * InterpProportion
	InterpHalf = InterpLength / 2.0
	BendRadius = 25.0 #in metres, constant curvature bend.
	SectionSize = sectionsize
	roadwidth = 3.0/2.0
	right_array = np.linspace(np.pi, 0.0, SectionSize) 
	left_array= np.linspace(0.0, np.pi,SectionSize)
	
	
	#trackorigin = [BendRadius, StraightLength/2.0] #origin of track for bias calculation
	trackorigin = [0.0, 0.0]
	trackparams = [BendRadius, StraightLength, InterpLength, SectionSize, InterpProportion]

	#For readability set key course markers. Use diagram for reference
	LeftStraight_x = -BendRadius
	RightStraight_x = BendRadius
	Top_Interp_z = InterpHalf
	Top_Straight_z = InterpHalf+StraightLength
	Bottom_Interp_z = -InterpHalf
	Bottom_Straight_z = -InterpHalf-StraightLength
	
	###create unbroken midline. 1000 points in each section.	
	#at the moment this is a line so I can see the effects. But this should eventually be an invisible array.
	#straight	
	#The interp periods have index numbers of sectionsize / 4. So midline size = SectionSize * 7 (6 sections + two interps)
	midlineSize = SectionSize* (6 + 2 * InterpProportion)
	midline = np.zeros((int(midlineSize),2))

	SectionBreaks = []
	SectionBreaks.append(0)
	SectionBreaks.append(int(SectionSize)) #end of StraightA #1
	SectionBreaks.append(int(SectionSize*2)) #end of BendB #2
	SectionBreaks.append(int(SectionSize*3)) #end of StraightC #3
	SectionBreaks.append(int(SectionSize* (3 + InterpProportion))) #end of InterpD #4
	SectionBreaks.append(int(SectionSize*(4 + InterpProportion))) #end of StraightE #5
	SectionBreaks.append(int(SectionSize*(5 + InterpProportion))) #end of BendF #6
	SectionBreaks.append(int(SectionSize*(6 + InterpProportion))) #end of StraightG #7
	SectionBreaks.append(int(SectionSize*(6 + 2*InterpProportion))) #end of InterpH #8

	#Straight A
	StraightA_z = np.linspace(Top_Interp_z, Top_Straight_z, SectionSize)
	midline[SectionBreaks[0]:SectionBreaks[1],0] = LeftStraight_x
	midline[SectionBreaks[0]:SectionBreaks[1],1] = StraightA_z

	#print (SectionBreaks)
	#print (midline[SectionBreaks[0]:SectionBreaks[1],:])
		
	#Bend B
	i=0
	while i < SectionSize:
		x = (BendRadius*np.cos(right_array[i])) #+ BendRadius 
		z = (BendRadius*np.sin(right_array[i])) + (Top_Straight_z)
		midline[i+SectionBreaks[1],0] = x
		midline[i+SectionBreaks[1],1] = z
		#viz.vertex(x,.1,z)
		#viz.vertexcolor(viz.WHITE)
		xend = x
		i += 1
	
	#StraightC
	rev_straight = StraightA_z[::-1] #reverse
	midline[SectionBreaks[2]:SectionBreaks[3],0] = xend
	midline[SectionBreaks[2]:SectionBreaks[3],1] = rev_straight
	
#		
 	#InterpD
	InterpD_z = np.linspace(Top_Interp_z, Bottom_Interp_z, int(SectionSize*InterpProportion))
	midline[SectionBreaks[3]:SectionBreaks[4],0] = xend
	midline[SectionBreaks[3]:SectionBreaks[4],1] = InterpD_z

	#StraightE
	StraightE_z = np.linspace(Bottom_Interp_z, Bottom_Straight_z, SectionSize)
	midline[SectionBreaks[4]:SectionBreaks[5],0] = xend
	midline[SectionBreaks[4]:SectionBreaks[5],1] = StraightE_z

	#BendF
	i=0
	while i < SectionSize:
		x = (BendRadius*np.cos(left_array[i]))
		z = -(BendRadius*np.sin(left_array[i])) + (Bottom_Straight_z)
		midline[i+(SectionBreaks[5]),0] = x
		midline[i+(SectionBreaks[5]),1] = z
	#	viz.vertex(x,.1,z)
	#	viz.vertexcolor(viz.WHITE)
		xend = x
		i += 1
	
	#StraightG
	StraightG_z = np.linspace(Bottom_Straight_z, Bottom_Interp_z, SectionSize)
	midline[SectionBreaks[6]:SectionBreaks[7],0] = xend
	midline[SectionBreaks[6]:SectionBreaks[7],1] = StraightG_z

	#InterpG
	InterpG_z = np.linspace(Bottom_Interp_z, Top_Interp_z, int(SectionSize*InterpProportion))
	midline[SectionBreaks[7]:SectionBreaks[8],0] = xend
	midline[SectionBreaks[7]:SectionBreaks[8],1] = InterpG_z

	TrackData = []
	TrackData.append(midline)
	TrackData.append(trackorigin)
	TrackData.append(SectionBreaks)
	TrackData.append(trackparams)

	return TrackData          
