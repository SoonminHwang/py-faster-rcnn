import np
import os

def calculate_mAP(clsNm, pth='.'):

	recalls = []
	precisions = {'Easy':[], 'Moderate':[], 'Hard':[]}

	fName = os.path.join(pth, '{:s}_detection.txt'.format(clsNm))
	
	with open(fName, 'r') as fp:
		lines = fp.readlines()

		assert len(lines[0]) == 4, '# of columns should be equal to 4.'

		for line in lines:
			recalls.append( line[0] )
			precisions['Easy'].append( lines[1] )
			precisions['Moderate'].append( lines[2] )
			precisions['Hard'].append( lines[3] )

	mAP = {}
	mAP[clsNm] = {'Easy': np.mean(precisions['Easy'][::4]), 
				  'Moderate': np.mean(precisions['Moderate'][::4]), 
				  'Hard': np.mean(precisions['Hard'][::4])}

	return recalls, mAP


