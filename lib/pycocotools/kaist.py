__author__ = 'soonmin'
__version__ = '0.1'

# Interface for accessing the KAIST dataset.
#
# The KAIST class inherited from COCO class

from coco import COCO
import itertools

class KAIST(COCO):
	def __init__(self, annotation_file=None):
		"""
        Constructor of KAIST helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
		COCO.__init__(self, annotation_file)


	# In MATLAB API (piotr's toolbox),
	# 	pLoad options: xRng, yRng, wRng, hRng, occLevel, labels, ignore lables, squarify
	def getAnnIds(self, imgIds=[], catIds=[], xRng=[], yRng=[], wRng=[], hRng=[], occLevel=[]):
		"""
		Get ann ids that satisfy given filter conditions. default skips that filter
		:param imgIds  (int array)     	: get anns for given imgs
		:param catIds  (int array)     	: get anns for given cats		
		:param xRng (float array)   	: get anns for given x range (e.g. [0 inf])
		:param yRng (float array)   	: get anns for given y range (e.g. [0 inf])
		:param wRng (float array)   	: get anns for given w range (e.g. [0 inf])
		:param hRng (float array)   	: get anns for given h range (e.g. [0 inf])
		:param occLevel (int array)    	: get anns for given occ level		
		:return: ids (int array)       	: integer array of ann ids
		"""
		imgIds = imgIds if type(imgIds) == list else [imgIds]
		catIds = catIds if type(catIds) == list else [catIds]

		if len(imgIds) == len(catIds) == len(xRng) == len(yRng) == len(wRng) == len(hRng) == len(occLevel) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(imgIds) == 0:
				lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.dataset['annotations']

			anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]			
			anns = anns if len(xRng) == 0 else [ann for ann in anns if ann['bbox'][0] >= xRng[0] and ann['bbox'][0] <= xRng[1]]
			anns = anns if len(yRng) == 0 else [ann for ann in anns if ann['bbox'][1] >= yRng[0] and ann['bbox'][1] <= yRng[1]]
			anns = anns if len(wRng) == 0 else [ann for ann in anns if ann['bbox'][2] >= wRng[0] and ann['bbox'][2] <= wRng[1]]
			anns = anns if len(hRng) == 0 else [ann for ann in anns if ann['bbox'][3] >= hRng[0] and ann['bbox'][3] <= hRng[1]]
			anns = anns if len(occLevel) == 0 else [ann for ann in anns if ann['occ'] in occLevel]						

		ids = [ann['id'] for ann in anns]

		return ids

	def description(self):
		"""
		Print format description about the annotation file.
		:return:
		"""
		for key, value in self.dataset['description'].items():
			if key == 'occ':
				print '%s[%d]: %s'%(key, value['id'], value['desc'])
			else:
				print '%s: %s'%(key, value)

	def showAnns(self, anns):
		"""
		Display the specified annotations.
		:param anns (array of object): annotations to display
		:return: None
		"""
		if len(anns) == 0: 
			return 0

        #ax = plt.gca()
        #    ax.set_autoscale_on(False)