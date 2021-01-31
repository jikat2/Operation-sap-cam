from sentinelhub import SHConfig
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest
import cv2
import sys

def plot_image(image, factor, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('plot2.png')

config = SHConfig()
config.sh_client_id = 'dd0d9906-9ccb-470a-b80e-1ff099f833cc'
config.sh_client_secret = 'CZ!T.hY-PFj272{v{)J}~9%BK*d2ph,8,5v0V5By'


coords = []

bbbox = BBox(coords, crs=CRS.WGS84)
bsize = bbox_to_dimensions(bbbox, resolution=15)

evalscript_clm = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "CLM"],
    output: { bands: 3 }
  }
}

function evaluatePixel(sample) {
  if (sample.CLM == 1) {
    return [0.75 + sample.B04, sample.B03, sample.B02]
  } 
  return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
}
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_clm,
    input_data=[
	SentinelHubRequest.input_data(
	    data_collection=DataCollection.SENTINEL2_L1C,
	    time_interval=('2020-06-01', '2020-06-30'),
	    mosaicking_order='leastCC'
	)
    ],
    responses=[
	SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=bbbox,
    size=bsize,
    config=config
)

data_with_cloud_mask = request_true_color.get_data()


image = data_with_cloud_mask[0]
#print(image)

plot_image(image, factor=1/255, clip_range=(0,1))
#  plt.show()





imagename = "C:\\Users\dchir\Desktop\TreesProject\TestImages\plot2.png"
image = cv2.imread(imagename,cv2.IMREAD_COLOR)  ## Read image file
#print(imagename)
#print("test")

#boundaries detects foliage from green to almost white
boundaries = [
	([27,0, 98], [252, 229, 210])
]

for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)


	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
#boundary2 detects water from dark blue to light	
boundaries2 = [
	([32, 0, 4], [184,104, 112])
]

for (lower, upper) in boundaries2:
	# create NumPy arrays from the boundaries
	lower2 = np.array(lower, dtype = "uint8")
	upper2 = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask2 = cv2.inRange(image, lower2, upper2)
	output2 = cv2.bitwise_and(image, image, mask = mask2)


	# show the images
	cv2.imshow("images", np.hstack([image, output2]))
	cv2.waitKey(0)


#counts number of pixels in the boundaries
nPixels = np.count_nonzero(mask == 0)
count = mask.size
#percentage = the percentage of pixels deemed bad soil, or not able to support trees  - this could include water, dry land, etc
percentage = nPixels/count

nPixels2 = np.count_nonzero(mask2 == 0)
count2 = mask2.size
percentage2 = nPixels2/count2
#percentage2 = the percentage of pixels that are not deemed overcrowded by vegetation 

print("The percent of land that is deemed unpplantable for tree is: " + str(percentage*100) + "%")
print("The percent of land that is not overcrowded by dense vegetation is: " +str(percentage2*100) + "%")
if percentage > 0.9:
	print("The soil is not suitable for trees")
if percentage2 < 0.3:
	print("There is to much vegetation to plant a tree")

if percentage < 0.9 and percentage2 > 0.3:
	print("This land is suitable for trees")
