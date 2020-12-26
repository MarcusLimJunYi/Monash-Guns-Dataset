import numpy as np
# Divide an image into multiple regions for object detection task
#  image_shape = (w,h)
#  dividers_list = A list of integer numbers. The numbers represent how many columns the row will be divided into, from top row to bottom row respectively. 
#  overlap_rate = How much the regions overlap each other
def divideImage(image_shape, dividers_list, overlap_rate=0.1):
    _W=0
    _H=1
    rows = len(dividers_list)

    region_list = []
    baseY = 0
    for row, num_divide in enumerate(dividers_list):
        region_width = image_shape[_W]/num_divide
        overlap = region_width * overlap_rate
        for i in range(num_divide):
            x1 = i * region_width - overlap
            y1 = baseY - overlap
            x2 = (i+1) * region_width + overlap
            y2 = baseY + region_width + overlap
            if x1<0:                x1=0
            if x1>=image_shape[_W]: x1=image_shape[_W]-1
            if y1<0:                y1=0
            if y1>=image_shape[_H]: y1=image_shape[_H]-1
            if x2<0:                x2=0
            if x2>=image_shape[_W]: x2=image_shape[_W]-1
            if y2<0:                y2=0
            if y2>=image_shape[_H]: y2=image_shape[_H]-1
            region_list.append((int(x1),int(y1),int(x2),int(y2)))
        baseY+=region_width
    return region_list

def revertImage(boxes, region_count, count, baseY, image_shape, dividers_list, overlap_rate=0.1):
    _W=0
    _H=1

    updated_boxes=[]
    region_width = image_shape[_W]/dividers_list[count]
    overlap = region_width * overlap_rate

    for i in range(len(boxes)):
        boxes[i][0] = boxes[i][0] + (region_count-1) * region_width 
        boxes[i][1] = boxes[i][1] + baseY 
        boxes[i][2] = boxes[i][2] + (region_count-1) * region_width 
        boxes[i][3] = boxes[i][3] + baseY 
        if boxes[i][0]<0:                boxes[i][0]=0
        if boxes[i][0]>=image_shape[_W]: boxes[i][0]=image_shape[_W]-1
        if boxes[i][1]<0:                boxes[i][1]=0
        if boxes[i][1]>=image_shape[_H]: boxes[i][1]=image_shape[_H]-1
        if boxes[i][2]<0:                boxes[i][2]=0
        if boxes[i][2]>=image_shape[_W]: boxes[i][2]=image_shape[_W]-1
        if boxes[i][3]<0:                boxes[i][3]=0
        if boxes[i][3]>=image_shape[_H]: boxes[i][3]=image_shape[_H]-1
        updated_boxes.append((float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3])))
    
    return np.asarray(updated_boxes)

# Prepare data for object detection task
#  - Crop input image based on the region list produced by divideImage()
#  - Create a list of task which consists of coordinate of the ROI in the input image, and the image of the ROI
def createObjectDectionTasks(img, region_list):
    task_id = 0
    task_list = []
    for region in region_list:
        ROI = img[region[1]:region[3], region[0]:region[2]]
        task_list.append([region, ROI])
    return task_list