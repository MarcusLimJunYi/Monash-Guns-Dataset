import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 22})

# with open('precision-recall/Table5_faster_rcnn.txt') as f:
#     lines = f.readlines()
#     data = [line.split()[0] for line in lines]
# y1 = data  
# y1 = np.asarray(y1,dtype=np.float32)

# with open('precision-recall/Table6_faster_rcnn.txt') as f:
#     lines = f.readlines()
#     data = [line.split()[0] for line in lines]
# y2 = data
# y2 = np.asarray(y2,dtype=np.float32)

# with open('precision-recall/Table5_M2Det.txt') as f:
#     lines = f.readlines()
#     data = [line.split()[0] for line in lines]
# y3 = data
# y3 = np.asarray(y3,dtype=np.float32)

# with open('precision-recall/Table6_M2Det.txt') as f:
#     lines = f.readlines()
#     data = [line.split()[0] for line in lines]
# y4 = data
# y4 = np.asarray(y4,dtype=np.float32)

# with open('precision-recall/Precision_Recall_Resnet101_Final.txt') as f:
#     lines = f.readlines()
#     data = [line.split()[0] for line in lines]
# y5 = data
# y5 = np.asarray(y5,dtype=np.float32)

# x = np.arange(0,1,step=1/len(data),dtype=float)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)

# ax1.set_title("Precision-Recall Curve",fontweight="bold")    
# ax1.set_xlabel('Recall',fontweight="bold")
# ax1.set_ylabel('Precision',fontweight="bold") 

# linestyle='--'
# # ax1.plot(x,y1, label='Faster R-CNN with FPN (Granada Dataset)', marker="^",markersize=16,fillstyle='none',linestyle=linestyle,linewidth=3)
# # ax1.plot(x,y2, label='Faster R-CNN with FPN (Granada and Our Dataset)', marker="s",markersize=10,fillstyle='none',linestyle=linestyle,linewidth=3)
# # ax1.plot(x,y3, label='M2Det (Granada Dataset)', marker="x",markersize=10,fillstyle='none',linestyle=linestyle,linewidth=3)
# # ax1.plot(x,y4, label='M2Det (Granada and Our Dataset)', marker="o",markersize=10,fillstyle='none',linestyle=linestyle,linewidth=3)
# ax1.plot(x,y5, label='M2Det with Focal Loss (Granada and Our Dataset)', marker="p",markersize=10,fillstyle='none',linestyle=linestyle,linewidth=3)
# plt.show()

# Plot Precision-Frames Graph
with open('test1/detections_ADNMS.txt') as f:
    lines = f.readlines()
    data_y = [line.split(" ")[0] for line in lines]

y_pf = data_y
x_pf = np.arange(0,len(y_pf),step=1,dtype=float)
y_pf = np.asarray(y_pf,dtype=np.float32)
x_pf = np.asarray(x_pf,dtype=np.float32)
x_pf = x_pf[np.where(y_pf>0.2)]
y_pf = y_pf[np.where(y_pf>0.2)]

#Plot PF Graph
fig = plt.figure(figsize=(20,15))
ax_pf = fig.add_subplot(111)

# ax_pf.set_title("Real-world Video Surveillance Performance",fontweight="bold")    
ax_pf.set_xlabel('Frames',fontweight="bold")
ax_pf.set_ylabel('Precision',fontweight="bold") 

linestyle='--'
ax_pf.plot(x_pf,y_pf, marker=".",markersize=16,fillstyle='none',linestyle=linestyle,linewidth=2)

plt.grid(b=True,which='both',axis='both',linestyle='--')
ax_pf.set_ylim(0, 1)
ax_pf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()
fig.savefig('Precision_Frames_Graph.jpg', format='jpg', dpi=200)

# Plot Precision-Distance Graph
with open('Demo_Distance/detections_ADNMS.txt') as f:
    lines = f.readlines()
    data_y = [line.split(" ")[0] for line in lines]
    data_x = [line.split(" ")[1] for line in lines]
y_pd = data_y
x_pd = data_x
y_pd = np.asarray(y_pd,dtype=np.float32)
x_pd = np.asarray(x_pd,dtype=np.float32)
x_pd = x_pd[np.where(y_pd>0.2)]
y_pd = y_pd[np.where(y_pd>0.2)]

#Plot PD Graph
fig = plt.figure(figsize=(20,15))
ax_pd = fig.add_subplot(111)

# ax_pd.set_title("Precision Across Varying Distances",fontweight="bold")    
ax_pd.set_xlabel('Distance (cm)',fontweight="bold")
ax_pd.set_ylabel('Precision',fontweight="bold") 

linestyle='--'
ax_pd.plot(x_pd,y_pd, marker=".",markersize=16,fillstyle='none',linestyle=linestyle,linewidth=2)

plt.grid(b=True,which='both',axis='both',linestyle='--')
ax_pd.set_ylim(0, 1)
ax_pd.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()
fig.savefig('Precision_Distance_Graph.jpg', format='jpg', dpi=200)

# #Plot GRAD-CAM Graph
# fig = plt.figure()
# ax_gcam = fig.add_subplot(111)

# ax_gcam.set_title("Gradient-weighted Class Activation Mapping",fontweight="bold")    
# ax_gcam.set_xlabel('Levels',fontweight="bold")
# ax_gcam.set_ylabel('Scales',fontweight="bold") 

# x_gcam = np.arange(0,10,step=1,dtype=int)
# y_gcam = np.arange(0,10,step=1,dtype=int)
# ax_gcam.plot(x_gcam,y_gcam,'w',linewidth=0)


# plt.grid(b=True,which='both',axis='both',linestyle='--')
# plt.yticks(np.arange(min(y_gcam), max(y_gcam)+1, 1.0))
# plt.xticks(np.arange(min(x_gcam), max(x_gcam)+1, 1.0))
# # plt.set_yticks(range(0,7,2))
# # ax_gcam.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# ax_gcam.set_ylim(top=7)
# plt.show()
# fig.savefig('grad_cam.jpg', format='jpg', dpi=300)