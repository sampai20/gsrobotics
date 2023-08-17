import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d
import numpy as np
import math
import enum
import os
import time
import cv2
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import fftpack
from collections import deque



# creating enumerations using class
class Finger(enum.Enum):
    R1 = 1
    R15 = 2
    DIGIT = 3
    MINI = 4

def find_marker(gray):
    mask = cv2.inRange(gray, 5, 55)
    mask = np.asarray(mask)
    mask = np.where(mask, 1, 0).astype('uint8')
    neighborhood_size = 6
    data_max = maximum_filter(mask, neighborhood_size, mode='constant', cval=0)
    new_mask = np.where(data_max > 0, 1, 0).astype('uint8')
    #dilation = cv2.dilate(mask, kernel, iterations=1)
    #print(mask, dilation)
    cv2.waitKey(1)
    return new_mask

def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def matching_rows(A,B):
    ### https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    matches=[i for i in range(B.shape[0]) if np.any(np.all(A==B[i],axis=1))]
    if len(matches)==0:
        return B[matches]
    return np.unique(B[matches],axis=0)

def interpolate_gradients(gx, gy, img, cm, markermask):
    ''' interpolate gradients at marker location '''

    # if np.where(cm)[0].shape[0] != 0:
    cmcm = np.zeros(img.shape[:2])
    ind1 = np.vstack(np.where(cm)).T
    ind2 = np.vstack(np.where(markermask)).T
    ind2not = np.vstack(np.where(~markermask)).T
    ind3 = matching_rows(ind1, ind2)
    cmcm[(ind3[:, 0], ind3[:, 1])] = 1.
    ind4 = ind1[np.all(np.any((ind1 - ind3[:, None]), axis=2), axis=0)]
    x = np.linspace(0, 240, 240)
    y = np.linspace(0,320, 320)
    X, Y = np.meshgrid(x, y)

    '''interpolate at the intersection of cm and markermask '''
    # gx_interpol = griddata(ind4, gx[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gx[(ind3[:, 0], ind3[:, 1])] = gx_interpol
    # gy_interpol = griddata(ind4, gy[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gy[(ind3[:, 0], ind3[:, 1])] = gy_interpol

    ''' interpolate at the entire markermask '''
    gx_interpol = griddata(ind2, gx[(ind2[:, 0], ind2[:, 1])], gx[(ind2not[:, 0], ind2not[:, 1])], method='nearest')
    gx[(ind2not[:, 0], ind2not[:, 1])] = gx_interpol
    gy_interpol = griddata(ind2, gy[(ind2[:, 0], ind2[:, 1])], gy[(ind2not[:, 0], ind2not[:, 1])], method='nearest')
    gy[(ind2not[:, 0], ind2not[:, 1])] = gy_interpol
    #print (gy_interpol.shape, gx_interpol.shape, gx.shape, gy.shape)

    ''' interpolate using samples in the vicinity of marker '''


    ''' method #3 '''
    # ind1 = np.vstack(np.where(markermask)).T
    # gx_interpol = scipy.ndimage.map_coordinates(gx, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gx_interpol
    # gy_interpol = scipy.ndimage.map_coordinates(gy, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gy_interpol

    ''' method #4 '''
    # x = np.arange(0, img.shape[0])
    # y = np.arange(0, img.shape[1])
    # fgx = scipy.interpolate.RectBivariateSpline(x, y, gx, kx=2, ky=2, s=0)
    # gx_interpol = fgx.ev(ind2[:,0],ind2[:,1])
    # gx[(ind2[:, 0], ind2[:, 1])] = gx_interpol
    # fgy = scipy.interpolate.RectBivariateSpline(x, y, gy, kx=2, ky=2, s=0)
    # gy_interpol = fgy.ev(ind2[:, 0], ind2[:, 1])
    # gy[(ind2[:, 0], ind2[:, 1])] = gy_interpol

    return gx_interpol, gy_interpol

def manhattan_griddata(points, values, marker_points):
    start = time.time()
    to_visit = []
    answers = np.ones((245, 325)) * 5
    print(points.shape[0], marker_points.shape[0])

    answers[points[:, 0], points[:, 1]] = values
    answers[marker_points[:, 0], marker_points[:, 1]] = 10


    for p in range(points.shape[0]):
        to_visit.append((points[p]))

    mid = time.time()

    deltas = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
    ct = 0
    while len(to_visit):
        ct += 1
        a = to_visit.pop()
        neighbors = a[None, :] + deltas
        neighbor_answers = answers[neighbors[:, 0], neighbors[:, 1]]
        for i in range(4):
            if neighbor_answers[i] == 10:
                answers[neighbors[i][0], neighbors[i][1]] = answers[a[0], a[1]]
                to_visit.append(neighbors[i])

    res = answers[marker_points[:, 0], marker_points[:, 1]]

    print(ct)

    end = time.time()

    return res


def interpolate_grad(gx, gy, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # pixel around markers
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & (mask == 0)
    # mask_around = mask == 0

    x, y = np.arange(gx.shape[0]), np.arange(gx.shape[1])
    yy, xx = np.meshgrid(y, x)

    # cv2.imshow("mask_zero", mask_zero*1.)

    # if np.where(mask_zero)[0].shape[0] != 0:
    #     print ('interpolating')
    mask_x = xx[mask_around]
    mask_y = yy[mask_around]
    points = np.stack([mask_x, mask_y], axis=1)
    values = np.stack((gx[mask_x, mask_y], gy[mask_x, mask_y]), axis=0)
    markers_points = np.stack([xx[mask != 0], yy[mask != 0]], axis=1)


    kdtree = KDTree(points)
    d, i = kdtree.query(markers_points, p=2, distance_upper_bound = 15, workers=1)
    interp_values = values[:, i]

    gx_interp = gx
    gy_interp = gy
    gx_interp[mask != 0] = interp_values[0, :]
    gy_interp[mask != 0] = interp_values[1, :]
    # else:
    #     ret = img
    return gx_interp, gy_interp

def demark(gx, gy, markermask):
    start = time.time()
    # mask = find_marker(img)
    gx_interp, gy_interp = interpolate_grad(gx, gy, markermask)
    # gx_interp = interpolate_grad(gx.copy(), markermask)
    # gy_interp = interpolate_grad(gy.copy(), markermask)
    end = time.time()
    return gx_interp, gy_interp

#@njit(parallel=True)
def get_features(img,pixels,features,imgw,imgh):
    features[:,3], features[:,4]  = pixels[:,0] / imgh, pixels[:,1] / imgw
    for k in range(len(pixels)):
        i,j = pixels[k]
        rgb = img[i, j] / 255.
        features[k,:3] = rgb

#
# 2D integration via Poisson solver
#
def poisson_dct_neumaan(gx,gy):

    gxx = 1 * (gy[(list(range(1,gx.shape[0]))+[gx.shape[0]-1]), :] - gy[([0]+list(range(gx.shape[0]-1))), :])
    gyy = 1 * (gx[:, (list(range(1,gx.shape[1]))+[gx.shape[1]-1])] - gx[:, ([0]+list(range(gx.shape[1]-1)))])
    f = gxx + gyy

    ### Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0,1:-2] = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0] = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0] = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1] = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1] = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0] = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    ## Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0,1:-2] = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0] = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    ## Modification near the corners (Eq. 54 in [1])
    f[0,-1] = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0] = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0] = f[0,0] - np.sqrt(2) * b[0,0]

    ## Cosine transform of f
    tt = fftpack.dct(f, norm='ortho')
    fcos = fftpack.dct(tt.T, norm='ortho').T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * ( (np.sin(0.5*math.pi*x/(f.shape[1])))**2 + (np.sin(0.5*math.pi*y/(f.shape[0])))**2)

    # 4 * ((sin(0.5 * pi * x / size(p, 2))). ^ 2 + (sin(0.5 * pi * y / size(p, 1))). ^ 2)

    f = -fcos / denom
    # Inverse Discrete cosine Transform
    tt = fftpack.idct(f, norm='ortho')
    img_tt = fftpack.idct(tt.T, norm='ortho').T

    img_tt = img_tt.mean() + img_tt
    # img_tt = img_tt - img_tt.min()

    return img_tt


class RGB2NormNetR1(nn.Module):
    def __init__(self):
        super(RGB2NormNetR1, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 2)
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc2(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc3(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc4(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc5(x))
        x = F.tanh(self.fc6(x))
        x = self.fc7(x)
        return x


''' nn architecture for r1.5 and mini '''
class RGB2NormNetR15(nn.Module):
    def __init__(self):
        super(RGB2NormNetR15, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x

class Reconstruction3D:
    def __init__(self, finger, dims, zero_path=None):
        self.finger = finger
        self.cpuorgpu = "cpu"
        if zero_path is None:
            self.dm_zero_counter = 0
            self.dm_zero = np.zeros(dims)
            self.gx_zero = np.zeros(dims)
            self.gy_zero = np.zeros(dims)
        else:
            self.dm_zero_counter = 50
            saved_data = np.load(zero_path)
            self.dm_zero, self.gx_zero, self.gy_zero = saved_data[0], saved_data[1], saved_data[2]

    def load_nn(self, net_path, cpuorgpu):

        self.cpuorgpu = cpuorgpu
        device = torch.device(cpuorgpu)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return

        #print('self.finger = ', self.finger)
        if self.finger == Finger.R1:
            print('calling nn R1...')
            net = RGB2NormNetR1().float().to(device)
        elif self.finger == Finger.R15:
            print(f'calling nn with {net_path}')
            net = RGB2NormNetR15().float().to(device)
        else:
            net = RGB2NormNetR15().float().to(device)

        if cpuorgpu=="cuda":
            ### load weights on gpu
            # net.load_state_dict(torch.load(net_path))
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(0))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            ### load weights on cpu which were actually trained on gpu
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])

        self.net = net

        return self.net

    def save_zero(self, save_path):
        if self.dm_zero_counter < 50:
            raise ValueError("not fully zero'd yet!")
        
        np.save(save_path, np.stack((self.dm_zero, self.gx_zero, self.gy_zero)))

    def get_depthmap(self, frame, mask_markers, cm=None, return_grads=False):
        MARKER_INTERPOLATE_FLAG = mask_markers

        start = time.time()

        ''' find contact region '''
        # cm, cmindx = find_contact_mask(f1, f0)
        ###################################################################
        ### check these sizes
        ##################################################################
        if (cm is None):
            cm, cmindx = np.ones(frame.shape[:2]), np.where(np.ones(frame.shape[:2]))


        if MARKER_INTERPOLATE_FLAG:
            ''' find marker mask '''
            markermask = find_marker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cm = ~markermask

        cmx, cmy = np.where(cm)
        imgh = frame.shape[:2][0]
        imgw = frame.shape[:2][1]

        ''' Get depth image with NN '''
        normals = np.zeros((2, frame.shape[0], frame.shape[1]))
        dm = np.zeros(frame.shape[:2])

        ''' ENTIRE CONTACT MASK THRU NN '''
        # if np.where(cm)[0].shape[0] != 0:
        rgb = frame[cmx, cmy] / 255
        # rgb = diffimg[np.where(cm)]
        pxpos = np.vstack((cmx, cmy)).T
        # pxpos[:, [1, 0]] = pxpos[:, [0, 1]] # swapping
        pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / imgh, pxpos[:, 1] / imgw
        # the neural net was trained using height=320, width=240
        # pxpos[:, 0] = pxpos[:, 0] / ((320 / imgh) * imgh)
        # pxpos[:, 1] = pxpos[:, 1] / ((240 / imgw) * imgw)

        features = np.column_stack((rgb, pxpos))
        features = torch.from_numpy(features).float().to(self.cpuorgpu)
        with torch.no_grad():
            self.net.eval()
            out = self.net(features).cpu().detach().numpy()



        normals[:, cmx, cmy] = out.T
        # print(nx.min(), nx.max(), ny.min(), ny.max())
        # nx = 2 * ((nx - nx.min()) / (nx.max() - nx.min())) -1
        # ny = 2 * ((ny - ny.min()) / (ny.max() - ny.min())) -1
        # print(nx.min(), nx.max(), ny.min(), ny.max())

        '''OPTION#1 normalize gradient between [a,b]'''
        # a = -5
        # b = 5
        # gx = (b-a) * ((gx - gx.min()) / (gx.max() - gx.min())) + a
        # gy = (b-a) * ((gy - gy.min()) / (gy.max() - gy.min())) + a
        '''OPTION#2 calculate gx, gy from nx, ny. '''
        ### normalize normals to get gradients for poisson
        nz = np.sqrt(1 - np.sum(normals ** 2, axis=0))
        nz[np.where(np.isnan(nz))] = 0
        grads = normals / nz[None, :]
        gx = grads[0]
        gy = grads[1]
        mid = time.time()


        if MARKER_INTERPOLATE_FLAG:
            # gx, gy = interpolate_gradients(gx, gy, img, cm, cmmm)
            gx_interp, gy_interp = demark(gx, gy, markermask)
        else:
            gx_interp, gy_interp = gx, gy


        # nz = np.sqrt(1 - nx ** 2 - ny ** 2)

        dm = poisson_dct_neumaan(gx_interp, gy_interp)
        dm = np.reshape(dm, (imgh, imgw))
        #print(dm.shape)
        # cv2.imshow('dm',dm)

        ''' remove initial zero depth '''
        if self.dm_zero_counter < 50:
            self.dm_zero += dm
            self.gx_zero += gx_interp
            self.gy_zero += gy_interp
            print ('zeroing depth. do not touch the gel!')
            if self.dm_zero_counter == 49:
                self.dm_zero /= self.dm_zero_counter
                self.gx_zero /= self.dm_zero_counter
                self.gy_zero /= self.dm_zero_counter
        if self.dm_zero_counter == 50:
            print ('Ok to touch me now!')
        self.dm_zero_counter += 1
        dm = dm - self.dm_zero
        # print(dm.min(), dm.max())

        ''' ENTIRE MASK. GPU OPTIMIZED VARIABLES. '''
        # if np.where(cm)[0].shape[0] != 0:
        ### Run things through NN. FAST!!??
        # pxpos = np.vstack(np.where(cm)).T
        # features = np.zeros((len(pxpos), 5))
        # get_features(img, pxpos, features, imgw, imgh)
        # features = torch.from_numpy(features).float().to(device)
        # with torch.no_grad():
        #     net.eval()
        #     out = net(features)
        # # Create gradient images and do reconstuction
        # gradx = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady[pxpos[:, 0], pxpos[:, 1]] = out[:, 0]
        # gradx[pxpos[:, 0], pxpos[:, 1]] = out[:, 1]
        # # dm = poisson_reconstruct_gpu(grady, gradx, denom).cpu().numpy()
        # dm = cv2.resize(poisson_reconstruct(grady, gradx, denom).cpu().numpy(), (640, 480))
        # dm = cv2.resize(dm, (imgw, imgh))
        # # dm = np.clip(dm / img.max(), 0, 1)
        # # dm = 255 * dm
        # # dm = dm.astype(np.uint8)

        end = time.time()
        #print(mid - start, end - mid)

        return dm


class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m
        self.init_open3D()
        self.cnt = 212
        self.save_path = save_path
        pass

    def init_open3D(self):
        x = np.arange(self.n)# * mmpp
        y = np.arange(self.m)# * mmpp
        self.X, self.Y = np.meshgrid(x,y)
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X) #/ self.m
        self.points[:, 1] = np.ndarray.flatten(self.Y) #/ self.n

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z):
        self.depth2points(Z)
        dx, dy = np.gradient(Z)
        dx, dy = dx * 0.5, dy * 0.5

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3): colors[:,_]  = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        #### SAVE POINT CLOUD TO A FILE
        if self.save_path != '':
            open3d.io.write_point_cloud(self.save_path + "/pc_{}.pcd".format(self.cnt), self.pcd)

        self.cnt += 1

    def save_pointcloud(self):
        open3d.io.write_point_cloud(self.save_path + "pc_{}.pcd".format(self.cnt), self.pcd)





