import numpy as np
from realsenseconfig import D435_para

def Report_PCL(RGB_Pix_POS):
    RGB_Pix_POS = np.array([(RGB_Pix_POS[0] + RGB_Pix_POS[1]) / 2, (RGB_Pix_POS[2] + RGB_Pix_POS[3]) / 2])
    print("POS", RGB_Pix_POS)
    pix_3d=np.mat(np.append(RGB_Pix_POS,[1.00]))
    Color_cam=D435_para.color_inner_matirx.I*pix_3d.T #D435_para.depthmat[int(self.RGB_Pix_POS[0]),int(self.RGB_Pix_POS[1])]
    Depth_cam=D435_para.color_to_depth_rotation.I*(Color_cam-D435_para.color_to_depth_translation.T)
    D_m=D435_para.depth_inner_matrix*Depth_cam
    # print(Color3[2,0],Color_cam[2,0])
    if D_m[0,0]/D_m[2,0]<479:
        P1=D_m[0,0]/D_m[2,0]
    else:
        P1=479
    if D_m[1,0]/D_m[2,0]<639:
        P2=D_m[1,0]/D_m[2,0]
    else:
        P2=639
    Depth_pix=np.array([P1,P2])
    Image_pix=np.mat(np.append(Depth_pix,[1.00]))
    PCL=D435_para.depth_inner_matrix.I*Image_pix.T*D435_para.depthmat[int(P1),int(P2)]
    PCL=[int(PCL[0,0]),int(PCL[1,0]),int(PCL[2,0])]
    # print(PCL)
    return PCL



# import numpy as np
# # from realsenseconfig import D435_para

# color_inner_matirx_I = np.matrix([[ 0.00162675 , 0.    ,     -0.52758936],
# [ 0. ,         0.00162652, -0.38762233],
# [ 0.  ,        0.    ,      1.        ]])
# color_to_depth_rotation_I = np.matrix([[ 0.99997475 , 0.00697401 , 0.00137164],
# [-0.00697127,  0.99997373 ,-0.00198921],
# [-0.00138548 , 0.0019796 ,  0.99999708]])
# color_to_depth_translation_T = np.matrix([[ 1.46501148e-02],
# [-6.37766498e-05],
# [ 4.11829911e-04]])
# depth_inner_matrix = np.matrix([[390.15472412  , 0.   ,      320.42520142],
# [  0.    ,     390.15472412, 246.77757263],
# [  0.     ,      0.      ,     1.        ]])
# depth_inner_matrix_I = np.matrix([[ 0.00256309 , 0.  ,       -0.82127726],
# [ 0.  ,        0.00256309, -0.63251207],
# [ 0.    ,      0.    ,      1.        ]])
# class Position_tansform:
#     def __init__(self,RGB_pix_Position):
#         # self.Depth_cam,self.Color_cam=np.mat(np.array([0.0,0.0,0.0]))
#         self.RGB_Pix_POS=RGB_pix_Position;
#     def Report_PCL(self,depth):
#         pix_3d=np.mat(np.append(self.RGB_Pix_POS,[1.00]))
#         self.Color_cam=color_inner_matirx_I*pix_3d.T #D435_para.depthmat[int(self.RGB_Pix_POS[0]),int(self.RGB_Pix_POS[1])]
#         self.Depth_cam= color_to_depth_rotation_I*(self.Color_cam-color_to_depth_translation_T)
#         D_m=depth_inner_matrix*self.Depth_cam
#         # print(Color3[2,0],self.Color_cam[2,0])
#         if D_m[0,0]/D_m[2,0]<479:
#             P1=D_m[0,0]/D_m[2,0]
#         else:
#             P1=479
#         if D_m[1,0]/D_m[2,0]<639:
#             P2=D_m[1,0]/D_m[2,0]
#         else:
#             P2=639
#         self.Depth_pix=np.array([P1,P2])
#         Image_pix=np.mat(np.append(self.Depth_pix,[1.00]))
#         PCL=depth_inner_matrix_I*Image_pix.T*depth[int(P1),int(P2)]
#         self.PCL=[int(PCL[0,0]),int(PCL[1,0]),int(PCL[2,0])]
#         # print(self.PCL)
#         return self.PCL


# # print(type(D435_para.color_inner_matirx.I))
# # """
# # [[ 0.00162675  0.         -0.52758936]
# #  [ 0.          0.00162652 -0.38762233]
# #  [ 0.          0.          1.        ]]
# # """
# # print(type(D435_para.color_to_depth_rotation.I))
# # """
# # [[ 0.99997475  0.00697401  0.00137164]
# #  [-0.00697127  0.99997373 -0.00198921]
# #  [-0.00138548  0.0019796   0.99999708]]
# # """
# # print(type(D435_para.color_to_depth_translation.T))
# # """
# # [[ 1.46501148e-02]
# #  [-6.37766498e-05]
# #  [ 4.11829911e-04]]
# # """
# # print(type(D435_para.depth_inner_matrix))
# # """
# # [[390.15472412   0.         320.42520142]
# #  [  0.         390.15472412 246.77757263]
# #  [  0.           0.           1.        ]]
# # """
# # print(type(D435_para.depth_inner_matrix.I))
# # """
# # [[ 0.00256309  0.         -0.82127726]
# #  [ 0.          0.00256309 -0.63251207]
# #  [ 0.          0.          1.        ]]
# # """






