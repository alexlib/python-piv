"""
This library analyzes PIV data for Low Swirl Burner from LBL Combustion Lab

V1.1
- Updated PivImage.image_is_valid() to catch blank images, v1.0 did not handle correctly
- Updated PivImage.find_flame_centerline_minxy() to handle case where contour begins
  on one side of centerline and ends on other side of centerline
  		- I now "close" the flame edge contour that cv2.findCountours() creates

Author: Darren Sholes
Last Updated: May 20, 2018
"""

from PIL import Image, ImageEnhance
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_style('ticks')

FLAG_CODE = {
    'empty_image': 1, # No pixels OR no flame detected, see PivImage.image_is_valid()
    'faint_image': 2, # Image is faint, treat result with caution, see PivImage.image_is_valid()
    'cen_xy_fail': 3, # Failure to find point on flame edge at centerline
}

BLUR_KERNEL = (13,13)

# Pixel values used for thresholding, THESE MUST BE INTEGERS, DO NOT CHANGE TO SCALAR
BINARY_PIXEL_MIN = 0 
BINARY_PIXEL_MAX = 255

CENTERLINE_X = 1024 # Based on 2048 x 2046 pixel image. How do we know burner is centered?

# Search Window Parameters, see PivImage.find_flame_edge() and PivImage.find_flame_brush_minxy()
X_MIN, X_MAX = (600,1500)
Y_MIN, Y_MAX = (50,800)

# Recursively increase Y_MAX by this amount each time PivImage.find_flame_centerline_minxy() fails
SEARCH_WIN_SIZE_INC = 200

class PivImage:
    def __init__(self, image_path, run_id):
        self.image = Image.open(image_path)
        self.run_id = run_id
        self.flags = []
        self.main()
        
    def main(self):
        
        self.normalize_image()
        
        if not self.image_is_valid():
            self.flame_min_x = np.nan
            self.flame_min_y = np.nan
            self.flame_centerline_x = np.nan
            self.flame_centerline_y = np.nan
            return
        
        else:
            self.remove_noise()
            self.morphology_close()
            self.binary_threshold()
            self.find_flame_brush_minxy(X_MIN, X_MAX, Y_MIN, Y_MAX)
            return
    
    @staticmethod
    def pixel_density(binary_image):
        pixel_count_max = (binary_image == BINARY_PIXEL_MAX).sum()
        pixel_count_min = (binary_image == BINARY_PIXEL_MIN).sum()
        pixeldensity = pixel_count_max/pixel_count_min
        return pixeldensity
    
    def image_is_valid(self):
        """
        - This is an attempt to check for 'empty' and 'faint' images
        - Empty can mean no pixels OR no flame detected
        
        """
        all_density = self.pixel_density(self.binary_image)
        bottom_density = self.pixel_density(self.binary_image[0:1000])
        top_density = self.pixel_density(self.binary_image[1000:])
        
        if (((bottom_density/all_density) - (top_density/all_density)) < 0.5) or \
           (all_density < 0.01):

            # - If bottom of image is not much more 'dense' than top of image,
            #   then most likely, no flame exists
            # - Warning: This is my best attempt to remove 'empty' images,
            #   but check final results/outliers to ensure no strange cases
            #   made it through
            
            self.flags.append(FLAG_CODE['empty_image'])
            return False
        
        else:
            if all_density < 0.1:
                # - Warning: arbitrary threshold, just want to sanity check very 'faint'
                #   images after analysis, and this provides opportunity to flag
                self.flags.append(FLAG_CODE['faint_image'])
            return True
    
    def normalize_image(self):
        """
        - Binary threshold used to normalize all images to the same "brightness"
        
        """
        im = np.flipud(np.copy(self.image))
        ret3, self.binary_image = cv2.threshold(im,BINARY_PIXEL_MIN,BINARY_PIXEL_MAX,
                                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return
    
    def remove_noise(self):
        """
        - Perform Gaussian Blur to remove noise in image
        
        """
        self.binary_blurred = cv2.GaussianBlur(self.binary_image,BLUR_KERNEL,0)
        return
    
    def morphology_close(self):
        """
        - After Gaussian Blur, image features may have "holes". This function attempts
          to close those holes. See this description: 
              'https://docs.opencv.org/3.0-beta/doc/py_tutorials/
               py_imgproc/py_morphological_ops/py_morphological_ops.html#closing'
               
        """ 
        kernel = np.ones(BLUR_KERNEL,np.uint8)
        self.binary_closed = cv2.morphologyEx(self.binary_blurred,cv2.MORPH_CLOSE,kernel)
        return
    
    def binary_threshold(self):
        """
        - After blurring and closing, need additional binary threshold to make contour detection easier
        """
        ret3, self.binary_im_final = cv2.threshold(self.binary_closed,BINARY_PIXEL_MIN,BINARY_PIXEL_MAX,
                                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return
    
    def find_flame_brush_minxy(self, xmin, xmax, ymin, ymax):
        """
        - Find coordinates of minimum point of flame brush,
          both along the centerline and absolute minimum
            - "Minimum" means along Y axis of images
              i.e. the height at which the flame "sits"
        - xmin, xmax, ymin, ymax define a smaller search "window"
          within the image to look for contours of the flame
            - This speeds up code tremendously
            - Window is recursively enlarged if 
              find_flame_centerline_minxy() fails
        """
        
        self.flags.append(FLAG_CODE['cen_xy_fail'])
        
        while (ymax < (self.binary_image.shape[0]-1)) and (3 in self.flags):
            search_window = (xmin, xmax, ymin, ymax)
            self.find_flame_edge(search_window)
            self.find_flame_centerline_minxy()
            self.find_flame_abs_minxy()
            ymax += SEARCH_WIN_SIZE_INC
        
        if 3 in self.flags:
            self.flame_min_x = np.nan
            self.flame_min_y = np.nan
            self.flame_centerline_x = np.nan
            self.flame_centerline_y = np.nan
        return
    
    def find_flame_edge(self, search_window):
        xmin, xmax, ymin, ymax = search_window
        self.binary_flame_edge = BINARY_PIXEL_MAX - self.binary_im_final[ymin:ymax,xmin:xmax]
        im2,flame_edges,hierarchy = cv2.findContours(np.copy(self.binary_flame_edge), 1, 2)
        con_id = np.argmax([cv2.contourArea(item) for item in flame_edges])
        self.flame_edge = flame_edges[con_id]
        self.flame_edge = np.concatenate([self.flame_edge,self.flame_edge[0:1]]) # Closing the contour
        self.flame_edgeX = xmin + self.flame_edge.transpose()[0][0]
        self.flame_edgeY = ymin + self.flame_edge.transpose()[1][0]
    
    @staticmethod
    def interp_on_centerline(x_desired,x_paired,y_paired):
        """
        - Interpolate between points on flame contour to find centerline XY
        - Takes in arrays of paired data, for example: 
            x_paired = [[1,2,3],
                        [2,4,5]] interpolate between 1 & 2, 2 & 4, etc.
        """
        x0 = x_paired[0]
        x1 = x_paired[1]
        y0 = y_paired[0]
        y1 = y_paired[1]
        if isinstance(x_desired, (list, tuple, np.ndarray)):
            x = x_desired
        else:
            x = x_desired*np.ones(len(x0))

        return y0 + (x-x0)*(y1-y0)/(x1-x0)
    
    def find_flame_centerline_minxy(self):
        """
        - Finds coordinates of minimum point along centerline of flame
        - Will interpolate between points on flame contour if no point exists 
          at intersection of contour and centerline
        """
        i_on_cent = np.where(self.flame_edgeX == CENTERLINE_X)[0]
        flame_edgeY_cent = self.flame_edgeY[i_on_cent]

        # indices of pts about centerline (on either side of centerline)
        i_near_cent = np.where(np.diff(np.sign(self.flame_edgeX - CENTERLINE_X)) != 0)[0] 
        
        # x about centerline (on either side of centerline)
        x_near_cent = np.array([self.flame_edgeX[i_near_cent],
                                self.flame_edgeX[i_near_cent+1]]) 
        # y about centerline (on either side of centerline)
        y_near_cent = np.array([self.flame_edgeY[i_near_cent],
                                self.flame_edgeY[i_near_cent+1]])

        flame_edgeY_cent_interp = self.interp_on_centerline(CENTERLINE_X, x_near_cent, y_near_cent)

        flame_edgeY_cent = np.concatenate([flame_edgeY_cent,
                                           flame_edgeY_cent_interp])
        
        if not flame_edgeY_cent.size == 0.:
            self.flame_centerline_x = CENTERLINE_X
            self.flame_centerline_y = int(flame_edgeY_cent.min())
            self.flags.remove(3)
    
        return
    
    def find_flame_abs_minxy(self):
        """
        - Finds coordinates of absolute minimum point of flame brush
          (not limited to centerline)
        """
        min_flame_id = self.flame_edgeY.argmin()
        self.flame_min_x = self.flame_edgeX[min_flame_id]
        self.flame_min_y = self.flame_edgeY[min_flame_id]
        return
    
    def plot_results(self,filename='piv_results_plot.jpg',savefig = False):
        """
        - NOTE: When savefig is True, plot will not show. It is closed using plt.close()
            - This is to avoid lots of plots and windows from showing when saving lots of plots
        """
        if savefig:
            # This will fail if saving in same directory as run program, at least for relative paths
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig1, (ax1,ax2) = plt.subplots(1,2,sharey=True)
        ax1.imshow(self.binary_image,origin='lower')
        if (1 in self.flags) or (3 in self.flags):
            ax1.tick_params(axis='both',labelsize=14)
            ax1.set_ylabel('Pixels',fontsize=16)
            ax1.set_xlabel('Pixels',fontsize=16)
            fig1.suptitle('Mini LSB Flame Leading Edge: {0}'.format(filename), fontsize = 20)
            fig1.set_size_inches(15,7)
            if savefig:
                fig1.savefig(filename)
                plt.close()
            return
        ax1.plot(self.flame_edgeX,
                 self.flame_edgeY,
                 color = 'r')
        ax1.annotate('AbsX = {0} Px,\n AbsY = {1} Px'.format(
            self.flame_min_x, self.flame_min_y),
                     xy=(self.flame_min_x,self.flame_min_y), fontsize = 12,
                     xytext=(1000,1000),arrowprops={'width':2,'headwidth':7,'color':'#f4a742'},backgroundcolor='w')
        ax1.plot(self.flame_min_x,self.flame_min_y,'o',ms=6,mfc = '#59ABE3',mec='#59ABE3')

        ax1.annotate('CenterX = {0} Px,\n CenterY = {1} Px'.format(
            self.flame_centerline_x,self.flame_centerline_y),
                     xy=(self.flame_centerline_x,self.flame_centerline_y), fontsize=12,
                     xytext=(1400,100),arrowprops={'width':2,'headwidth':7,'color':'#f4a742'},backgroundcolor='w')
        ax1.plot(self.flame_centerline_x,self.flame_centerline_y,'o',ms=8,mfc = '#59ABE3',mec='#59ABE3')
        ax1.tick_params(axis='both',labelsize=14)
        ax1.set_ylabel('Pixels',fontsize=16)
        ax1.set_xlabel('Pixels',fontsize=16)
        ax2.tick_params(axis='both',labelsize=14)
        ax2.imshow(self.binary_im_final,origin='lower')
        fig1.suptitle('Mini LSB Flame Leading Edge: {0}'.format(filename), fontsize = 20)
        fig1.set_size_inches(15,7)
        if savefig:
            fig1.savefig(filename)
            plt.close()