from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
import time
from landpy import Debugger, Land
from landpy import MarksLand, QuadMarks, SquareMarks, DualMarks, HeatLand, Filter
from landpy.heat import zero_heat, full_heat
import numpy as np
import cv2
import inspect
import matplotlib.pyplot as plt
import math
from PIL import Image


class ImageLand(Land):
    def __init__(self, image=None, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)
        if type(image) is str:
            image = np.array(Image.open(image))
            ImageLand.set_image(self, image)
        elif image is not None:
            ImageLand.set_image(self, image)

    @Debugger.method_decorator('Setting image', ['ImageLand.set_data', 'ImageLand'])
    def set_image(self, image: np.ndarray):
        if type(image) is Image:
            image = np.array(image)

        self._debugger.log_image(
            image, 'Set Image', )

        if image.dtype in ('float32', 'float64'):
            image = (image*255)

        super()._set_data(image.astype('uint8'))
        if self._flow_data.shape[2] == 3:
            self.mode = 'RGB'
        else:
            self.mode = 'RGBA'
        return self

    @Debugger.method_decorator('Copying image', ['ImageLand.copy', 'ImageLand'])
    def copy(self, copy_trace=False):
        copied_self = Land.copy(self, copy_trace)
        copied_self.mode = self.mode
        return copied_self

    @Debugger.method_decorator('Changing mode to rgba', ['ImageLand.to_rgba', 'ImageLand'])
    def to_rgba(self):
        if self.mode == 'RGBA':
            self._debugger.log_warning('Its already an rgba image!')
            return self

        self._flow_data = np.dstack((self._flow_data, np.ones(
            (self._flow_data.shape[0], self._flow_data.shape[1]))*255))
        self.mode = 'RGBA'
        return self

    @Debugger.method_decorator('Changing mode to rgb', ['ImageLand.to_rgb', 'ImageLand'])
    def to_rgb(self):
        if self.mode == 'RGB':
            self._debugger.log_warning('Its already an rgb image!')
            return self

        self._flow_data = np.dstack((self._flow_data[..., 0]*self._flow_data[..., 3], self._flow_data[..., 1]
                                     * self._flow_data[..., 3], self._flow_data[..., 2]*self._flow_data[..., 3]))
        self.mode = 'RGB'
        return self

    @Debugger.method_decorator('Returning image!', ['ImageLand.get_image', 'ImageLand'])
    def get_image(self):
        return self._get_data()

    @Debugger.method_decorator('Showing the image', ['ImageLand.show', 'ImageLand'])
    def show(self, desc=''):
        Debugger(visual_debug=True).log_image(self._flow_data, desc=desc)
        return self

    @Debugger.method_decorator('Rotating image', ['ImageLand.rotate', 'ImageLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace=False, fill_color=(0, 0, 0)):
        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), )
        
        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(
            self._flow_data, 'Image before rotation', )
        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
        self._debugger.save_image(
            self._flow_data, 'Image after rotation', )

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_images(
        )

        return self

    @Debugger.static_method_decorator('Rotating back image', ['ImageLand.__rotate_back', 'ImageLand'])
    def __rotate_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(angle, center) = ({}, {})'
                     .format(-pars['angle'], pars['center']), )

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)

        debugger.save_image(
            image, 'Image before rotation back', )
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(
            image, 'Image after rotation back', )

        debugger.log_saved_images(
        )

        return image

    @Debugger.method_decorator('Reshaping image', ['ImageLand.reshape', 'ImageLand'])
    def reshape(self, shape, mode='RGB', save_trace=False):
        width, height = shape[1], shape[0]
        self._debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            width, height, mode), )

        image = Image.fromarray(self._flow_data)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'width': image.width,
                                                        'height': image.height,
                                                        'mode': image.mode})

        self._debugger.save_image(
            self._flow_data, 'Image before reshape', )
        image = image.resize((width, height))
        image = image.convert(mode)
        image = np.array(image, dtype='uint8')
        self._flow_data = image
        self._debugger.save_image(
            self._flow_data, 'Image after reshape', )

        self._debugger.log_saved_images(
        )

        return self

    @Debugger.static_method_decorator('Reshaping back image', ['ImageLand.__reshape_back', 'ImageLand'])
    def __reshape_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            pars['width'], pars['height'], pars['mode']), )

        debugger.save_image(image, 'Image before reshape',
                            )
        image = Image.fromarray(image)
        image = image.resize((pars['width'], pars['height']))
        image = image.convert(pars['mode'])
        image = np.array(image, dtype='uint8')
        debugger.save_image(image, 'Image after reshape',
                            )

        debugger.log_saved_images(
        )

        return image

    @Debugger.method_decorator('Zooming in image', ['ImageLand.zoom', 'ImageLand'])
    def zoom(self, quadrilateral=None, zoom_factor=None, center=None, save_trace=False, fill_color=(0, 0, 0)):
        if quadrilateral is None:
            quadrilateral = SquareMarks(shape=self._flow_data.shape[:2], zoom_factor=zoom_factor, center=center)

        if save_trace:
            self._make_trace(ImageLand.__zoom_back, {'last_image': self._flow_data.copy(),
                                                     'quadrilateral': quadrilateral.copy(copy_trace=False)})

        self._debugger.save_image(self._flow_data, 'Image before zoom',
                                  )

        if type(quadrilateral) is SquareMarks:
            minX = int(quadrilateral.minX())
            maxX = int(quadrilateral.maxX())
            minY = int(quadrilateral.minY())
            maxY = int(quadrilateral.maxY())
            if minX >= 0 and maxX <= self._flow_data.shape[0] and minY >= 0 and maxY <= self._flow_data.shape[1]:
                self._flow_data = self._flow_data[minX:maxX, minY:maxY]
            else:
                def genX(x):
                    return np.ones([x, self._flow_data.shape[1], self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.vstack(
                    [genX(max(0, -minX)), self._flow_data, genX(max(0, maxX-self._flow_data.shape[0]))])

                def genY(x):
                    return np.ones([self._flow_data.shape[0], x, self._flow_data.shape[2]], dtype='uint8')*fill_color
                self._flow_data = np.hstack(
                    [genY(max(0, -minY)), self._flow_data, genY(max(0, maxY-self._flow_data.shape[1]))])

                x_len = maxX-max(0, minX)+max(0, -minX)
                y_len = maxY-max(0, minY)+max(0, -minY)

                self._flow_data = self._flow_data[max(0, minX):max(
                    0, minX)+x_len, max(0, minY):max(0, minY)+y_len].astype('uint8')

        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception(
                "Zooming in image using QuadMarks is not supported yet!")

        self._debugger.save_image(self._flow_data, 'Image after zoom')

        self._debugger.log_saved_images()

        return self

    @Debugger.static_method_decorator('Zooming back image', ['ImageLand.__zoom_back', 'ImageLand'])
    def __zoom_back(image, pars: Dict, debugger: Debugger):
        last_image = pars['last_image']
        quadrilateral = pars['quadrilateral']

        debugger.save_image(image, 'Image before zoom back',
                            )
        if type(quadrilateral) is SquareMarks:
            minX = quadrilateral.minX()
            maxX = quadrilateral.maxX()
            minY = quadrilateral.minY()
            maxY = quadrilateral.maxY()

            x_len = min(maxX, last_image.shape[0])-max(0, minX)
            y_len = min(maxY, last_image.shape[1])-max(0, minY)

            if last_image.shape[2] == 4:
                mask = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len, 3] == 0
                mask = np.stack([mask, mask, mask, mask], axis=2)
                last_image[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len] = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]*(1-mask) + last_image[max(
                        0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len]*(mask)
            else:
                last_image[max(0, minX):max(0, minX)+x_len, max(0, minY):max(0, minY)+y_len] = image[max(
                    0, -minX):max(0, -minX)+x_len, max(0, -minY):max(0, -minY)+y_len]

        elif type(quadrilateral) is QuadMarks:
            debugger.log_exception(
                "Zooming back in image using QuadMarks is not supported yet!")

        debugger.save_image(pars['last_image'], 'Image after zoom back')
        debugger.log_saved_images()

        return last_image

    @Debugger.method_decorator('Applying histogram equalization method on the image', ['ImageLand.histogram_equalization', 'ImageLand'])
    def equalize_histogram(self, save_trace=False):
        lab_image = cv2.cvtColor(self._flow_data, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        if save_trace:
            self._make_trace(ImageLand.__equalize_histogram_back, {
                             'last_l_channel': l_channel.copy()})

        self._debugger.save_image(self._flow_data, 'Image before h.e.',
                                  )
        equ_l_channel = cv2.equalizeHist(l_channel)
        equ_lab_image = cv2.merge((equ_l_channel, a_channel, b_channel))
        equalized_image = cv2.cvtColor(equ_lab_image, cv2.COLOR_Lab2RGB)
        self._flow_data = equalized_image
        self._debugger.save_image(self._flow_data, 'Image after h.e.',
                                  )

        self._debugger.log_saved_images(
        )

        return self

    @Debugger.static_method_decorator('Applying inverse of histogram equalization method on the image', ['ImageLand.__equalizatie_histogram_back', 'ImageLand'])
    def __equalize_histogram_back(image, pars: Dict, debugger: Debugger):
        last_l_channel = pars['last_l_channel']
        if last_l_channel.shape != image.shape[:2]:
            debugger.log_exception("You can\'t apply equalize_histogram_back beacuase shapes have been changed " +
                                   "from {} to {}".format(last_l_channel.shape, image.shape[:2]))

        debugger.save_image(image, 'Image before h.e. inverse')
        equ_lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        equ_l_channel, a_channel, b_channel = cv2.split(equ_lab_image)
        back_lab_image = cv2.merge((last_l_channel, a_channel, b_channel))
        back_image = cv2.cvtColor(back_lab_image, cv2.COLOR_Lab2RGB)
        debugger.save_image(back_image, 'Image after h.e. inverse')

        debugger.log_saved_images()

        return back_image

    @Debugger.method_decorator('implanting filter in image', ['ImageLand.implant_filter', 'ImageLand'])
    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        ImageLand.__save_plot(self, 'base', self._debugger)
        ImageLand.__save_plot(heat, 'heat', self._debugger)

        if heat is None:
            heat = full_heat(
                self._flow_data.shape[0], self._flow_data.shape[1], self._debugger)
        if heat._flow_data.shape != self._flow_data.shape[:2]:
            self._debugger.log_warning(
                "Heat and Image weren\'t in the same size so heat is going to reshape!")
            heat = heat.copy(copy_trace=False).reshape(
                self._flow_data.shape[:2])

        self._flow_data = filter.apply_function(
            self._flow_data, np.dstack([heat._flow_data]*3)).astype('uint8')

        ImageLand.__save_plot(self, 'result', self._debugger)

        self._debugger.log_saved_images()

        return self

    @Debugger.method_decorator('implanting image in image', ['ImageLand.implant_image', 'ImageLand'])
    def implant_image(self, plant_img: np.ndarray, src_land: SquareMarks = None, plant_land: SquareMarks = None, ratio=0.5):
        ImageLand.__save_plot(self, 'base', self._debugger)
        
        if src_land is None:
            src_land = SquareMarks(shape=self._flow_data.shape[:2])
        if plant_land is None:
            plant_land = SquareMarks(shape=plant_img.shape[:2])
        
        self._debugger.save_image(
            plant_img, 'plant image')

        if self.mode == 'RGBA':
            self = self.to_rgb()
        if plant_img.shape[2] == 3:
            plant_img = ImageLand(plant_img).to_rgba()._flow_data
        src_img = np.array(self._flow_data/255, dtype=float)
        plant_img = plant_img/255

        if type(src_land) is SquareMarks and type(plant_land) is SquareMarks:
            M = cv2.getPerspectiveTransform(plant_land._flow_data.astype(np.float32), src_land._flow_data.astype(np.float32))
            transformed_img = cv2.warpPerspective(
                plant_img, M, (src_img.shape[1], src_img.shape[0]))

            self._debugger.save_image(
                transformed_img, 'transformed image')
        else:
            self._debugger.log_exception(
                'other implant types are not supported yet!')

        p = np.dstack([transformed_img[..., 3]]*3)

        overlay = (
            ((1-ratio*p)*src_img[..., 0:3]+ratio*p*(transformed_img[..., 0:3])))
        self._debugger.save_image(
            overlay, 'overlay')
        self._debugger.log_saved_images()
        self.set_image(overlay)
        return self

    def __save_plot(object, desc, debugger: Debugger):
        if type(object) is ImageLand:
            debugger.save_image(object._flow_data, desc,
                                Debugger.LogTypes.IMAGE)
        elif type(object) is MarksLand or type(object) is SquareMarks or type(object) is QuadMarks:
            debugger.save_image(object._flow_data, desc,
                                Debugger.LogTypes.SCATTER)
        elif type(object) is HeatLand:
            debugger.save_image(object._flow_data, desc,
                                Debugger.LogTypes.HEAT)

    ADDED_POINTS_RADIOUS = 1.5
    ADDED_POINTS_COLOR = (255, 0, 0)
    ADDED_LINES_WIDTH = 2
    ADDED_LINES_COLOR = (255,255,0)
    ADDED_VECTORS_WIDTH = 1
    ADDED_VECTORS_COLOR = (255,255,255)
    @Debugger.method_decorator('Adding an image to another object (image or heat or marks)', ['ImageLand.__add', 'ImageLand'])
    def __add(self, object2):
        object1_c = self.copy(copy_trace=True)
        object2_c = object2.copy(copy_trace=False)
        if type(object2) is ImageLand:
            if object1_c._flow_data.shape != object2_c._flow_data.shape:
                self._debugger.log_warning(
                    "The images weren\'t in the same size so the second image is going to reshape!")
                object2_c.reshape(object1_c._flow_data)
                if object1_c.mode == 'RGB':
                    object2_c = object2_c.to_rgb(save_trace=False)
                else:
                    object2_c = object2_c.to_rgba(save_trace=False)

            object1_c._flow_data = ((object1_c._flow_data.astype(
                'uint16')+object2_c._flow_data.astype('uint16'))/2).astype('uint8')

        elif isinstance(object2, MarksLand):
            if object2.shape() != object1_c._flow_data.shape and object2.shape() is not None:
                self._debugger.log_warning(
                    "The image and marksland weren\'t in the same size so the marksland is going to reshape!")
                object2.reshape(object1_c._flow_data.shape[:2])
                
            points = object2._flow_data[0][:,::-1].astype(int)
            dot_radius = int(min(self._flow_data.shape[0], self._flow_data.shape[1])*ImageLand.ADDED_POINTS_RADIOUS*0.01)
            for point in points:
                cv2.circle(object1_c._flow_data, point.tolist(), dot_radius, ImageLand.ADDED_POINTS_COLOR, -1)
            line_width = max(int(min(self._flow_data[0].shape[0], self._flow_data[0].shape[1])*ImageLand.ADDED_LINES_WIDTH), 1)
            if isinstance(object2, SquareMarks):
                #minX, maxX, minY, maxY = int(object2.minX()*object1_c._flow_data.shape[0]), int(object2.maxX()*object1_c._flow_data.shape[0]), int(object2.minY()*object1_c._flow_data.shape[1]), int(object2.maxY()*object1_c._flow_data.shape[1])
                cv2.line(object1_c._flow_data, points[0].tolist(), points[1].tolist(), ImageLand.ADDED_LINES_COLOR, line_width)
                cv2.line(object1_c._flow_data, points[1].tolist(), points[2].tolist(), ImageLand.ADDED_LINES_COLOR, line_width)
                cv2.line(object1_c._flow_data, points[2].tolist(), points[3].tolist(), ImageLand.ADDED_LINES_COLOR, line_width)
                cv2.line(object1_c._flow_data, points[3].tolist(), points[0].tolist(), ImageLand.ADDED_LINES_COLOR, line_width)

            if isinstance(object2, DualMarks):
                start_point = int(object2._flow_data[0][0][1]), int(object2._flow_data[0][0][0])
                end_point   = int(object2._flow_data[0][1][1]), int(object2._flow_data[0][1][0])
                line_width = max(int(min(self._flow_data.shape[0], self._flow_data.shape[1])*0.01*ImageLand.ADDED_VECTORS_WIDTH), 1)
                distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) 
                cv2.arrowedLine(object1_c._flow_data, start_point, end_point, color=ImageLand.ADDED_VECTORS_COLOR, thickness=line_width, tipLength=float(distance)/2e3)

        if type(object2) is HeatLand:
            if object2_c._flow_data.shape != object1_c._flow_data.shape[:2]:
                self._debugger.log_warning(
                    "Heat and Image weren\'t in the same size so heat is going to reshape!")
                object2_c = object2_c.reshape(object1_c._flow_data.shape[:2])

            object1_c._flow_data = Filter((255, 165, 0), 0.5, self._debugger).apply_function(
                object1_c._flow_data, np.dstack([object2_c._flow_data]*3)).astype('uint8')

        return object1_c

    @Debugger.method_decorator('Adding an image to another object (image or heat or marks)', ['ImageLand.__add__', 'ImageLand'])
    def __add__(self, other):
        ImageLand.__save_plot(
            self, 'Left', self._debugger)
        ImageLand.__save_plot(
            other, 'Right', self._debugger)

        result = self.__add(other)

        ImageLand.__save_plot(result, 'Result', self._debugger)
        self._debugger.log_saved_images()
        return result

    @Debugger.method_decorator('Adding an image to another object (image or heat or marks)', ['ImageLand.__radd__', 'ImageLand'])
    def __radd__(self, other):
        ImageLand.__save_plot(
            other, 'Left', self._debugger)
        ImageLand.__save_plot(
            self, 'Right', self._debugger)

        result = other.__add(self)

        ImageLand.__save_plot(result, 'Result', self._debugger)
        self._debugger.log_saved_images()
        return result
