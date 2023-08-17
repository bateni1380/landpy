from typing import Callable, List, Tuple, Dict, Literal, Any, Iterable, Union
import time
from landpy import Debugger, Land
from landpy import MarksLand, QuadMarks, SquareMarks, HeatLand, Filter
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
            image, 'Set Image', categories=self._debugger.get_last_log_categories())

        if image.dtype in ('float32', 'float64'):
            image = (image*255)

        super()._set_data(image.astype('uint8'))
        if self._flow_data.shape[2] == 3:
            self.mode = 'RGB'
        else:
            self.mode = 'RGBA'
        return self

    @Debugger.method_decorator('Calling convertor.rgb_to_rgba from to_rgba method (if its an rgb image)', ['ImageLand.to_rgba', 'ImageLand'])
    def to_rgba(self, copy_trace=True):
        copied_self = self.copy(copy_trace)
        if self.mode == 'RGBA':
            self._debugger.log('Its already an rgba image!',
                               self._debugger.get_last_log_categories())
            return copied_self

        copied_self._flow_data = np.dstack((copied_self._flow_data, np.ones(
            (copied_self._flow_data.shape[0], copied_self._flow_data.shape[1]))*255))
        copied_self.mode = 'RGBA'
        return copied_self

    @Debugger.method_decorator('Calling convertor.rgba_to_rgb from to_rgb method (if its an rgba image)', ['ImageLand.to_rgb', 'ImageLand'])
    def to_rgb(self, copy_trace=True):
        copied_self = self.copy(copy_trace)
        if self.mode == 'RGB':
            self._debugger.log('Its already an rgb image!',
                               self._debugger.get_last_log_categories())
            return copied_self

        copied_self._flow_data = np.dstack((copied_self._flow_data[..., 0]*copied_self._flow_data[..., 3], copied_self._flow_data[..., 1]
                                            * copied_self._flow_data[..., 3], copied_self._flow_data[..., 2]*copied_self._flow_data[..., 3]))
        copied_self.mode = 'RGB'
        return copied_self

    @Debugger.method_decorator('Returning image!', ['ImageLand.get_image', 'ImageLand'])
    def get_image(self):
        return self._get_data()

    @Debugger.method_decorator('Showing the image', ['ImageLand.show', 'ImageLand'])
    def show(self):
        Debugger(visual_debug=True).log_image(self._flow_data)

    @Debugger.method_decorator('Rotating image', ['ImageLand.rotate', 'ImageLand'])
    def rotate(self, angle: float, center: Tuple = None, save_trace: bool = True, fill_color=(0, 0, 0)):
        height, width = self._flow_data.shape[:2]
        if center == None:
            center = (int(width/2), int(height/2))

        self._debugger.log('(angle, center) = ({}, {})'.format(
            angle, center), categories=self._debugger.get_last_log_categories())

        rotation_matrix = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1)

        self._debugger.save_image(
            self._flow_data, 'Image before rotation', categories=self._debugger.get_last_log_categories())
        self._flow_data = cv2.warpAffine(src=self._flow_data, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
        self._debugger.save_image(
            self._flow_data, 'Image after rotation', categories=self._debugger.get_last_log_categories())

        if save_trace:
            self._make_trace(ImageLand.__rotate_back, {
                             'angle': angle, 'center': center})

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        return self

    @Debugger.static_method_decorator('Rotating back image', ['ImageLand.__rotate_back', 'ImageLand'])
    def __rotate_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(angle, center) = ({}, {})'
                     .format(-pars['angle'], pars['center']), categories=debugger.get_last_log_categories())

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(
            center=pars['center'], angle=-pars['angle'], scale=1)

        debugger.save_image(
            image, 'Image before rotation back', categories=debugger.get_last_log_categories())
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=(
            width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        debugger.save_image(
            image, 'Image after rotation back', categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    @Debugger.method_decorator('Reshaping image', ['ImageLand.reshape', 'ImageLand'])
    def reshape(self, shape, mode='RGB', save_trace: bool = True):
        width, height = shape[1], shape[0]
        self._debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            width, height, mode), categories=self._debugger.get_last_log_categories())

        image = Image.fromarray(self._flow_data)

        if save_trace:
            self._make_trace(ImageLand.__reshape_back, {'width': image.width,
                                                        'height': image.height,
                                                        'mode': image.mode})

        self._debugger.save_image(
            self._flow_data, 'Image before reshape', categories=self._debugger.get_last_log_categories())
        image = image.resize((width, height))
        image = image.convert(mode)
        image = np.array(image, dtype='uint8')
        self._flow_data = image
        self._debugger.save_image(
            self._flow_data, 'Image after reshape', categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        return self

    @Debugger.static_method_decorator('Reshaping back image', ['ImageLand.__reshape_back', 'ImageLand'])
    def __reshape_back(image, pars: Dict, debugger: Debugger):
        debugger.log('(w, h, mode) = ({}, {}, {})'.format(
            pars['width'], pars['height'], pars['mode']), categories=debugger.get_last_log_categories())

        debugger.save_image(image, 'Image before reshape',
                            categories=debugger.get_last_log_categories())
        image = Image.fromarray(image)
        image = image.resize((pars['width'], pars['height']))
        image = image.convert(pars['mode'])
        image = np.array(image, dtype='uint8')
        debugger.save_image(image, 'Image after reshape',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return image

    @Debugger.method_decorator('Zooming in image', ['ImageLand.zoom', 'ImageLand'])
    def zoom(self, quadrilateral: Union[QuadMarks, SquareMarks], save_trace: bool = True, fill_color=(0, 0, 0)):
        if save_trace:
            self._make_trace(ImageLand.__zoom_back, {'last_image': self._flow_data.copy(),
                                                     'quadrilateral': quadrilateral.copy()})

        self._debugger.save_image(self._flow_data, 'Image before zoom',
                                  categories=self._debugger.get_last_log_categories())

        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * self._flow_data.shape[0])
            maxX = math.ceil(quadrilateral.maxX * self._flow_data.shape[0])
            minY = math.floor(quadrilateral.minY * self._flow_data.shape[1])
            maxY = math.ceil(quadrilateral.maxY * self._flow_data.shape[1])
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

                x_len = maxX+max(0, -minX)
                y_len = maxY+max(0, -minY)

                self._flow_data = self._flow_data[max(0, minX):max(
                    0, minX)+x_len, max(0, minY):max(0, minY)+y_len].astype('uint8')

        elif type(quadrilateral) is QuadMarks:
            self._debugger.log_exception(
                "Zooming in image using QuadMarks is not supported yet!", self._debugger.get_last_log_categories())

        self._debugger.save_image(self._flow_data, 'Image after zoom',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        return self

    @Debugger.static_method_decorator('Zooming back image', ['ImageLand.__zoom_back', 'ImageLand'])
    def __zoom_back(image, pars: Dict, debugger: Debugger):
        last_image = pars['last_image']
        quadrilateral = pars['quadrilateral']

        debugger.save_image(image, 'Image before zoom back',
                            categories=debugger.get_last_log_categories())
        if type(quadrilateral) is SquareMarks:
            minX = math.floor(quadrilateral.minX * last_image.shape[0])
            maxX = math.ceil(quadrilateral.maxX * last_image.shape[0])
            minY = math.floor(quadrilateral.minY * last_image.shape[1])
            maxY = math.ceil(quadrilateral.maxY * last_image.shape[1])

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
                "Zooming back in image using QuadMarks is not supported yet!", debugger.get_last_log_categories())

        debugger.save_image(pars['last_image'], 'Image after zoom back',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return last_image

    @Debugger.method_decorator('Applying histogram equalization method on the image', ['ImageLand.histogram_equalization', 'ImageLand'])
    def equalize_histogram(self, save_trace: bool = True):
        lab_image = cv2.cvtColor(self._flow_data, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        if save_trace:
            self._make_trace(ImageLand.__equalize_histogram_back, {
                             'last_l_channel': l_channel.copy()})

        self._debugger.save_image(self._flow_data, 'Image before h.e.',
                                  categories=self._debugger.get_last_log_categories())
        equ_l_channel = cv2.equalizeHist(l_channel)
        equ_lab_image = cv2.merge((equ_l_channel, a_channel, b_channel))
        equalized_image = cv2.cvtColor(equ_lab_image, cv2.COLOR_Lab2RGB)
        self._flow_data = equalized_image
        self._debugger.save_image(self._flow_data, 'Image after h.e.',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        return self

    @Debugger.static_method_decorator('Applying inverse of histogram equalization method on the image', ['ImageLand.__equalizatie_histogram_back', 'ImageLand'])
    def __equalize_histogram_back(image, pars: Dict, debugger: Debugger):
        last_l_channel = pars['last_l_channel']
        if last_l_channel.shape != image.shape[:2]:
            debugger.log_exception("You can\'t apply equalize_histogram_back beacuase shapes have been changed " +
                                   "from {} to {}".format(last_l_channel.shape, image.shape[:2]), debugger.get_last_log_categories())

        debugger.save_image(image, 'Image before h.e. inverse',
                            categories=debugger.get_last_log_categories())
        equ_lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        equ_l_channel, a_channel, b_channel = cv2.split(equ_lab_image)
        back_lab_image = cv2.merge((last_l_channel, a_channel, b_channel))
        back_image = cv2.cvtColor(back_lab_image, cv2.COLOR_Lab2RGB)
        debugger.save_image(back_image, 'Image after h.e. inverse',
                            categories=debugger.get_last_log_categories())

        debugger.log_saved_images(
            categories=debugger.get_last_log_categories())

        return back_image

    @Debugger.method_decorator('implanting filter in image', ['ImageLand.implant_filter', 'ImageLand'])
    def implant_filter(self, filter: Filter, heat: HeatLand = None):
        if heat is None:
            heat = full_heat(
                self._flow_data.shape[0], self._flow_data.shape[1], self._debugger)
        if heat._flow_data.shape != self._flow_data.shape[:2]:
            self._debugger.log_warning("Heat and Image weren\'t in the same size so heat is going to reshape!",
                                       self._debugger.get_last_log_categories())
            heat = heat.copy(save_trace=False).reshape(
                self._flow_data.shape[:2])

        self._debugger.save_image(self._flow_data, 'Image before implant',
                                  categories=self._debugger.get_last_log_categories())

        self._flow_data = filter.apply_function(
            self._flow_data, np.dstack([heat._flow_data]*3)).astype(int)

        self._debugger.save_image(self._flow_data, 'Image after implant',
                                  categories=self._debugger.get_last_log_categories())

        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())

        return self

    @Debugger.method_decorator('implanting image in image', ['ImageLand.implant_image', 'ImageLand'])
    def implant_image(self, plant_img: np.ndarray, src_land: SquareMarks = SquareMarks(), plant_land: SquareMarks = SquareMarks(), ratio=0.5):
        if self.mode == 'RGBA':
            self = self.to_rgb()
        if plant_img.shape[2] == 3:
            plant_img = ImageLand(plant_img).to_rgba()._flow_data
        src_img = np.array(self._flow_data/255, dtype=float)
        plant_img = plant_img/255

        if type(src_land) is SquareMarks and type(plant_land) is SquareMarks:
            dst_pts = src_land._flow_data * \
                np.array([src_img.shape[0],
                         src_img.shape[1]], dtype=float)
            src_pts = plant_land._flow_data * \
                np.array([plant_img.shape[0], plant_img.shape[1]], dtype=float)
            src_pts, dst_pts = np.floor(src_pts, dtype=np.float32), np.floor(dst_pts, dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            transformed_img = cv2.warpPerspective(
                plant_img, M, (src_img.shape[1], src_img.shape[0]))
            
            self._debugger.save_image(
                transformed_img, 'transformed image', self._debugger.get_last_log_categories())
        else:
            self._debugger.log_exception(
                'other implant types are not supported yet!', self._debugger.get_last_log_categories())

        p = np.dstack([transformed_img[..., 3]]*3)
        
        overlay = (((1-ratio*p)*src_img[..., 0:3]+ratio*p*(transformed_img[..., 0:3])))
        self._debugger.save_image(
            overlay, 'overlay', self._debugger.get_last_log_categories())
        self._debugger.log_saved_images(
            categories=self._debugger.get_last_log_categories())
        self.set_image(overlay)
        return self
