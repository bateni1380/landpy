from landpy import Debugger, Land

class HeatLand(Land):
    def __init__(self, debugger: Debugger = Debugger(debug=False)):
        super().__init__(debugger)

    @Debugger.method_decorator('Sending image to flow_data', ['ImageLand.set_data', 'ImageLand'])
    def set_image(self, image: np.ndarray):
        if type(image) is Image:
            image = np.array(image)

        self._debugger.log_image(
            image, 'Sent Image', categories=self._debugger.get_last_log_categories())

        if image.dtype in ('float32', 'float64'):
            image = (image*255)
        super()._set_data(image.astype('uint8'))

        return self