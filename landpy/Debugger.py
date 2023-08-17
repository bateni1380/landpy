import inspect
import time
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, AnyStr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

class Debugger:
    '''
        This class is a class to observe status of the code. 

        You can create an instance of this class by (debugger = Debugger())
            and whenever you want to print something to see the status of your code, 
            you can use debugger.log(description: str) to print status.

        This class provides you many tools in order to observe the status of you're program better.
        - 1st, you can print all of your debugging-related logs with debugger.log() and
            you have the ability to turn on/of all of the debugging-related logs by changing
            textual_debug/visual_debug in the line that you instantiated the Debugger object.
        - 2nd, you can see the line/file that the log function has been called by setting
            show_file/show_line parameter's to True in the line that you instantiated the 
            Debugger object.
        - 3rd, you can specify categories of each log and categorize log's, then you can 
            change categories parameter in constructor to see the related logs.
        - 4th, you can specify processes using log_start and log_end, and then you can
            see the passed time of each process in logs. If you use log_end or log, you will see 
            the passed time since last non finished process has started.
        - 5th, you can ilustrate np array images using log_image (for one image) and 
            log_images (for multiple images).

    '''

    def __init__(self, categories=[], textual_debug: bool = False, visual_debug: bool = False,
                 debug=True, show_file: bool = False, show_line: bool = False):
        '''
            - categories parameter is the tool to categorize your debugging-related logs. you can
                specify some categories here and in log function you can specify some categories too,
                then the log will be printed only if this two set's have some elements in common.
            - debug parameter is the tool to turn of/on all of the logs at one shot.
            - textual_debug parameter is the tool to turn of/on all not visual logs at one shot.
            - visual_debug parameter is the tool to turn of/on all visual logs at one shot.
            - show_file is the tool to show the file that the log funtion is running at in each log.
            - show_line is the tool to show the line that the log funtion is running at in each log.

        '''
        self.__dubug = debug
        if not self.__dubug:
            return
        self.__visual_debug: bool = visual_debug
        self.__textual_debug: bool = textual_debug
        self.__show_file: bool = show_file
        self.__show_line: bool = show_line
        self.__categories: Iterable[str] = categories
        self.__time_stack: List[Tuple[str, float]] = []
        self.__start_tabs: int = 0
        self.__images_to_show = []
        self.__scatters_to_show = []
        self.__heats_to_show = []
        self.__last_log_categories = [[]]
        self.__set_frame()
        self.__log_start('Debugger init', ['Debugger.__init__', 'Debugger'])

    def __del__(self):
        if not self.__dubug:
            return
        if self.__start_tabs > 0:
            self.__set_frame()
            self.__log_end()

    def append_last_log_categories(self, last_log_categories):
        if not self.__dubug:
            return
        self.__last_log_categories.append(last_log_categories.copy())
        
    def get_last_log_categories(self):
        if not self.__dubug:
            return []
        return self.__last_log_categories[-1]
    
    def pop_last_log_categories(self):
        if not self.__dubug:
            return
        self.__last_log_categories.pop()

    def log(self, desc: str, categories: Iterable[str] = []):
        '''
            You can use this function to log an event in console.
            You just have two parameters here. desc is the description
                of your event and categories is the list of events, that
                this log is related to (then you can set categories parameter
                in constructor to see every log from some specific categories)
            - desc is the description of the log.
            - categories is a list or set of some strings that specifies the
                categories that this log belong's to them.
        '''
        if not self.__dubug:
            return

        if self.__textual_debug and self.__is_in_categories(categories):
            self.__set_frame()
            _, last_time = self.__time_stack[-1]
            self.__print(desc, self.__get_line_info(),
                         Debugger.__get_time_info(time.time()-last_time))

    def log_start(self, desc: str, categories: Iterable[str] = []):
        '''
            This funtion is like a simple log but here, you are specifying that
                some process has been started. Then you see passed time of all logs after 
                that, with respect to this log.
            - desc is the description of the log.
            - categories is like categories in log function.

        '''
        if not self.__dubug:
            return

        self.__set_frame()
        self.__log_start(desc, categories)

    def log_end(self, categories: Iterable[str] = []):
        '''
            This funtion is like a simple log but here, you are specifying that
                last process has been ended. We don't have any description as 
                you see the description of the process that have been ended.
            - categories is like categories in log function.
        '''
        if not self.__dubug:
            return

        self.__set_frame()
        self.__log_end(categories)

    def log_exception(self, desc: str, categories: Iterable[str] = []):
        print('\033[91', end='')
        self.log(desc, ["Exception"]+categories)
        print('\033[0m', end='')
        raise Exception(desc)
    
    def log_warning(self, desc: str, categories: Iterable[str] = []):
        print('\033[93m', end='')
        self.log(desc, ["Warning"]+categories)
        print('\033[0m', end='')
    
    class LogTypes:
        IMAGE = 0
        HEAT = 1
        SCATTER = 2

    def log_images(self, images, descs: List[str], types, dim: Tuple = None, categories: Iterable[str] = []):
        '''
            This funtion is like a simple log but here, you feed some pictures as
                a list of np arrays.
            - images is the list of np arrays.
            - descs is title of those images.
            - dim is dimentions of the grid that you want to see the pictures in it (by default,
                log_images show the images in a 1-row grid)
            - categories is like categories in log function.
        '''
        if not self.__dubug:
            return

        if self.__visual_debug and self.__is_in_categories(categories):
            fig = plt.figure()
            for i in range(len(images)):
                if dim is None:
                    fig.add_subplot(1, len(images), i+1)
                else:
                    fig.add_subplot(dim[0], dim[1], i+1)

                if types[i] == Debugger.LogTypes.IMAGE:
                    plt.imshow(images[i])
                    plt.axis('image')
                elif types[i] == Debugger.LogTypes.SCATTER:
                    rect = patches.Rectangle(
                    (0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                    if not plt.gca().yaxis_inverted():
                        plt.gca().invert_yaxis()
                    plt.gca().add_patch(rect)
                    plt.scatter(images[i][:, 0], images[i][:, 1])
                    plt.axis('equal')
                elif types[i] == Debugger.LogTypes.HEAT:
                    plt.imshow(images[i], cmap='YlOrRd', vmin=0.0, vmax=1.0)
                    plt.colorbar()

                if descs[i] is not None and len(descs[i]) > 0:
                    plt.title(descs[i])

            plt.show()

    def log_image(self, image: np.ndarray, desc: str = "", type = 0, categories: Iterable[str] = []):
        '''
            This funtion is like a simple log but here, you show some picture too.
            - desc is the title of the image.
            - image is the image that you want to show as a np array format.
            - categories is like categories in log function.
        '''
        if not self.__dubug:
            return

        if self.__visual_debug and self.__is_in_categories(categories):
            self.log_images([image], [desc], [type], None, categories)

    def save_image(self, image: np.ndarray, desc: str = "", type = 0, categories: Iterable[str] = []):
        if not self.__dubug:
            return

        if self.__visual_debug and self.__is_in_categories(categories):
            self.log(desc+' saved!', categories)
            self.__images_to_show.append({'desc': desc, 'image': image.copy(), 'type': int(type)})

    def log_saved_images(self, dim: Tuple = None, categories: Iterable[str] = []):
        if not self.__dubug:
            return

        if self.__visual_debug and self.__is_in_categories(categories):
            images = [self.__images_to_show[i]['image']
                      for i in range(len(self.__images_to_show))]
            descs = [self.__images_to_show[i]['desc']
                     for i in range(len(self.__images_to_show))]
            types = [self.__images_to_show[i]['type']
                     for i in range(len(self.__images_to_show))]
            self.log_images(images, descs, types, dim, categories)
            for i in range(len(self.__images_to_show)):
                del self.__images_to_show[i]['image']
            self.__images_to_show = []

    def __set_frame(self):
        if self.__show_file or self.__show_line:
            self.__frame = inspect.getouterframes(
                inspect.currentframe())[2].frame

    def __is_in_categories(self, categories: Iterable):
        if categories is None or len(categories) == 0 or self.__categories is None or len(self.__categories) == 0:
            return True
        elif len(set(categories).intersection(set(self.__categories))) > 0:
            return True
        else:
            return False

    def __get_line_info(self):
        result = ''
        if self.__show_file:
            file_name = self.__frame.f_code.co_filename
            result += '/{}'.format(file_name[file_name.rfind('\\')+1:])

        if self.__show_line:
            line_num = self.__frame.f_lineno
            result += '/line {}'.format(line_num)

        result = result[1:]
        if len(result) == 0:
            return ""
        return '['+result+']'

    def __get_time_info(t):
        return ('time = {:.4f}'.format(t))

    def __print(self, desc: str = '', line_info: str = '', time_info: str = ''):
        tabs = self.__start_tabs*'\t'
        print(tabs + line_info + ' ' + desc + ' ' + time_info)

    def __log_start(self, desc: str, categories: Iterable[str] = []):
        self.__time_stack.append((desc, time.time()))
        if self.__textual_debug and self.__is_in_categories(categories):
            self.__print(desc + ' started', self.__get_line_info())
            self.__start_tabs += 1

    def __log_end(self, categories: Iterable[str] = []):
        last_desc, last_time = self.__time_stack.pop()
        if self.__textual_debug and self.__is_in_categories(categories):
            if self.__start_tabs == 0:
                raise Exception('There is no proccess to end it ' + self.__get_line_info())
            self.__start_tabs -= 1
            self.__print(last_desc + ' ended', self.__get_line_info(),
                         Debugger.__get_time_info(time.time()-last_time))

    def __modify_method_args(func, self, func_args, func_kargs):
        # Get the original function's signature and bind any default values
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(self, *func_args, **func_kargs)
        for param_name, param in sig.parameters.items():
            if param_name not in bound_args.arguments and param.default is not inspect.Parameter.empty:
                bound_args.arguments[param_name] = param.default
        return bound_args
    
    def __modify_static_method_args(func, func_args, func_kargs):
        # Get the original function's signature and bind any default values
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*func_args, **func_kargs)
        for param_name, param in sig.parameters.items():
            if param_name not in bound_args.arguments and param.default is not inspect.Parameter.empty:
                bound_args.arguments[param_name] = param.default
        return bound_args
                
    def method_decorator(desc: str, log_categories: List):
        def dec_wrapper(func):
            def func_wrapper(self, *func_args, **func_kargs):
                bound_args = Debugger.__modify_method_args(func, self, func_args, func_kargs)

                self._debugger.append_last_log_categories(log_categories)
                self._debugger.log_start(desc, categories=self._debugger.get_last_log_categories())
                result = func(**bound_args.arguments)
                self._debugger.log_end(categories=self._debugger.get_last_log_categories())
                self._debugger.pop_last_log_categories()

                return result
            return func_wrapper
        return dec_wrapper

    
    def static_method_decorator(desc: str, log_categories: List):
        def dec_wrapper(func):
            def func_wrapper(*func_args, **func_kargs):
                bound_args =Debugger.__modify_static_method_args(func, func_args, func_kargs)

                debugger = None
                if 'debugger' in func_kargs:
                    debugger = func_kargs['debugger']
                elif type(func_args[-1]) is Debugger:
                    debugger = func_args[-1]
                else:
                    raise Exception('static_method_decorator expects a debugger object as last parameter.')
                debugger.append_last_log_categories(log_categories)
                debugger.log_start(desc, categories=debugger.get_last_log_categories())
                result = func(**bound_args.arguments)
                debugger.log_end(categories=debugger.get_last_log_categories())
                debugger.pop_last_log_categories()

                return result
            return func_wrapper
        return dec_wrapper
    