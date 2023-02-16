import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

class Debugger:
    def __init__(self, categories=[], textual_debug: bool = False, visual_debug: bool = False,
                 show_file: bool = False, show_line: bool = False,
                 frame=None):
        self.__visual_debug: bool = visual_debug
        self.__textual_debug: bool = textual_debug
        self.__show_file: bool = show_file
        self.__categories: List[str] = categories
        self.__time_stack: List[Tuple[str, float]] = []
        self.__start_tabs: int = 0
        self.__frame = frame
        self.log_start('Debugger init', frame=frame)

    def __del__(self):
        self.log_end()

    def __is_in_categories(self, categories: Iterable):
        if len(categories) == 0:
            return True
        elif len(set(categories).intersection(set(self.__categories))) > 0:
            return True
        else:
            return False

    def __change_frame(self, frame=None):
        if frame is not None:
            self.__frame = frame

    def __get_line_info(self):
        if self.__frame is None:
            return ""
        file_name, line_num = self.__frame.f_code.co_filename, self.__frame.f_lineno
        result = '/{}'.format(
            file_name[file_name.rfind('\\')+1:]) if self.__show_file else ""
        result += '/line {}'.format(line_num) if self.__show_line else ""
        result = result[1:]
        if len(result) == 0:
            return ""
        return '['+result+']'

    def __get_time_info(t):
        return ('time = {:.4f}'.format(t))

    def __print(self, desc: str = '', line_info: str = '', time_info: str = ''):
        tabs = self.__start_tabs*'\t'
        print(tabs + line_info + ' ' + desc + ' ' + time_info)

    def log_start(self, desc: str, categories: List[str] = [], frame=None):
        self.__change_frame(frame)
        self.__time_stack.append((desc, time.time()))
        if self.__textual_debug and self.__is_in_categories(categories):
            self.__print(desc + ' started', self.__get_line_info())
        self.__start_tabs += 1

    def log(self, desc: str, categories: List[str] = [], frame=None):
        self.__change_frame(frame)
        last_desc, last_time = self.__time_stack[-1]
        if self.__textual_debug and self.__is_in_categories(categories):
            self.__print(desc, self.__get_line_info(),
                         Debugger.__get_time_info(time.time()-last_time))

    def log_end(self, categories: List[str] = [], frame=None):
        self.__change_frame(frame)
        if self.__start_tabs == 0:
            raise Exception('There is no proccess to end it ' +
                            self.__get_line_info())
        self.__start_tabs -= 1

        last_desc, last_time = self.__time_stack.pop()
        if self.__textual_debug and self.__is_in_categories(categories):
            self.__print(last_desc + ' ended', self.__get_line_info(),
                         Debugger.__get_time_info(time.time()-last_time))

    def log_image(self, desc: str, image: np.ndarray, categories: List[str] = [], frame=None):
        self.log(desc, categories, frame)
        if self.__visual_debug and self.__is_in_categories(categories):
            plt.imshow(image)
            plt.title(desc)
            plt.show()

    def log_images(self, descs: List[str], images: List[np.ndarray], dim: Tuple=None, categories: List[str] = [], frame=None):
        self.log('', categories, frame)
        if self.__visual_debug and self.__is_in_categories(categories):
            fig = plt.figure()
            for i in range(len(images)):
                if dim is None:
                    fig.add_subplot(1, len(images), i+1)
                else:
                    fig.add_subplot(dim[0], dim[1], i+1)
                plt.imshow(images[i])
                plt.axis('off')
                plt.title(descs[i])
            plt.show()




def main():
    debugger = Debugger(time_debug=True, show_file=False,
                        show_line=True, frame=inspect.currentframe())
    debugger.log_start('PHASE1', frame=inspect.currentframe())
    a = [i for i in range(1000000)]
    debugger.log('PHASE2', frame=inspect.currentframe())
    a = [i for i in range(1000000)]
    debugger.log('PHASE3', frame=inspect.currentframe())
    debugger.log_end(frame=inspect.currentframe())
    del debugger
