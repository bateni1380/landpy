import inspect

filename = inspect.currentframe().f_code.co_filename
linenum = inspect.currentframe().f_lineno
print('file: ' + filename[filename.rfind('\\')+1:] + ' , line: '+ str(linenum))