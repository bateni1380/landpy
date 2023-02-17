from Debugger import Debugger
import inspect


def main():
    debugger = Debugger(textual_debug=True, show_file=False, show_line=True)
    debugger.log_start('PHASE1')
    a = [i for i in range(1000000)]
    debugger.log('PHASE2')      
    a = [i for i in range(1000000)]
    debugger.log('PHASE3')
    debugger.log_end()
    del debugger
