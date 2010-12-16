import cPickle

from input_container import *



def import_pickle_results(file):

    try:
        f = open(file, 'r')
        up = cPickle.Unpickler(f)
        adc = up.load()
    finally:
        f.close()
    return adc



def export_pickle_results(file, adc):
    try:
        f = open(file, 'w')
        p = cPickle.Pickler(f)
        p.dump(adc)
    finally:
        f.close()
