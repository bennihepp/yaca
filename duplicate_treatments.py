import sys
import os
import random
import numpy


from src.core.ioutil.import_export_hdf5 import *

option = '--random'

n = 1
if len(sys.argv) >= 4:
    option = sys.argv[1]
    n += 1

hdf5_input_file = sys.argv[n]
n += 1
hdf5_output_file = sys.argv[n]

pdc = import_hdf5_results(hdf5_input_file)


def split_random(pdc):

    new_treatments = []

    new_tr_ids = numpy.empty((pdc.objFeatures.shape[0],))

    offset = len(pdc.treatments)
    n = 0
    for tr in pdc.treatments:
        n1 = n
        n2 = n + 1
        n += 2
        new_treatment = yaca_data_treatment(tr.name + '*')
        new_treatments.append(tr)
        new_treatments.append(new_treatment)
        tr_mask = pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.rowId
        tr_size = numpy.sum(tr_mask)
        new_tr_size = tr_size / 2
        print 'splitting treatment %s (%d) into %d, %d cells' % (tr.name, tr_size, tr_size - new_tr_size, new_tr_size)
        new_tr_mask = tr_mask.copy()
        prange = numpy.arange(new_tr_mask.shape[0])
        for i in xrange(tr_size - new_tr_size):
            tmp_mask = new_tr_mask[new_tr_mask]
            index = random.randint(0, tmp_mask.shape[0] - 1)
            index2 = prange[new_tr_mask][index]
            if not new_tr_mask[index2]:
                print 'new_tr_mask[%d] = False, %d, %d' % (index2, index, tmp_mask[index])
            new_tr_mask[index2] = False
        old_tr_mask = numpy.logical_and(numpy.invert(new_tr_mask), tr_mask)
        print 'check: tr_mask=%d, new_tr_mask=%d, tr_mask^new_tr_mask=%d, tr_maskvnew_tr_mask=%d' % (numpy.sum(old_tr_mask), numpy.sum(new_tr_mask), numpy.sum(numpy.logical_and(old_tr_mask, new_tr_mask)), numpy.sum(numpy.logical_or(old_tr_mask, new_tr_mask)))
        new_tr_ids[old_tr_mask] = n1
        new_tr_ids[new_tr_mask] = n2
        tr.rowId = n1
        new_treatment.rowId = n2
        pdc.treatmentByName[tr.name] = tr.rowId
        pdc.treatmentByName[new_treatment.name] = new_treatment.rowId
    
    pdc.objFeatures[:,pdc.objTreatmentFeatureId] = new_tr_ids
    
    pdc.treatments = new_treatments
    treatmentByName = {}
    for i in xrange(len(pdc.treatments)):
        name = pdc.treatments[i].name
        treatmentByName[name] = i
    
    pdc.treatmentByName = treatmentByName
    
    """for i in xrange(pdc.objFeatures.shape[0]):
        switch = bool(random.randint(0, 1))
        if switch:
            pdc.objFeatures[i, pdc.objTreatmentFeatureId] += offset"""


def split_replicates(pdc):

    new_treatments = []
    
    new_tr_ids = numpy.empty((pdc.objFeatures.shape[0],))
    
    offset = len(pdc.treatments)
    n = 0
    for tr in pdc.treatments:
        n1 = n
        n2 = n + 1
        n += 2
        new_treatment = yaca_data_treatment(tr.name + '*')
        new_treatments.append(tr)
        new_treatments.append(new_treatment)
        tr_mask = pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.rowId
        tr_size = numpy.sum(tr_mask)

        tr_img_mask = pdc.imgFeatures[:, pdc.imgTreatmentFeatureId] == tr.rowId

        imgDict = {}
        for imgId in pdc.imgFeatures[tr_img_mask][:, pdc.imgImageFeatureId]:
            imgId = int(imgId)
            img = pdc.images[imgId]
            #imgPath = img.properties['Metadata_BaseImagePath']
            #print 'BaseImagePath:', imgPath
            imgPath = img.properties['Metadata_RelativeImagePath']
            #print 'RelativeImagePath:', imgPath
            replicateName = os.path.split(imgPath)[0]
            #print 'ReplicateName:', replicateName
            if replicateName not in imgDict:
                imgDict[replicateName] = []
            imgDict[replicateName].append(img)

        for k in imgDict.keys():
            if len(imgDict[k]) == 0:
                del imgDict[k]

        keys = imgDict.keys()
        keys1 = []
        keys2 = []
        for i in xrange(len(keys)):
            if i < len(keys) / 2:
                keys1.append(keys[i])
            else:
                keys2.append(keys[i])

        tr_mask1 = numpy.zeros((pdc.objFeatures.shape[0],), dtype=bool)
        tr_mask2 = numpy.zeros((pdc.objFeatures.shape[0],), dtype=bool)

        print 'img1'
        img1_min_id = None
        img1_max_id = None
        tr_img_size1 = 0
        for k in keys1:
            for img in imgDict[k]:
                mask = pdc.objFeatures[:, pdc.objImageFeatureId] == img.rowId
                tr_mask1 = numpy.logical_or(tr_mask1, mask)
                tr_img_size1 += 1
                if img1_min_id == None or img.rowId < img1_min_id:
                    img1_min_id = img.rowId
                if img1_max_id == None or img.rowId > img1_max_id:
                    img1_max_id = img.rowId

        print 'img1 min=%d max=%d' % (img1_min_id, img1_max_id)

        print 'img2'
        img2_min_id = None
        img2_max_id = None
        tr_img_size2 = 0
        for k in keys2:
            for img in imgDict[k]:
                mask = pdc.objFeatures[:, pdc.objImageFeatureId] == img.rowId
                tr_mask2 = numpy.logical_or(tr_mask2, mask)
                tr_img_size2 += 1
                if img2_min_id == None or img.rowId < img2_min_id:
                    img2_min_id = img.rowId
                if img2_max_id == None or img.rowId > img2_max_id:
                    img2_max_id = img.rowId

        print 'img2 min=%d max=%d' % (img2_min_id, img2_max_id)

        print 'splitting treatment %s with %d images into %d, %d images' % (tr.name, numpy.sum(tr_img_mask), tr_img_size1, tr_img_size2)
        print '  splitting treatment %s with %d cells into %d, %d cells' % (tr.name, numpy.sum(tr_mask), numpy.sum(tr_mask1), numpy.sum(tr_mask2))
        print '  ', keys1
        print '  ', keys2
        print

        new_tr_ids[tr_mask1] = n1
        new_tr_ids[tr_mask2] = n2
        tr.rowId = n1
        new_treatment.rowId = n2
        pdc.treatmentByName[tr.name] = tr.rowId
        pdc.treatmentByName[new_treatment.name] = new_treatment.rowId

    pdc.objFeatures[:,pdc.objTreatmentFeatureId] = new_tr_ids

    pdc.treatments = new_treatments
    treatmentByName = {}
    for i in xrange(len(pdc.treatments)):
        name = pdc.treatments[i].name
        treatmentByName[name] = i
    
    pdc.treatmentByName = treatmentByName
    
    """for i in xrange(pdc.objFeatures.shape[0]):
        switch = bool(random.randint(0, 1))
        if switch:
            pdc.objFeatures[i, pdc.objTreatmentFeatureId] += offset"""



if option == '--random':
    split_random(pdc)
elif option == '--replicates':
    split_replicates(pdc)
else:
    print 'Unknown option'
    sys.exit(1)


for tr in pdc.treatments:
    tr_mask = pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.rowId
    print 'treatment %s: %d cells' % (tr.name, numpy.sum(tr_mask))

export_hdf5_results(hdf5_output_file, pdc)

