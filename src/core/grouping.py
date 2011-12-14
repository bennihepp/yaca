# -*- coding: utf-8 -*-

"""
grouping.py -- Helpers for organizing cell objects into various groups.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import itertools
import functools
from cStringIO import StringIO
import numpy

import importer
import parameter_utils as utils


custom_get_treatment_groups = None
custom_get_well_groups = None
custom_get_replicate_groups = None
custom_get_treatment_replicate_groups = None

default_get_treatment_groups_src = \
"""for tr in pdc.treatments:
    if tr.name not in reject_treatments:
        tr_mask = pdc.objFeatures[:,pdc.objTreatmentFeatureId][mask] == tr.index
        masks.append(tr_mask)
        names.append(tr.name)
"""
default_get_well_groups_src = \
"""for well in pdc.wells:
    well_mask = pdc.objFeatures[:,pdc.objWellFeatureId][mask] == well.index
    masks.append(well_mask)
    names.append(well.name)
"""
default_get_replicate_groups_src = \
"""for repl in pdc.replicates:
    repl_mask = pdc.objFeatures[:,pdc.objReplicateFeatureId][mask] == repl.index
    masks.append(repl_mask)
    names.append('[%d]' % repl.index)
"""
default_get_treatment_replicate_groups_src = \
"""for tr in pdc.treatments:
       tr_mask = pdc.objFeatures[:,pdc.objTreatmentFeatureId][mask] == tr.index
       subnames = []
       submasks = []
       for repl in pdc.replicates:
           if (reject_treatments == None) or (tr.name not in reject_treatments):
               repl_mask = pdc.objFeatures[:,pdc.objReplicateFeatureId][mask] == repl.index
               tr_repl_mask = numpy.logical_and(tr_mask, repl_mask)
               repl_name = '%d' % repl.index
               #name = '%s,[%d]' % (tr.name, repl.index)
               subnames.append(repl_name)
               submasks.append(tr_repl_mask)
       subgroups = zip(subnames, submasks)
       groups.append([tr.name, subgroups])
"""

# define necessary parameters for this module (see parameter_utils.py for details)
#
#
__dict__ = sys.modules[ __name__ ].__dict__
#
utils.register_module( __name__, 'Grouping of cell objects', __dict__ )
#
utils.add_required_state( __name__, importer.__name__, 'imported' )
#
utils.register_parameter(__name__, 'custom_get_treatment_groups_src', utils.PARAM_LONGSTR, 'Function defining the grouping of cell objects by treatment', param_default=default_get_treatment_groups_src)
#
utils.register_parameter(__name__, 'custom_get_well_groups_src', utils.PARAM_LONGSTR, 'Function defining the grouping of cell objects by well', param_default=default_get_well_groups_src)
#
utils.register_parameter(__name__, 'custom_get_replicate_groups_src', utils.PARAM_LONGSTR, 'Function defining the grouping of cell objects by replicate', param_default=default_get_replicate_groups_src)
#
utils.register_parameter(__name__, 'custom_get_treatment_replicate_groups_src', utils.PARAM_LONGSTR, 'Function defining the grouping of cell objects by treatment and replicate', param_default=default_get_treatment_replicate_groups_src)
#
#utils.register_parameter(__name__, 'custom_get_groups_dict_src', utils.PARAM_DICT, 'Dictionary defining custom groupings of cell objects', param_default={}, hidden=True)
#
utils.register_parameter(__name__, 'reject_treatments', utils.PARAM_TREATMENTS, 'Treatments not to be used', param_default=[])


custom_get_groups_dict = {}
custom_get_groups_dict_src = {}

__SOURCE_TEMPLATE_PRE = """
def __temp_function(pdc, mask=None, flat=True):
    if mask == None:
        mask = slice(0, pdc.objFeatures.shape[0])
    # either names and masks ...
    names = []
    masks = []
    # ... or groups should be used
    groups = []
"""
__SOURCE_TEMPLATE_POST = """
    if len(groups) <= 0:
        groups = zip(names, masks)
    if flat:
        groups = flatten_groups(groups)
    return groups
"""

def compile_get_groups_function(source):
    global __SOURCE_TEMPLATE_PRE, __SOURCE_TEMPLATE_POST
    new_source = StringIO()
    for line in source.strip().split('\n'):
        new_source.write('    %s\n' % line)
    source = new_source.getvalue()
    source = __SOURCE_TEMPLATE_PRE + source + __SOURCE_TEMPLATE_POST
    exec source
    return __temp_function

def update_custom_get_groups_functions():
    global custom_get_groups_dict
    for group_description in custom_get_groups_dict_src.keys():
        if custom_get_groups_dict_src[group_description].strip():
            custom_get_groups_dict[group_description] = compile_get_groups_function(
                custom_get_groups_dict_src[group_description]
            )

def get_available_group_descriptions():
    update_custom_get_groups_functions()
    global custom_get_groups_dict
    group_descriptions = ['treatment', 'replicate', 'well', 'treatment+replicate']
    group_descriptions.extend(custom_get_groups_dict.keys())
    return group_descriptions

def join_groups(groups):
    if len(groups) <= 0:
        return None
    groups = flatten_groups(groups)
    join_mask = None
    for name, mask in groups:
        if join_mask == None:
            join_mask = mask
        else:
            join_mask = numpy.logical_or(join_mask, mask)
    return join_mask

def flatten_groups(groups):
    if len(groups) <= 0:
        return groups
    if type(groups[0][0]) not in (list, tuple):
        return groups
    result = []
    for subgroups in groups:
        subgroups = flatten_groups(subgroups)
        result.extend(subgroups)
    return result

def get_groups(group_descriptions, pdc, mask=None, flat=True):
    """group_descriptions can be any combination of the following strings seperated by commas:
           'treatment', 'replicate', 'well'
       e.g.: 'treatment,well' would group by treatment and well"""
    global custom_get_groups_dict
    if type(group_descriptions) != list:
        group_descriptions = group_descriptions.split(',')
    groups_list = []
    for group_description in group_descriptions:
        if group_description == 'treatment' or group_description == 'tr':
            groups_list.append(get_treatment_groups(pdc, mask, flat))
        elif group_description == 'replicate' or group_description == 'repl':
            groups_list.append(get_replicate_groups(pdc, mask, flat))
        elif group_description == 'well':
            groups_list.append(get_well_groups(pdc, mask, flat))
        #elif group_description == 'treatment+replicate':
        #    groups_list.append(get_treatment_replicate_groups(pdc, mask, flat))
        else:
            if group_description in custom_get_groups_dict_src and \
               group_description not in custom_get_groups_dict:
                update_custom_get_groups_functions()
            if group_description in custom_get_groups_dict and \
               callable(custom_get_groups_dict[group_description]):
                groups_list.append(custom_get_groups_dict[group_description](pdc, mask, flat))
            else:
                raise Exception('Unknown group description: %s' % group_description)
    groups = combine_groups(groups_list, flat)
    return groups

def combine_groups(groups_list, flat=True, default_mask=None, default_name=None):
    if len(groups_list) <= 1:
        return groups_list[0]
    result = []
    if flat:
        for groups in itertools.product(*groups_list):
            names,masks = zip(*groups)
            name = ','.join(names)
            mask = reduce(numpy.logical_and, masks)
            if default_mask != None:
                mask = numpy.logical_and(mask, default_mask)
            result.append([name,mask])
    else:
        first_groups = groups_list[0]
        other_groups = groups_list[1:]
        if len(other_groups) > 1:
            for first_name,first_mask in first_groups:
                if default_mask != None:
                    mask = numpy.logical_and(first_mask, default_mask)
                else:
                    mask = first_mask
                if default_name != None:
                    name = default_name + ',' + first_name
                else:
                    name = first_name
                result.append(combine_groups(other_groups, flat, default_mask=mask, default_name=name))
        else:
            second_groups = other_groups[0]
            for first_name,first_mask in first_groups:
                temp_result = []
                for second_name,second_mask in second_groups:
                    if default_name != None:
                        name = default_name + ',' + first_name + ',' + second_name
                    else:
                        name = first_name + ',' + second_name
                    mask = numpy.logical_and(first_mask, second_mask)
                    if default_mask != None:
                        mask = numpy.logical_and(mask, default_mask)
                    temp_result.append([name, mask])
                result.append(temp_result)
    return result

def get_treatment_groups(pdc, mask=None, flat=None):
    global custom_get_treatment_groups
    if mask == None:
        mask = slice(0, pdc.objFeatures.shape[0])
    if custom_get_treatment_groups == None:
        custom_get_treatment_groups = compile_get_groups_function(custom_get_treatment_groups_src)
    if callable(custom_get_treatment_groups):
        return custom_get_treatment_groups(pdc, mask)
    else:
        masks = []
        names = []
        for tr in pdc.treatments:
            if tr.name not in reject_treatments:
                tr_mask = pdc.objFeatures[:,pdc.objTreatmentFeatureId][mask] == tr.index
                masks.append(tr_mask)
                names.append(tr.name)
        return zip(names, masks)

def get_well_groups(pdc, mask=None, flat=None):
    global custom_get_well_groups
    if mask == None:
        mask = slice(0, pdc.objFeatures.shape[0])
    if custom_get_well_groups == None:
        custom_get_well_groups = compile_get_groups_function(custom_get_well_groups_src)
    if callable(custom_get_well_groups):
        return custom_get_well_groups(pdc, mask)
    else:
        masks = []
        names = []
        for well in pdc.wells:
            well_mask = pdc.objFeatures[:,pdc.objWellFeatureId][mask] == well.index
            masks.append(well_mask)
            names.append(well.name)
        return zip(names, masks)

def get_replicate_groups(pdc, mask=None, flat=None):
    global custom_get_replicate_groups
    if mask == None:
        mask = slice(0, pdc.objFeatures.shape[0])
    if custom_get_replicate_groups == None:
        custom_get_replicate_groups = compile_get_groups_function(custom_get_replicate_groups_src)
    if callable(custom_get_replicate_groups):
        return custom_get_replicate_groups(pdc, mask)
    else:
        masks = []
        names = []
        for repl in pdc.replicates:
            repl_mask = pdc.objFeatures[:,pdc.objReplicateFeatureId][mask] == repl.index
            masks.append(repl_mask)
            names.append('[%d]' % repl.index)
        return zip(names, masks)

#def get_treatment_replicate_groups(pdc, mask=None, flat=None):
    #global custom_get_treatment_groups
    #if mask == None:
        #mask = slice(0, pdc.objFeatures.shape[0])
    #groups = []
    #for tr in pdc.treatments:
        #tr_mask = pdc.objFeatures[:,pdc.objTreatmentFeatureId][mask] == tr.index
        #subnames = []
        #submasks = []
        #for repl in pdc.replicates:
            #if (reject_treatments == None) or (tr.name not in reject_treatments):
                #repl_mask = pdc.objFeatures[:,pdc.objReplicateFeatureId][mask] == repl.index
                #tr_repl_mask = numpy.logical_and(tr_mask, repl_mask)
                #tr_repl_name = '%s,[%d]' % (tr.name, repl.index)
                #subnames.append(tr_repl_name)
                #submasks.append(tr_repl_mask)
        #if len(subnames):
            #subgroups = zip(subnames, submasks)
            #groups.append(subgroups)
    #if flat:
        #flatten_groups(groups)
    #return groups

custom_get_groups_dict['treatment+well'] = functools.partial(get_groups, 'treatment,well')
custom_get_groups_dict['treatment+replicate'] = functools.partial(get_groups, 'treatment,replicate')
custom_get_groups_dict['well+replicate'] = functools.partial(get_groups, 'well,replicate')
custom_get_groups_dict['treatment+well+replicate'] = functools.partial(get_groups, 'treatment,replicate,well')
#custom_get_groups_dict_src['treatment+replicate'] = custom_get_treatment_replicate_groups_src
#custom_get_groups_dict['treatment+replicate'] = get_treatment_replicate_groups
