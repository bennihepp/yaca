# -*- coding: utf-8 -*-

"""
parameter_utils.py -- Automatic parameter importing from configuration files.

This module allows other modules to register themselves, required parameters,
provided attributes and possible actions.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import os
import yaml

PARAM_ANY = 'any'
PARAM_INT = 'int'
PARAM_BOOL = 'bool'
PARAM_FLOAT = 'float'
PARAM_STR = 'str'
PARAM_LONGSTR = 'longstr'
PARAM_OBJ_FEATURE = 'objFeature'
PARAM_IMG_FEATURE = 'imgFeature'
PARAM_TREATMENT = 'treatment'
PARAM_PATH = 'path'
PARAM_INPUT_FILE = 'input_file'
PARAM_OUTPUT_FILE = 'output_file'
PARAM_INTS = 'ints'
PARAM_STRS = 'strs'
PARAM_DICT = 'dict'
PARAM_TREATMENTS = 'treatments'
PARAM_REPLICATE_TREATMENTS = 'replicate_treatments'
PARAM_OBJ_FEATURES = 'objFeatures'
PARAM_INPUT_FILES = 'input_files'

PARAM_TYPES = [
    PARAM_ANY,
    PARAM_INT,
    PARAM_BOOL,
    PARAM_FLOAT,
    PARAM_STR,
    PARAM_LONGSTR,
    PARAM_OBJ_FEATURE,
    PARAM_IMG_FEATURE,
    PARAM_TREATMENT,
    PARAM_PATH,
    PARAM_INPUT_FILE,
    PARAM_OUTPUT_FILE,
    PARAM_INTS,
    PARAM_STRS,
    PARAM_DICT,
    PARAM_TREATMENTS,
    PARAM_REPLICATE_TREATMENTS,
    PARAM_OBJ_FEATURES,
    PARAM_INPUT_FILES
]


REQUIREMENT_PARAM = 0
REQUIREMENT_STATE = 1


DEFAULT_STATE = 'default'


__OBJECTS = {}
__OBJECTS_TO_NAMES = {}
__OBJECT_ATTRIBUTE_CALLBACKS = {}

__MODULES = []
__MODULE_DESCR = {}
__MODULE_CONTEXTS = {}
__MODULE_PARAMETERS = {}
__MODULE_PARAMETER_DESCR = {}
__MODULE_PARAMETER_KWARGS = {}
__MODULE_PARAMETER_HOOKS = {}
__MODULE_REQUIREMENTS = {}
__MODULE_STATES = {}
__MODULE_STATE_CALLBACKS = {}
__MODULE_ACTIONS = {}
__MODULE_VALID = {}

__PRE_LOADED_MODULES = {}
__PRE_LOADED_OBJECTS = {}


class ParameterException(object):
    pass


def register_object(name, obj):
    __OBJECTS[name] = obj
    __OBJECTS_TO_NAMES[obj] = name
    __OBJECT_ATTRIBUTE_CALLBACKS[obj] = {}

def retrieve_object(name):
    if not name in __OBJECTS:
        raise Exception('Object has to be registered before being retrieved: object %s' % name)
    return __OBJECTS[name]

def register_module(module, descr, param_context, state=DEFAULT_STATE):
    __MODULES.append(module)
    __MODULE_DESCR[module] = descr
    __MODULE_CONTEXTS[module] = param_context
    __MODULE_PARAMETERS[module] = []
    __MODULE_PARAMETER_DESCR[module] = {}
    __MODULE_PARAMETER_KWARGS[module] = {}
    __MODULE_PARAMETER_HOOKS[module] = {}
    __MODULE_STATES[module] = 'default'
    __MODULE_STATE_CALLBACKS[module] = None
    __MODULE_ACTIONS[module] = {}
    __MODULE_VALID[module] = False

def validate_module(module, valid=True):
    if not module in __MODULES:
        raise Exception('Module has to be registered before being validated: module %s' % module)

    __MODULE_VALID[module] = valid

def invalidate_module(module, invalid=True):
    if not module in __MODULES:
        raise Exception('Module has to be registered before being invalidated: module %s' % module)

    __MODULE_VALID[module] = not invalid

def register_attribute(obj, attr_name, attr_get, attr_set, attr_reset = None):
    if not obj in __OBJECTS_TO_NAMES:
        raise Exception('Object has to be registered before adding attributes: object %s' % obj)

    name = __OBJECTS_TO_NAMES[obj]
    __OBJECT_ATTRIBUTE_CALLBACKS[obj][attr_name] = (attr_get, attr_set, attr_reset)

    if name in __PRE_LOADED_OBJECTS:
        if attr_name in __PRE_LOADED_OBJECTS[name]:
            attr = __PRE_LOADED_OBJECTS[name][attr_name]
            attr_set(attr)
            del __PRE_LOADED_OBJECTS[name][attr_name]
            if len(__PRE_LOADED_OBJECTS[name]) <= 0:
                del __PRE_LOADED_OBJECTS[name]


def register_parameter(module, param_name, param_type, param_descr, param_default=None, param_min=None, param_max=None, optional=False, hidden=False, **kwargs):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)

    if not param_type in PARAM_TYPES:
        raise Exception('Unknown parameter type: %s' % param_type)

    #if 'optional' in kwargs:
        #optional = kwargs['optional']
    #if 'param_default' in kwargs:
        #param_default = kwargs['param_default']
    #if 'param_min' in kwargs:
        #param_min = kwargs['param_min']
    #if 'param_max' in kwargs:
        #param_max = kwargs['param_max']

    hook = kwargs.pop('hook', None)

    new_parameter = param_name not in __MODULE_PARAMETERS[module]

    if new_parameter:
        __MODULE_PARAMETERS[module].append(param_name)
    __MODULE_PARAMETER_DESCR[module][param_name] = (param_type, param_descr, param_default, param_min, param_max, optional, hidden)
    set_parameter_hook(module, param_name, hook)
    set_parameter_kwargs(module, param_name, **kwargs)

    if param_default is not None \
       and (new_parameter or get_parameter_value(module, param_name) is None):
            set_parameter_value(module, param_name, param_default)

    if module in __PRE_LOADED_MODULES:
        if param_name in __PRE_LOADED_MODULES[module]['parameters']:
            value = __PRE_LOADED_MODULES[module]['parameters'][param_name]
            set_parameter_value(module, param_name, value)
            del __PRE_LOADED_MODULES[module][param_name]
            if len(__PRE_LOADED_MODULES[module]) <= 0:
                del __PRE_LOADED_MODULES[module]

def set_parameter_kwargs(module, param_name, **kwargs):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    hook = __MODULE_PARAMETER_HOOKS[module][param_name]
    if hook is not None and callable(hook):
        value = get_parameter_value(module, param_name)
        value = hook(module, param_name, value, None)

    __MODULE_PARAMETER_KWARGS[module][param_name] = kwargs

def set_parameter_hook(module, param_name, hook):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))
    
    __MODULE_PARAMETER_HOOKS[module][param_name] = hook

def set_parameter_optional(module, param_name, optional=True):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    (param_type, param_descr, param_default, param_min, param_max, old_optional, hidden) = \
        __MODULE_PARAMETER_DESCR[module][param_name]
    __MODULE_PARAMETER_DESCR[module][param_name] = (param_type, param_descr, param_default, param_min, param_max, optional, hidden)

def set_parameter_visible(module, param_name, visible=True):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    (param_type, param_descr, param_default, param_min, param_max, optional, hidden) = __MODULE_PARAMETER_DESCR[module][param_name]
    hidden = not visible
    __MODULE_PARAMETER_DESCR[module][param_name] = (param_type, param_descr, param_default, param_min, param_max, optional, hidden)

def set_parameter_hidden(module, param_name, hidden=True):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    (param_type, param_descr, param_default, param_min, param_max, optional, old_hidden) = __MODULE_PARAMETER_DESCR[module][param_name]
    __MODULE_PARAMETER_DESCR[module][param_name] = (param_type, param_descr, param_default, param_min, param_max, optional, hidden)


def register_action(module, action_name, action_descr, action_callback):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding actions: module %s' % module)

    __MODULE_ACTIONS[module][action_name] = (action_descr, action_callback)


def set_module_state_callback(module, state_callback):
    if not module in __MODULES:
        raise Exception('Module has to be registered before setting a state callback: module %s' % module)

    __MODULE_STATE_CALLBACKS[module] = state_callback

    if module in __PRE_LOADED_MODULES:
        if 'state' in __PRE_LOADED_MODULES[module]:
            state = __PRE_LOADED_MODULES[module]['state']
            state_callback(state)
            del __PRE_LOADED_MODULES[module]['state']
            if len(__PRE_LOADED_MODULES[module]) <= 0:
                del __PRE_LOADED_MODULES[module]


def reset_module_configuration():

    for module in __MODULES:

        for param_name in __MODULE_PARAMETERS[module]:

            param_default = __MODULE_PARAMETER_DESCR[module][param_name][2]

            set_parameter_value(module, param_name, param_default)

        update_state(module, DEFAULT_STATE)

        cb = __MODULE_STATE_CALLBACKS[module]
        if cb != None:
            cb(DEFAULT_STATE)

    for obj in __OBJECTS.itervalues():

        for attr_name in __OBJECT_ATTRIBUTE_CALLBACKS[obj]:

            (attr_get, attr_set, attr_reset) = __OBJECT_ATTRIBUTE_CALLBACKS[obj][attr_name]

            if attr_reset != None:
                attr_reset()
            else:
                attr_set(None)


def save_module_configuration(filename):

    module_container = {}

    for module in __MODULES:

        parameters = {}
        for param_name in __MODULE_PARAMETERS[module]:

            try:

                context = __MODULE_CONTEXTS[module]
                value = context[param_name]

                parameters[param_name] = value
 
            except:
                pass


        state = __MODULE_STATES[module]

        module_container[module] = {
            'state' : state,
            'parameters' : parameters,
        }

    obj_container = {}

    for name, obj in __OBJECTS.iteritems():

        attributes = {}
        for attr_name in __OBJECT_ATTRIBUTE_CALLBACKS[obj]:

            (attr_get, attr_set, attr_reset) = __OBJECT_ATTRIBUTE_CALLBACKS[obj][attr_name]

            value = attr_get()
            if value != None:
                attributes[attr_name] = value

        obj_container[name] = attributes

    yaml_container = {}
    yaml_container['modules'] = module_container
    yaml_container['objects'] = obj_container

    file = None
    try:
        file = open(filename, 'w')

        yaml.dump(yaml_container, file)

    finally:
        if file:
            file.close()


def load_module_configuration(filename , pathname=None):
    file = None
    try:
        if pathname != None:
            filename = os.path.join(pathname, filename)
        file = open(filename, 'r')
        pathname = os.path.join(os.getcwd(), os.path.dirname(filename))

        yaml_container = yaml.load(file)

    finally:
        if file:
            file.close()

    #if (not 'modules' in yaml_container) or (not 'objects' in yaml_container):
    #    raise Exception('Invalid YACA configuration file')


    if 'modules' in yaml_container:
        module_container = yaml_container['modules']
        for module in module_container:
    
            if module in __MODULES:
    
                if 'parameters' in module_container[module]:
                    parameters = module_container[module]['parameters']
                else:
                    parameters = {}
                for param_name in parameters:
                    value = parameters[param_name]
                    set_parameter_value(module, param_name, value, filename)

                if 'state' in module_container[module]:
                    state = module_container[module]['state']
                    update_state(module, state)
                    cb = __MODULE_STATE_CALLBACKS[module]
                    if cb != None:
                        cb(state)

            else:
    
                __PRE_LOADED_MODULES[module] = module_container[module]

    if 'objects' in yaml_container:
        obj_container = yaml_container['objects']
        for name in obj_container:
    
            if name in __OBJECTS:

                try:
                    attributes = obj_container[name]
                except:
                    raise Exception('Invalid YACA configuration file')

                obj = __OBJECTS[name]        
                for attr_name in attributes:

                    value = attributes[attr_name]

                    (attr_get, attr_set, attr_reset) = __OBJECT_ATTRIBUTE_CALLBACKS[obj][attr_name]

                    attr_set(value)

            else:

                __PRE_LOADED_OBJECTS[name] = obj_container[name]

    if 'config_files' in yaml_container:
        config_files_container = yaml_container['config_files']
        for config_file in config_files_container:
            load_module_configuration(config_file, pathname)


def update_state(module, state):
    if not module in __MODULES:
        raise Exception('Module has to be registered before updating states: module %s' % module)

    __MODULE_STATES[module] = state


def get_state(module):
    if not module in __MODULES:
        raise Exception('Module has to be registered before reading states: module %s' % module)

    return __MODULE_STATES[module]


def list_modules():
    return list(__MODULES)


def get_module_descr(module):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)

    return __MODULE_DESCR[module]


def list_parameters(module):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)

    return list(__MODULE_PARAMETERS[module])


def get_parameter_descr(module, param_name):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    return __MODULE_PARAMETER_DESCR[module][param_name]


def get_parameter_kwargs(module, param_name):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    return __MODULE_PARAMETER_KWARGS[module][param_name]


def get_parameter_value(module, param_name):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    context = __MODULE_CONTEXTS[module]
    try:
        value = context[param_name]
    except:
        return None

    return value


def list_actions(module):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)

    return list(__MODULE_ACTIONS[module].keys())


def get_action_descr(module, action_name):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not action_name in __MODULE_ACTIONS[module]:
        raise Exception('Action has not been registered yet: module/action %s/%s' % (module,action_name))

    return __MODULE_ACTIONS[module][action_name][0]


def trigger_action(module, action_name, *args, **kwargs):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not action_name in __MODULE_ACTIONS[module]:
        raise Exception('Action has not been registered yet: module/action %s/%s' % (module,action_name))

    callback = __MODULE_ACTIONS[module][action_name][1]
    return callback(*args, **kwargs)


def add_required_parameter(module, required_module, required_param_name=None):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding requirements: module %s' % module)

    if not module in __MODULE_REQUIREMENTS:
        __MODULE_REQUIREMENTS[module] = []
    __MODULE_REQUIREMENTS[module].append((REQUIREMENT_PARAM, required_module, required_param_name))


def add_required_state(module, required_module, required_state):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding requirements: module %s' % module)

    if not module in __MODULE_REQUIREMENTS:
        __MODULE_REQUIREMENTS[module] = []
    __MODULE_REQUIREMENTS[module].append((REQUIREMENT_STATE, required_module, required_state))


def all_requirements_met(module):
    if not module in __MODULES:
        raise Exception('Module has to be registered before adding parameters: module %s' % module)

    all_met = True

    if module in __MODULE_REQUIREMENTS:

        for req_type,req_module,req_name in __MODULE_REQUIREMENTS[module]:

            if req_type == REQUIREMENT_STATE:
                state = __MODULE_STATES[req_module]
                if req_name != state:
                    all_met = False
                    break

            if req_type == REQUIREMENT_PARAM:
                if req_name == None:
                    if not all_parameters_set(req_module):
                        all_met = False
                        break
                else:
                    if not is_parameter_set(req_module, req_name):
                        all_met = False
                        break

    return all_met


def is_module_valid(module):
    if not module in __MODULES:
        raise Exception('Module has to be registered before checking valid state: module %s' % module)

    return __MODULE_VALID[module]

def is_module_invalid(module):
    if not module in __MODULES:
        raise Exception('Module has to be registered before checking invalid state: module %s' % module)

    return not __MODULE_VALID[module]

def is_parameter_set(module, param_name):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    context = __MODULE_CONTEXTS[module]
    try:
        context[param_name]
    except:
        return False

    return True


def all_parameters_set(module):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)

    all_set = True

    for param_name in __MODULE_PARAMETERS[module]:
        param_type, param_descr, param_default, param_min, param_max, optional, hidden = \
                  __MODULE_PARAMETER_DESCR[module][param_name]
        if not optional:
            context = __MODULE_CONTEXTS[module]
            try:
                context[param_name]
            except:
                all_set = False
                break

    return all_set


def set_parameter_value(module, param_name, value, yaml_filename=None):
    if not module in __MODULES:
        raise Exception('Module has not been registered yet: module %s' % module)
    if not param_name in __MODULE_PARAMETERS[module]:
        raise Exception('Parameter has not been registered yet: module/parameter %s/%s' % (module, param_name))

    param_type = __MODULE_PARAMETER_DESCR[module][param_name][0]

    if value != None:

        try:
            if param_type == PARAM_INT:
                value = int(value)
            elif param_type == PARAM_BOOL:
                value = bool(value)
            elif param_type == PARAM_FLOAT:
                value = float(value)
            elif param_type == PARAM_STR:
                value = str(value)
            elif param_type == PARAM_LONGSTR:
                value = str(value)
            elif param_type == PARAM_OBJ_FEATURE:
                value = str(value)
            elif param_type == PARAM_IMG_FEATURE:
                value = str(value)
            elif param_type == PARAM_TREATMENT:
                value = str(value)
            elif param_type == PARAM_PATH:
                value = str(value)
            elif param_type == PARAM_INPUT_FILE:
                value = str(value)
            elif param_type == PARAM_INTS:
                value = list(value)
                for i in xrange(len(value)):
                    value[i] = int(value[i])
            elif param_type == PARAM_STRS:
                value = list(value)
                for i in xrange(len(value)):
                    value[i] = str(value[i])
            elif param_type == PARAM_DICT:
                value = dict(value)
                for k,v in value.iteritems():
                    value[k] = str(value[v])
            elif param_type == PARAM_TREATMENTS:
                value = list(value)
                for i in xrange(len(value)):
                    value[i] = str(value[i])
            elif param_type == PARAM_REPLICATE_TREATMENTS:
                value = list(value)
                #for i in xrange(len(value)):
                #    value[i] = str(value[i])
            elif param_type == PARAM_OBJ_FEATURES:
                value = list(value)
                for i in xrange(len(value)):
                    value[i] = str(value[i])
            elif param_type == PARAM_INPUT_FILES:
                value = list(value)
                for i in xrange(len(value)):
                    value[i] = str(value[i])
            elif param_type == PARAM_ANY:
                pass
        except:
            raise Exception('Parameter value is of wrong type: module/param %s/%s' % (module, param_name))

        param_min = __MODULE_PARAMETER_DESCR[module][param_name][3]
        param_max = __MODULE_PARAMETER_DESCR[module][param_name][4]
        if param_min != None:
            if value < param_min:
                raise Exception('Parameter value is too small: module/param %s/%s' % (module, param_name))
        if param_max != None:
            if value > param_max:
                raise Exception('Parameter value is too big: module/param %s/%s' % (module, param_name))

        hook = __MODULE_PARAMETER_HOOKS[module][param_name]
        if hook is not None and callable(hook):
            value = hook(module, param_name, value, yaml_filename)

        context = __MODULE_CONTEXTS[module]
        context[param_name] = value
