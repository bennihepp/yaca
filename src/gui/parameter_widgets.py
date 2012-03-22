# -*- coding: utf-8 -*-

"""
parameter_widgets.py -- Widgets providing input for parameters of parameter_utils.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ..core.parameter_utils import *

def create_widget(module, param_name, pdc=None):

    param_descr = get_parameter_descr(module, param_name)
    param_type = param_descr[0]
    kwargs = get_parameter_kwargs(module, param_name)

    if param_type == PARAM_ANY:
        widget = ParameterWidgetAny(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_INT:
        widget = ParameterWidgetInt(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_BOOL:
        widget = ParameterWidgetBool(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_FLOAT:
        widget = ParameterWidgetFloat(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_STR:
        widget = ParameterWidgetStr(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_LONGSTR:
        widget = ParameterWidgetLongStr(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_OBJ_FEATURE:
        widget = ParameterWidgetObjFeature(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_IMG_FEATURE:
        widget = ParameterWidgetImgFeature(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_TREATMENT:
        widget = ParameterWidgetTreatment(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_INTS:
        widget = ParameterWidgetInts(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_STRS:
        widget = ParameterWidgetStrs(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_DICT:
        widget = ParameterWidgetBase(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_PATH:
        widget = ParameterWidgetPath(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_INPUT_FILE:
        widget = ParameterWidgetInputFile(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_OUTPUT_FILE:
        widget = ParameterWidgetOutputFile(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_TREATMENTS:
        widget = ParameterWidgetTreatments(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_REPLICATE_TREATMENTS:
        widget = ParameterWidgetReplicateTreatments(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_OBJ_FEATURES:
        widget = ParameterWidgetObjFeatures(module, param_name, pdc, **kwargs)
    elif param_type == PARAM_INPUT_FILES:
        widget = ParameterWidgetInputFiles(module, param_name, pdc, **kwargs)
    else:
        widget = ParameterWidgetBase(module, param_name, pdc, **kwargs)

    return widget



class ParameterWidgetBase(QWidget):

    __pyqtSignals__ = ('parameterChanged',)

    def __init__(self, module, param_name, pdc=None, **kwargs):

        QWidget.__init__(self)

        self.module = module
        self.param_name = param_name
        self.pdc = pdc

        param_descr = get_parameter_descr(module, param_name)
        self.param_default = param_descr[2]
        self.param_min = param_descr[3]
        self.param_max = param_descr[4]
        self.optional = param_descr[5]
        self.hidden = param_descr[6]

        if is_parameter_set(module, param_name):
            self.value_set = True
            self.value = get_parameter_value(module, param_name)
        elif self.param_default != None:
            self.value_set = True
            self.value = self.param_default
        else:
            self.value_set = False

        label = QLabel('%s:' % param_descr[1])

        self.widgets = [(label, 0)]
        self.widget_layout = Qt.Vertical

        if self.hidden:
            self.hide()

    def build_widget(self):
        if self.widget_layout == Qt.Vertical:
            box = QVBoxLayout()
        else:
            box = QHBoxLayout()
        for widget,stretch in self.widgets:
            if issubclass(widget.__class__, QWidget):
                box.addWidget(widget, stretch)
            if issubclass(widget.__class__, QLayout):
                box.addLayout(widget, stretch)
        self.setLayout(box)

    def update_value(self, value):
        set_parameter_value(self.module, self.param_name, value)
        self.emit(SIGNAL('parameterChanged'), self.module, self.param_name)



class ParameterWidgetAny(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.lineedit = QLineEdit()
        if self.param_default != None:
            self.lineedit.setText(self.param_default)
        if self.value_set and self.value is not None:
            self.lineedit.setText(self.value)
        self.connect(self.lineedit, SIGNAL('textChanged(QString)'), self.update_value)

        self.widgets.append((self.lineedit, 1))

        self.build_widget()


class ParameterWidgetInt(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.spinbox = QSpinBox()
        if self.param_min != None:
            self.spinbox.setMinimum(self.param_min)
        else:
            self.spinbox.setMinimum(-10000)
        if self.param_max != None:
            self.spinbox.setMaximum(self.param_max)
        else:
            self.spinbox.setMaximum(10000)

        if self.value_set and self.value is not None:
            self.spinbox.setMinimum(min(self.spinbox.minimum(), self.value))
            self.spinbox.setMaximum(max(self.spinbox.maximum(), self.value))
            self.spinbox.setValue(self.value)

        self.connect(self.spinbox, SIGNAL('valueChanged(int)'), self.update_value)

        self.widgets.append((self.spinbox, 1))
        self.widget_layout = Qt.Horizontal

        self.build_widget()



class ParameterWidgetFloat(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.spinbox = QDoubleSpinBox()

        if self.param_min != None:
            self.spinbox.setMinimum(self.param_min)
        else:
            self.spinbox.setMinimum(-1000.0)
        if self.param_max != None:
            self.spinbox.setMaximum(self.param_max)
        else:
            self.spinbox.setMaximum(1000.0)

        self.spinbox.setSingleStep((self.spinbox.maximum() - self.spinbox.minimum()) / 100.0)

        if self.value_set and self.value is not None:
                self.spinbox.setMinimum(min(self.spinbox.minimum(), self.value))
                self.spinbox.setMaximum(max(self.spinbox.maximum(), self.value))
                self.spinbox.setValue(self.value)

        self.connect(self.spinbox, SIGNAL('valueChanged(double)'), self.update_value)

        self.widgets.append((self.spinbox, 1))
        self.widget_layout = Qt.Horizontal

        self.build_widget()


class ParameterWidgetStr(ParameterWidgetAny):

    def __init__(self, module, param_name, pdc, **kwargs):

        if 'items' in kwargs:
            ParameterWidgetBase.__init__(self, module, param_name, pdc, **kwargs)

            self.comboBox = QComboBox()
            selected_index = -1
            items = list(kwargs['items'])
            items.sort()
            for item in items:
                self.comboBox.addItem(item)

            if self.value_set and self.value is not None:
                for i in xrange(self.comboBox.count()):
                    text = self.comboBox.itemText(i)
                    if text == self.value:
                        selected_index = i
                        break
    
            self.comboBox.setCurrentIndex(selected_index)
    
            self.connect(self.comboBox, SIGNAL('currentIndexChanged(int)'), self.update_value)
    
            self.widgets.append((self.comboBox, 1))
    
            self.build_widget()

        else:
            ParameterWidgetAny.__init__(self, module, param_name, pdc, **kwargs)
            self.comboBox = None

    def update_value(self, value):
        if self.comboBox is None:
            ParameterWidgetBase.update_value(self, value)
        else:
            ParameterWidgetBase.update_value(self, self.comboBox.itemText(value))


class ParameterWidgetLongStr(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.textedit = QPlainTextEdit()
        self.textedit.lineWrapMode = QPlainTextEdit.NoWrap
        self.textedit.wordWrapMode = QPlainTextEdit.NoWrap
        if self.param_default != None:
            self.textedit.setPlainText(self.param_default)
        self.connect(self.textedit, SIGNAL('textChanged()'), self.on_text_changed)

        self.widgets.append((self.textedit, 1))

        self.build_widget()

    def on_text_changed(self):
        text = self.textedit.toPlainText()
        text.replace('\t', '    ')
        self.textedit.setPlainText(text)
        self.update_value(text)

class ParameterWidgetDict(ParameterWidgetAny):
    pass


class ParameterWidgetObjFeature(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.comboBox = QComboBox()
        selected_index = -1
        featureIds = self.pdc.objFeatureIds.keys()
        featureIds.sort()
        for k in featureIds:
            v = self.pdc.objFeatureIds[k]
            self.comboBox.addItem(k, v)

        if self.value_set and self.value is not None:
            for i in xrange(self.comboBox.count()):
                text = self.comboBox.itemText(i)
                if text == self.value:
                    selected_index = i
                    break

        self.comboBox.setCurrentIndex(selected_index)

        self.connect(self.comboBox, SIGNAL('currentIndexChanged(int)'), self.update_value)

        self.widgets.append((self.comboBox, 1))

        self.build_widget()

    def update_value(self, value):
       ParameterWidgetBase.update_value(self, self.comboBox.itemText(value))



class ParameterWidgetImgFeature(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.comboBox = QComboBox()
        selected_index = -1
        featureIds = self.pdc.imgFeatureIds.keys()
        featureIds.sort()
        for k in featureIds:
            v = self.pdc.imgFeatureIds[k]
            self.comboBox.addItem(k, v)

        if self.value_set and self.value is not None:
            for i in xrange(self.comboBox.count()):
                text = self.comboBox.itemText(i)
                if text == self.value:
                    selected_index = i
                    break

        self.comboBox.setCurrentIndex(selected_index)

        self.connect(self.comboBox, SIGNAL('currentIndexChanged(int)'), self.update_value)

        self.widgets.append((self.comboBox, 1))

        self.build_widget()

    def update_value(self, value):
        ParameterWidgetBase.update_value(self, self.comboBox.itemText(value))



class ParameterWidgetTreatment(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.comboBox = QComboBox()
        selected_index = -1
        treatments = self.pdc.treatmentByName.keys()
        treatments.sort()
        for tr_name in treatments:
            j = self.pdc.treatmentByName[tr_name]
            tr = self.pdc.treatments[j]
            self.comboBox.addItem(tr.name)

        if self.value_set and self.value is not None:
            for i in xrange(self.comboBox.count()):
                text = self.comboBox.itemText(i)
                if text == self.value:
                    selected_index = i
                    break

        self.comboBox.setCurrentIndex(selected_index)

        self.connect(self.comboBox, SIGNAL('currentIndexChanged(int)'), self.update_value)

        self.widgets.append((self.comboBox, 1))

        self.build_widget()

    def update_value(self, value):
        ParameterWidgetBase.update_value(self, self.comboBox.itemText(value))



class ParameterWidgetFileDialog(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, caption='Select file'):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.lineedit = QLineEdit()
        self.lineedit.setReadOnly(True)
        self.button = QPushButton('Browse')
        self.filedialog = QFileDialog(self, caption, '')
        self.filedialog.setAcceptMode(QFileDialog.AcceptOpen)
        #self.filedialog.setAcceptMode(QFileDialog.AcceptSave)
        self.filedialog.setFileMode(QFileDialog.ExistingFile)
        #self.filedialog.setFileMode(QFileDialog.Directory)

        if self.value_set and self.value is not None:
            self.lineedit.setText(self.value)
            self.filedialog.selectFile(self.value)

        self.connect(self.filedialog, SIGNAL('fileSelected(QString)'), self.lineedit.setText)
        self.connect(self.button, SIGNAL('clicked()'), self.filedialog.exec_)
        self.connect(self.filedialog, SIGNAL('fileSelected(QString)'), self.update_value)

        self.widgets.append((self.lineedit, 1))
        self.widgets.append((self.button, 0))

        self.build_widget()

    def update_value(self, value):
        value = str(value)
        if value != None:
            ParameterWidgetBase.update_value(self, value)


class ParameterWidgetPath(ParameterWidgetFileDialog):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetFileDialog.__init__(self, module, param_name, pdc, 'Select path')

        self.filedialog.setAcceptMode(QFileDialog.AcceptOpen)
        self.filedialog.setFileMode(QFileDialog.Directory)



class ParameterWidgetInputFile(ParameterWidgetFileDialog):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetFileDialog.__init__(self, module, param_name, pdc, 'Select input file')

        self.filedialog.setAcceptMode(QFileDialog.AcceptOpen)
        self.filedialog.setFileMode(QFileDialog.ExistingFile)



class ParameterWidgetOutputFile(ParameterWidgetFileDialog):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetFileDialog.__init__(self, module, param_name, pdc, 'Select output file')

        self.filedialog.setAcceptMode(QFileDialog.AcceptSave)
        self.filedialog.setFileMode(QFileDialog.AnyFile)



class ParameterWidgetInputFiles(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc=None, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.listwidget = QListWidget()
        self.button = QPushButton('Browse')
        self.filedialog = QFileDialog(self, 'Select input files', '')
        self.filedialog.setAcceptMode(QFileDialog.AcceptOpen)
        self.filedialog.setFileMode(QFileDialog.ExistingFiles)

        if self.value_set and self.value is not None:
            for file in self.value:
                self.listwidget.addItem(file)

        self.connect(self.button, SIGNAL('clicked()'), self.filedialog.exec_)
        self.connect(self.filedialog, SIGNAL('filesSelected(QStringList)'), self.update_value)

        self.widgets.append((self.listwidget, 1))
        self.widgets.append((self.button, 0))

        self.build_widget()

    def update_value(self, value):

        files = []

        self.listwidget.clear()
        for file in value:
            self.listwidget.addItem(file)
            files.append(file)

        if len(files) > 0:
            ParameterWidgetBase.update_value(self, files)



class ParameterWidgetStrs(ParameterWidgetAny):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.MultiSelection)

        items = list(kwargs['items'])
        items.sort()
        for item in items:
            self.listwidget.addItem(item)
            if self.value_set and self.value is not None:
                if item in self.value:
                    self.listwidget.setCurrentRow(self.listwidget.count()-1)

        self.connect(self.listwidget, SIGNAL('itemSelectionChanged()'), self.on_selection_changed)

        self.widgets.append((self.listwidget, 1))

        self.build_widget()

    def on_selection_changed(self):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for item in self.listwidget.selectedItems():
            value.append(str(item.text()))

        ParameterWidgetBase.update_value(self, value)


class ParameterWidgetTreatments(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.treatment_names = [tr.name for tr in pdc.treatments]
        self.treatment_names.sort()
        for tr_name in self.treatment_names:
            self.listwidget.addItem(tr_name)
            if self.value_set and self.value is not None:
                if tr_name in self.value:
                    self.listwidget.setCurrentRow(self.listwidget.count()-1)

        self.connect(self.listwidget, SIGNAL('itemSelectionChanged()'), self.on_selection_changed)

        self.widgets.append((self.listwidget, 1))

        self.build_widget()

    def on_selection_changed(self):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for item in self.listwidget.selectedItems():
            value.append(str(item.text()))

        ParameterWidgetBase.update_value(self, value)

class ParameterWidgetReplicateTreatments(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.entry_names = []
        self.replicateTreatmentMap = {}
        for tr in pdc.treatments:
            for repl in pdc.replicates:
                name = '%s,[%d]' % (tr.name, repl.index)
                self.entry_names.append(name)
                self.replicateTreatmentMap[name] = [tr.name, repl.index]

        #print 'value:', self.value
        self.entry_names.sort()
        for name in self.entry_names:
            self.listwidget.addItem(name)
            if self.value_set and self.value is not None:
                #print 'name:', name
                #print 'v:', self.replicateTreatmentMap[name]
                if self.replicateTreatmentMap[name] in self.value:
                    self.listwidget.setCurrentRow(self.listwidget.count()-1)

        self.connect(self.listwidget, SIGNAL('itemSelectionChanged()'), self.on_selection_changed)

        self.widgets.append((self.listwidget, 1))

        self.build_widget()

    def on_selection_changed(self):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for item in self.listwidget.selectedItems():
            value.append(self.replicateTreatmentMap[str(item.text())])
        print 'value:', value

        ParameterWidgetBase.update_value(self, value)


class ParameterWidgetObjFeatures(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.MultiSelection)

        self.featureIds = list(self.pdc.objFeatureIds.keys())
        self.featureIds.sort()
        for i in xrange(len(self.featureIds)):
            k = self.featureIds[i]
            self.listwidget.addItem(k)
            if self.value_set and self.value is not None:
                if k in self.value:
                    self.listwidget.setCurrentRow(self.listwidget.count()-1)

        self.connect(self.listwidget, SIGNAL('itemSelectionChanged()'), self.on_selection_changed)

        self.widgets.append((self.listwidget, 1))

        self.build_widget()

    def on_selection_changed(self):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for item in self.listwidget.selectedItems():
            value.append(str(item.text()))

        ParameterWidgetBase.update_value(self, value)


class ParameterWidgetObjFeatures2(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):

        ParameterWidgetBase.__init__(self, module, param_name, pdc)

        scrollarea = QScrollArea()
        self.buttongroup = QButtonGroup()
        self.buttongroup.setExclusive(False)
        self.buttons = []
        vbox = QVBoxLayout()

        self.featureIds = list(self.pdc.objFeatureIds.keys())
        self.featureIds.sort()
        for i in xrange(len(self.featureIds)):
            k = self.featureIds[i]
            self.checkBox = QCheckBox(k)
            self.checkBox.setChecked(False)
            if self.value_set and self.value is not None:
                if k in self.value:
                    self.checkBox.setChecked(True)
            vbox.addWidget(self.checkBox)
            self.buttongroup.addButton(self.checkBox, i)
            self.buttons.append(self.checkBox)

        w = QWidget()
        w.setLayout(vbox)
        scrollarea.setWidget(w)

        self.connect(self.buttongroup, SIGNAL('buttonClicked(int)'), self.update_value)

        self.widgets.append((scrollarea, 1))

        self.build_widget()

    def update_value(self, value):
        value = []
        for i in xrange(len(self.buttons)):
            if self.buttons[i].isChecked():
                value.append(self.featureIds[i])

        ParameterWidgetBase.update_value(self, value)

class ParameterWidgetBool(ParameterWidgetBase):

    def __init__(self, module, param_name, pdc, **kwargs):
        ParameterWidgetBase.__init__(self, module, param_name, pdc)
        self.checkBox = QCheckBox()
        vbox = QVBoxLayout()
        vbox.addWidget(self.checkBox)

        if self.value_set and self.value is not None:
            if type(self.value) == str:
                self.checkBox.setChecked(self.value.lower() == 'true')
            else:
                self.checkBox.setChecked(self.value)

        self.connect(self.checkBox, SIGNAL('stateChanged(int)'), self.update_value)

        self.widgets.append((self.checkBox,1))
        self.widget_layout = Qt.Horizontal

        self.build_widget()

    def update_value(self, value):
        value = bool(value)
        ParameterWidgetBase.update_value(self, value)
