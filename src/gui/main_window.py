import sys, os, random
import numpy
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from results_window import ResultsWindow
from channel_description_widgets import ChannelDescriptionTab
from cluster_configuration_widgets import ClusterConfigurationTab
from gallery_window import GalleryWindow

from gui_utils import ImagePixmapFactory, ImageFeatureTextFactory

import parameter_widgets

from ..core import pipeline
from ..core import importer

from ..core import parameter_utils as utils



class ActionButton(QPushButton):

    __pyqtSignals__ = ('action',)

    def __init__(self, descr, module, action_name, parent=None):

        QPushButton.__init__( self, descr, parent )

        self.module = module
        self.action_name = action_name
        self.connect( self, SIGNAL('clicked()'), self.on_clicked )

    def on_clicked(self):
        self.emit( SIGNAL('action'), self.module, self.action_name )


class MainWindow(QMainWindow):

    def __init__(self, parent=None):

        self.importer = importer.Importer()

        QMainWindow.__init__(self, parent)
        self.setWindowTitle('APC Main')

        self.project_unsaved = False

        self.results_window = None

        self.build_menu()
        self.build_main_frame()
        self.build_status_bar()

        self.image_viewer = None
        self.results_window = None

        self.channelDescriptionTab = None
        self.clusterConfigurationTab = None

        """self.channelDescription = {}
        self.channelDescription['R'] = 'Nucleus staining (A568)'
        self.channelDescription['G'] = 'Protein staining (A488)'
        self.channelDescription['B'] = 'Cell staining (DAPI)'
        self.channelDescription['O1'] = 'Cell segmentation'
        self.channelDescription['O2'] = 'Nucleus segmentation'"""


    def closeEvent(self, event):
        if self.project_saved():
            if self.results_window:
                self.results_window.close()
            if self.image_viewer:
                self.image_viewer.close()
            event.accept()
        else:
            event.ignore()

    def project_saved(self):

        if self.project_unsaved:

            msgBox = QMessageBox(
                QMessageBox.Question,
                'Unsaved project',
                'The current project has not been saved!',
                QMessageBox.Cancel | QMessageBox.Save | QMessageBox.Discard,
                self
            )
            result = msgBox.exec_()

            if result == QMessageBox.Save:
                self.on_save_project()
            elif result == QMessageBox.Cancel:
                return False
            elif result == QMessageBox.Discard:
                self.project_unsaved = False

        return True


    def on_new_project(self):
        if self.project_saved():
            utils.reset_module_configuration()
            self.reset_module_tab_widget( self.tab_widget )
            self.update_module_tab_widget( self.tab_widget )
            self.statusBar().showMessage( 'New project' )

    def load_project_file(self, path):
        self.statusBar().showMessage( 'Loading project file...' )
        utils.load_module_configuration( path )
        self.reset_module_tab_widget( self.tab_widget )
        self.update_module_tab_widget( self.tab_widget )
        self.statusBar().showMessage( 'Project file loaded' )

    def save_project_file(self, path):
        self.statusBar().showMessage( 'Saving project file...' )
        utils.save_module_configuration( path )
        self.statusBar().showMessage( 'Project file saved' )

    def on_open_project(self):
        if self.project_saved():
            file_choices = "APC Project file (*.apc *.yaml);;All files (*)"
    
            path = unicode(QFileDialog.getOpenFileName(self, 
                            'Open file', '', 
                            file_choices))
            if path:
                self.on_new_project()
                self.load_project_file( path )
                self.statusBar().showMessage( 'Opened %s' % path )
                return True

            return False

    def on_save_project(self):
        file_choices = "APC Project file (*.apc *.yaml);;All files (*)"

        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.save_project_file( path )
            self.project_unsaved = False
            self.statusBar().showMessage( 'Saved to %s' % path )
            return True

        return False

    def on_close(self):
        if self.project_saved():
            self.close()

    def on_about(self):
        msg = """ APC GUI """
        QMessageBox.about(self, "About", msg.strip())


    def on_update_progress(self, progress):
        self.progress_bar.setValue( progress )

    def on_start_cancel(self):

        run_pipeline = True

        modules = utils.list_modules()
        for module in modules:

            if not utils.all_parameters_set( module ):

                QMessageBox(
                    QMessageBox.Warning,
                    'Not all required parameters for module %s have been set' % module,
                    'Unable to start pipeline',
                    QMessageBox.Ok,
                    self
                ).exec_()
                run_pipeline = False
                break

            elif not utils.all_requirements_met( module ):

                QMessageBox(
                    QMessageBox.Warning,
                    'Not all requirements for module %s have been fulfilled' % module,
                    'Unable to start pipeline',
                    QMessageBox.Ok,
                    self
                ).exec_()
                run_pipeline = False
                break

        if run_pipeline:

            self.start_cancel_button.setText( 'Cancel' )

            self.progress_bar.setRange( 0, 100 )
            self.progress_bar.setFormat( 'processing input data - %p%' )
    
            try:

                adc = self.importer.get_adc()
                clusterConfiguration = self.clusterConfigurationTab.clusterConfiguration

                self.pl = pipeline.Pipeline( adc, clusterConfiguration )
                self.pl.connect( self.pl, SIGNAL('updateProgress'), self.on_update_progress )
                self.pl.connect( self.pl, SIGNAL('finished()'), self.on_pipeline_finished )
                self.pl.start()
                #pl.run( self.on_update_progress )

            except:
    
                self.progress_bar.setFormat( 'Idling...' )
    
                self.statusBar().showMessage( 'Unable to start pipeline thread!' )
                raise


    def on_pipeline_finished(self):

        if self.pl.result:
            self.statusBar().showMessage( 'Showing results window' )
            print 'creating results window...'
            channelMapping = self.channelDescriptionTab.channelMapping
            channelDescription = self.channelDescriptionTab.channelDescription
            self.results_window = ResultsWindow( self.pl, channelMapping, channelDescription )
            self.results_window.show()
    
            self.progress_bar.setFormat( 'Idling...' )
    
            self.start_cancel_button.setText( 'Start' )

        else:
            self.statusBar().showMessage( 'Error while running pipeline' )
    
            self.progress_bar.setFormat( 'Idling...' )
    
            self.start_cancel_button.setText( 'Start' )

        del self.pl


    def on_parameter_changed(self, module, param_name):

        self.project_unsaved = True

        self.update_module_tab_widget( self.tab_widget )


    def on_module_action(self, module, action_name):

        self.project_unsaved = True

        try:

            result = utils.trigger_action( module, action_name )
            self.statusBar().showMessage( result )

            self.update_module_tab_widget( self.tab_widget )

        except Exception,e:
            QMessageBox(
                QMessageBox.Warning,
                '%s/%s' % (module,action_name),
                str( e ),
                QMessageBox.Ok,
                self
            ).exec_()
            raise


    def build_module_tab_widget(self):

        tab_widget = QTabWidget()

        self.modules_used = []
        self.update_module_tab_widget( tab_widget )

        return tab_widget

    def reset_module_tab_widget(self, tab_widget):

        while tab_widget.count() > 0:
            widget = tab_widget.widget( 0 )
            tab_widget.removeTab( 0 )
            del widget

        self.modules_used = []

    def update_module_tab_widget(self, tab_widget):

        all_requirements_and_parameters_met = True

        modules = utils.list_modules()
        for module in modules:

            if not utils.all_parameters_set( module ):
                #print 'Not all required parameters for module %s have been set' % module
                all_requirements_and_parameters_met = False
                break

            elif not utils.all_requirements_met( module ):
                #print 'Not all requirements for module %s have been fulfilled' % module
                all_requirements_and_parameters_met = False
                break

        self.start_cancel_button.setEnabled( all_requirements_and_parameters_met )
        self.view_images_button.setEnabled( all_requirements_and_parameters_met )

        modules = utils.list_modules()
        for module in modules:

            if module not in self.modules_used and utils.all_requirements_met( module ):

                self.modules_used.append( module )

                params = utils.list_parameters( module )
                actions = utils.list_actions( module )
                if len( params ) > 0 or len( actions ) > 0:
                    layout = QVBoxLayout()

                    for param in params:

                        param_widget = parameter_widgets.create_widget( module, param, self.importer.get_adc() )
                        self.connect( param_widget, SIGNAL('parameterChanged'), self.on_parameter_changed )
            
                        layout.addWidget( param_widget )

                    for action in actions:

                        descr = utils.get_action_descr( module, action )
                        btn = ActionButton( descr, module, action )
                        self.connect( btn, SIGNAL('action'), self.on_module_action)

                        layout.addWidget( btn )
                
                    widget = QWidget()
                    widget.setLayout( layout )
    
                    scrollarea = QScrollArea()
                    scrollarea.setWidget( widget )
    
                    tab_widget.addTab( scrollarea, utils.get_module_descr( module ) )

        if self.importer.get_adc() != None:

            if self.channelDescriptionTab == None:
                self.channelDescriptionTab = ChannelDescriptionTab(
                    self.importer.get_adc()
                )
                tab_widget.addTab( self.channelDescriptionTab, 'Channels' )

            if self.clusterConfigurationTab == None:
                self.clusterConfigurationTab = ClusterConfigurationTab(
                    self.importer.get_adc()
                )
                tab_widget.addTab( self.clusterConfigurationTab, 'Clustering configuration' )


    def on_view_images(self):
        channelMapping = self.channelDescriptionTab.channelMapping
        channelDescription = self.channelDescriptionTab.channelDescription
        adc = self.importer.get_adc()
        self.image_viewer = GalleryWindow( adc.imgFeatureIds, channelMapping, channelDescription, True )
        selectionIds = numpy.arange( len( adc.images ) )
        pixmapFactory = ImagePixmapFactory( adc, channelMapping )
        featureFactory = ImageFeatureTextFactory( adc )
        self.image_viewer.on_selection_changed( -1, selectionIds, pixmapFactory, featureFactory )
        self.image_viewer.show()

    def build_main_frame(self):

        self.main_frame = QWidget()

        self.start_cancel_button = QPushButton('Start')
        self.start_cancel_button.setEnabled( False )
        self.connect( self.start_cancel_button, SIGNAL('clicked()'), self.on_start_cancel )

        self.view_images_button = QPushButton('View images')
        self.view_images_button.setEnabled( False )
        self.connect( self.view_images_button, SIGNAL('clicked()'), self.on_view_images )

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat( 'Idling...' )
        self.progress_bar.setValue( self.progress_bar.minimum() )

        self.tab_widget = self.build_module_tab_widget()


        #
        # Layout with box sizers
        #

        hbox1 = QHBoxLayout()
        hbox1.addWidget( self.start_cancel_button )
        hbox1.addWidget( self.progress_bar, 1 )
        hbox1.addWidget( self.view_images_button )

        vbox = QVBoxLayout()
        vbox.addLayout( hbox1 )
        vbox.addWidget( self.tab_widget, 1 )

        self.main_frame.setLayout( vbox )
        self.setCentralWidget( self.main_frame )
    
    def build_status_bar(self):
        self.status_text = QLabel( 'Welcome to APC' )
        self.statusBar().addWidget(self.status_text, 1)
        
    def build_menu(self):        
        self.project_menu = self.menuBar().addMenu("&Project")
        
        new_project_action = self.make_action("&New project",
            shortcut="Ctrl+N", slot=self.on_new_project, 
            tip="Create a new project")
        open_project_action = self.make_action("&Open project",
            shortcut="Ctrl+O", slot=self.on_open_project, 
            tip="Open a project file")
        save_project_action = self.make_action("&Save project",
            shortcut="Ctrl+S", slot=self.on_save_project, 
            tip="Save the current project to a file")
        quit_action = self.make_action("&Quit", slot=self.on_close, 
            shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions( self.project_menu, 
            ( new_project_action, open_project_action, save_project_action, None, quit_action )
        )
        
        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.make_action("&About", 
            shortcut='F1', slot=self.on_about, 
            tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def make_action(  self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action