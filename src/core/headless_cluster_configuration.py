from ..core import parameter_utils as utils



class HeadlessClusterConfiguration:

    __OBJECT_NAME = 'ClusterConfiguration'

    def __init__(self):

        self.clusterConfiguration = []

        utils.register_object( self.__OBJECT_NAME )
        utils.register_attribute( self.__OBJECT_NAME, 'clusterConfiguration', self.getClusterConfiguration, self.setClusterConfiguration )

    def getClusterConfiguration(self):
        return self.clusterConfiguration
    def setClusterConfiguration(self, configuration):
        self.clusterConfiguration = configuration
