import sys, os, random

from . core import pipeline
from . core import importer

from . core import parameter_utils as utils




importer = importer.Importer()
pipeline = pipeline.Pipeline( self.importer )

if len( sys.argv ) > 1:
    utils.load_module_configuration( sys.argv[1] )


run_pipeline = True

modules = utils.list_modules()
for module in modules:

    if not utils.all_requirements_met( module ):

        run_pipeline = False


if run_pipeline:

    print 'running pipeline...'
    pipeline.run()

    print 'clustering...'
    pipeline.run_clustering()

