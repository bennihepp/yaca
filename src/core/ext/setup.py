from distutils.core import setup, Extension
from Cython.Distutils import build_ext

#cluster_ext_module = Extension('ccluster',
#                   sources=['ccluster_ext.c'])

profile_ext_module = Extension('ccluster_profiles',
                   sources=['ccluster_profiles.pyx'])

#profile_mp_ext_module = Extension('ccluster_profiles_mp',
#                   sources=['ccluster_profiles_mp.pyx'])

profile_mp_worker_ext_module = Extension('ccluster_profiles_mp_worker',
                   sources=['ccluster_profiles_mp_worker.pyx'])

setup(name='ccluster',
      version='1.0',
      description \
        ='An extension implementing k-means and fast cluster profiling',
      cmdclass={'build_ext': build_ext},
      ext_modules=[profile_ext_module,
                   profile_mp_worker_ext_module])

