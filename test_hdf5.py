import src.core.ioutil.import_export_hdf5 as hdf5

pdc = hdf5.import_hdf5_results(
    'yaca_data/HCM_new_results_29.04.2011.hdf5',
    [ 'yaca_data/new_data_noc_bfa2_results_27.04.2011.hdf5' ]
)

