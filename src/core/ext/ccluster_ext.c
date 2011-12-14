/* ccluster_ext.c - C cluster extension module

This module implements the K-means clustering that partitions a set of
data vectors along a set of cluster vectors.


This software is distributed under the FreeBSD License.
See the accompanying file LICENSE for details.

Copyright 2011 Benjamin Hepp*/


#include <Python.h>

#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <omp.h>


static PyObject *cclusterError;

/* Returns a kmeans-clustering of all the N observations.
   points is a NxM matrix whereas each row is an observation
   k is the number of clusters
   the clustering stops if less than swap_threshold swaps have
   been performed in an iteration. */
static PyObject*
cluster_kmeans(PyObject *module, PyObject *args)
{
    PyThreadState *thread_state;
    PyObject *py_points, *py_clusters, *py_partition, *callback;
    PyObject *callback_result;
    PyObject *py_new_partition;
    int exception, i_callback_result;
    long int i, j, k;
    long int swap_threshold, minkowski_p;
    long int swaps, iterations;
    double *points, *clusters;
    long int *partition, *cluster_size, *new_partition;
    npy_intp *clusters_shape, *points_shape;
    npy_intp *partition_strides, *points_strides, *clusters_strides;
    npy_intp *new_partition_strides;

    thread_state = NULL;

    exception = 0;

    /* Parse function arguments from Python code */
    callback = NULL;
    if (!PyArg_ParseTuple(args, "OOOll|O:cluster_kmeans", &py_points, &py_clusters, &py_partition, &minkowski_p, &swap_threshold, &callback))
        return NULL;

    /* Convert numpy-arrays for use as contingous arrays */
    py_points = PyArray_FROM_OTF(py_points, NPY_DOUBLE, NPY_IN_ARRAY);
    if (py_points == NULL)
        return NULL;
    py_clusters = PyArray_FROM_OTF(py_clusters, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (py_clusters == NULL) {
        Py_DECREF(py_points);
        return NULL;
    }
    py_partition = PyArray_FROM_OTF(py_partition, NPY_LONG, NPY_OUT_ARRAY);
    if (py_partition == NULL) {
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        return NULL;
    }
    Py_XINCREF(callback);

    /* Check dimensions of py_points, py_clusters and py_partition */
    if (PyArray_NDIM(py_points) != 2) {
        PyErr_SetString(cclusterError, "points has to be a 2-dimensional array");
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }
    if (PyArray_NDIM(py_clusters) != 2) {
        PyErr_SetString(cclusterError, "clusters has to be a 2-dimensional array");
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }
    if (PyArray_NDIM(py_partition) != 1) {
        PyErr_SetString(cclusterError, "partition has to be a 1-dimensional array");
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }
    if (PyArray_DIMS(py_points)[1] != PyArray_DIMS(py_clusters)[1]) {
        PyErr_SetString(cclusterError,
                        "dimension 2 of points and clusters has to be the same size");
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }
    if (PyArray_DIMS(py_partition)[0] != PyArray_DIMS(py_points)[0]) {
        PyErr_SetString(cclusterError,
                        "dimension 1 of points and partition has to be the same size");
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }

    /* Create an array with the same shape as py_partition */
    /* PyArray_NewLikeArray only available in numpy >=1.6 */
    /* py_new_partition = PyArray_NewLikeArray((PyArrayObject*)py_partition) */
    py_new_partition = PyArray_NewCopy((PyArrayObject*)py_partition, 0);
    if (py_new_partition == NULL) {
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        return NULL;
    }
    Py_XINCREF(callback);

    /* Acquire data-pointers, shape and stride information of py_points and py_clusters */
    points = (double*)PyArray_DATA(py_points);
    points_strides = PyArray_STRIDES(py_points);
    points_shape = PyArray_DIMS(py_points);
    clusters = (double*)PyArray_DATA(py_clusters);
    clusters_strides = PyArray_STRIDES(py_clusters);
    clusters_shape = PyArray_DIMS(py_clusters);

    /* Acquire data-pointers, shape and stride information of py_partition and py_new_partition */
    partition = (long int*)PyArray_DATA(py_partition);
    partition_strides = PyArray_STRIDES(py_partition);
    new_partition = (long int*)PyArray_DATA(py_new_partition);
    new_partition_strides = PyArray_STRIDES(py_new_partition);
    /* Initialize all elements in py_partition to zero */
    PyArray_FILLWBYTE(py_partition, 0);

    /*printf("clusters_shape: %ld,%ld\n", clusters_shape[0], clusters_shape[1]);
    printf("points_shape: %ld,%ld\n", points_shape[0], points_shape[1]);
    printf("clusters_strides: %ld,%ld\n", clusters_strides[0], clusters_strides[1]);
    printf("points_strides: %ld,%ld\n", points_strides[0], points_strides[1]);
    printf("partition_strides: %ld\n", partition_strides[0]);
    printf("new_partition_strides: %ld\n", new_partition_strides[0]);*/

    /* Create an array that stores the number of vectors belonging to each cluster */
    cluster_size = (long int*)malloc(clusters_shape[0]*sizeof(long int));
    if (cluster_size == NULL) {
        Py_DECREF(callback);
        Py_DECREF(py_points);
        Py_DECREF(py_clusters);
        Py_DECREF(py_partition);
        Py_DECREF(py_new_partition);
        PyErr_SetString(PyExc_MemoryError, "malloc() for 'cluster_size' failed");
        return NULL;
    }

    if (callback != NULL) {
        /* callback was provided, check if it's callable. */
        if (!PyCallable_Check(callback)) {
            Py_DECREF(callback);
            callback = NULL;
        }
    }
    if (callback == NULL) {
        /* No callable object was provided, release GIL. */
        thread_state = PyEval_SaveThread();
    }

    iterations = 0;
    swaps = swap_threshold + 1;

    #pragma omp parallel
    while (swaps > swap_threshold)
    {
        double min_dist, dist, d;
        long int min_dist_index, cindex, pindex, cluster_id;
        int break_flag = 0;

        /* Recompute partitioning along clusters */
        #pragma omp for private(j,k,min_dist,min_dist_index,dist,cindex,pindex,d)
        for (i=0; i < points_shape[0]; i++) {
            min_dist = DBL_MAX;
            min_dist_index = -1;
            for (j=0; j < clusters_shape[0]; j++) {
                dist = 0.0;
                for (k=0; k < clusters_shape[1]; k++) {
                    cindex = j*clusters_shape[1] + k;
                    pindex = i*points_shape[1] + k;
                    d = points[pindex] - clusters[cindex];
                    dist += d * d;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_index = j;
                }
            }
            new_partition[i] = min_dist_index;
        }

        /* Give some feedback to the user */
        #pragma omp master
        if (iterations % 1 == 0) {
            if (callback != NULL) {
                callback_result = PyObject_CallFunction(callback, "ii", iterations, swaps);
                if (callback_result == NULL) {
                    swaps = swap_threshold;
                    exception = 1;
                    break_flag = 1;
                }
                if (!PyArg_Parse(callback_result, "i", &i_callback_result)) {
                    swaps = swap_threshold;
                    exception = 1;
                    break_flag = 1;
                }
                if (!i_callback_result) {
                    swaps = swap_threshold;
                    break_flag = 1;
                }
            }
            else {
                printf("\riteration %ld: swaps = %ld ... ", iterations, swaps );
                fflush(stdout);
            }
        }

        /* Count swaps */
        #pragma omp master
        {
            swaps = 0;
            for (i=0; i < points_shape[0]; i++) {
                if (partition[i] != \
                    new_partition[i]) {
                    swaps++;
                    partition[i] = new_partition[i];
                }
            }
        }

        /* calculate new cluster means. */

        /* reset cluster means and sizes to zero */
        #pragma omp for private(j)
        for (i=0; i < clusters_shape[0]; i++) {
            for (j=0; j < clusters_shape[1]; j++) {
                clusters[i*clusters_shape[1] + j] = 0.0;
            }
            cluster_size[i] = 0;
        }
        /* sum up the features of points belonging to each cluster
           and count the number of points per cluster. */
        /* #pragma omp for private(j,cluster_id,cindex,pindex) */
        #pragma omp single
        for (i=0; i < points_shape[0]; i++) {
            cluster_id = new_partition[i];
            cluster_size[cluster_id]++;
            for (j=0; j < points_shape[1]; ++j) {
                cindex = cluster_id*clusters_shape[1] + j;
                pindex = i*points_shape[1] + j;
                clusters[cindex] += points[pindex];
            }
        }
        /* Then divide by the number of points per cluster. */
        #pragma omp for private(j,cindex)
        for (i=0; i < clusters_shape[0]; i++) {
            for (j=0; j < clusters_shape[1]; j++) {
                cindex = i*clusters_shape[1] + j;
                clusters[cindex] /= cluster_size[i];
            }
        }

        #pragma omp master
        {
            if (break_flag)
                /* callback has signaled to stop the clustering. */
                swaps = swap_threshold;
            iterations++;
        }
        #pragma omp barrier
    }

    if (callback == NULL)
        /* If it was released, acquire the GIL */
        PyEval_RestoreThread(thread_state);

    free(cluster_size);

    Py_XDECREF(callback);
    Py_DECREF(py_partition);
    Py_DECREF(py_new_partition);
    Py_DECREF(py_clusters);
    Py_DECREF(py_points);

    if (exception)
        return NULL;

    if (callback == NULL) {
        printf("done\n");
        fflush(stdout);
    }

    Py_RETURN_NONE;
}

static PyMethodDef cclusterMethods[] = {
    {"kmeans", cluster_kmeans, METH_VARARGS,
     "kmeans(points, clusters, partition,\n"
     "       minkowski_p, swap_threshold, callback=None)\n"
     "Run k-means clustering.\n"
     "  points is a NxM float array of N points with M features.\n"
     "  clusters is a CxM float array of C initial clusters.\n"
     "  partition is a 1-dimensional integer array of size N.\n"
     "  minkowski_p is the minkowski parameter to use for distance computation.\n"
     "The algorithm stops when less than 'swap_threshold' points have changed\n"
     "their cluster in an iteration. When callback is not NULL, it is called\n"
     "on every 10 iterations with the two integer arguments (iteration,swaps).\n"
     "If callback is NULL then function will release the GIL for the computation.\n"
    },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initccluster(void)
{
    PyObject *m;

    m = Py_InitModule("ccluster", cclusterMethods);
    if (m == NULL)
        return ;

    cclusterError = PyErr_NewException("ccluster.error", NULL, NULL);
    Py_INCREF(cclusterError);
    PyModule_AddObject(m, "error", cclusterError);

    /* Make sure the C-API of numpy is safe to use. */
    import_array();
}
