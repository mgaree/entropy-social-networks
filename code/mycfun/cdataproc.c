#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "C versions of data processing for entropy experiment";
static char relative_entropy_docstring[] = "given Nxt array of integer bin ids, compute relative entropy vs. uniform dist.";
static char mutual_information_docstring[] = "given Nxt array of integer bin ids and edges, compute mutual information for all provided edges.";
static char transfer_entropy_docstring[] = "given Nxt array of integer bin ids and edges, compute transfer entropy for all provided edges.";

/* Declare the C functions here. */
static PyObject *relative_entropy_for_replication(PyObject *self, PyObject *args);
static PyObject *mutual_information_for_replication(PyObject *self, PyObject *args);
static PyObject *transfer_entropy_for_replication(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"relative_entropy_for_replication", relative_entropy_for_replication, METH_VARARGS, relative_entropy_docstring},
    {"mutual_information_for_replication", mutual_information_for_replication, METH_VARARGS, mutual_information_docstring},
    {"transfer_entropy_for_replication", transfer_entropy_for_replication, METH_VARARGS, transfer_entropy_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(cdataproc)
{
    PyObject *m;
    MOD_DEF(m, "cdataproc", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

//
// from Python Scripting for Computational Science, modified
//
#ifndef __NUMPY_MACROS_H__
#define __NUMPY_MACROS_H__

#define QUOTE(s) # s    /* turn s into string "s" */

#define NDIM_CHECK(a, expected_ndim) \
    if (PyArray_NDIM(a) != expected_ndim) { \
        PyErr_Format(PyExc_ValueError, \
                     "%s array is %d-dimensional, but expected to be %d-dimensional",\
                     QUOTE(a), PyArray_NDIM(a), expected_ndim); \
        return NULL; \
    }

#define DIM_CHECK(a, dim, expected_length) \
    if (dim > PyArray_NDIM(a)) { \
        PyErr_Format(PyExc_ValueError, \
                     "%s array has no %d dimension (max dim is %d)", \
                     QUOTE(a), dim, PyArray_NDIM(a)); \
        return NULL; \
    } \
    if (PyArray_DIM(a, dim) != expected_length) { \
        PyErr_Format(PyExc_ValueError, \
                     "%s array has wrong %d-dimension (wanted %d, got %d)", \
                     QUOTE(a), dim, expected_length, PyArray_DIM(a, dim)); \
        return NULL; \
    }

#define TYPE_CHECK(a, tp) \
    if (PyArray_TYPE(a) != tp) { \
        PyErr_Format(PyExc_TypeError, \
                     "%s array is not of correct type (want %d, got %d)", \
                     QUOTE(a), tp, PyArray_TYPE(a)); \
        return NULL; \
    }

#define DIND1(a, i)  *((double *) PyArray_GETPTR1(a, i))
#define DIND2(a, i, j)  *((double *) PyArray_GETPTR2(a, i, j))
#define DIND3(a, i, j, k)  *((double *) PyArray_GETPTR3(a, i, j, k))
#define IIND1(a, i)  *((int *) PyArray_GETPTR1(a, i))
#define IIND2(a, i, j)  *((int *) PyArray_GETPTR2(a, i, j))
#define IIND3(a, i, j, k)  *((int *) PyArray_GETPTR3(a, i, j, k))

#endif

// the actual functions


static PyObject *relative_entropy_for_replication(PyObject *self, PyObject *args) {
// works! vs. what I use for data_processors.py:relative_entropy_for_replication, I do need to prepare
// the data by digitizing it, but this method is 95% faster (for 10Kx500, 595ms vs 16s)
  PyArrayObject *count_array, *input_data_array, *output_data_array;

  npy_intp dims[1];
  int response_var_num, num_bins, N, t_max;
  double q, p, plogp;
  double *output_data_itrv;
  int *count_i;
  int bin_i, jc, i, t, count_val;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!iO!i:relative_entropy_for_replication",
        &PyArray_Type, &input_data_array,
        &num_bins,
        &PyArray_Type, &output_data_array,
        &response_var_num)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* How many data points are there? */
  N = (int)PyArray_DIM(input_data_array, 0);
  t_max = (int)PyArray_DIM(input_data_array, 1);

  q = 1.0 / num_bins;

  /* Build arrays */
  dims[0] = num_bins;
  count_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT);  // 1d pmf/counts

  // error trapping on array creation
  if (count_array == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build array");
    Py_XDECREF(count_array);
    return NULL;
  }

  PyArray_FILLWBYTE(count_array, 0);

  // consistency checks before calculations begin
  NDIM_CHECK(input_data_array, 2);  // N, t
  TYPE_CHECK(input_data_array, NPY_INT32);

  NDIM_CHECK(output_data_array, 3);  // N, t, RV
  TYPE_CHECK(output_data_array, NPY_DOUBLE);
  DIM_CHECK(output_data_array, 0, N);
  DIM_CHECK(output_data_array, 1, t_max);

  NDIM_CHECK(count_array, 1);
  DIM_CHECK(count_array, 0, num_bins);

  // the actual function
  Py_BEGIN_ALLOW_THREADS

  for (i = 0; i < N; i++) {
      PyArray_FILLWBYTE(count_array, 0);  // reusing count_array for each agent

      for (t = 0; t < t_max; t++) {
          bin_i = IIND2(input_data_array, i, t);

          count_i = (int *) PyArray_GETPTR1(count_array, bin_i);
          *count_i += 1;

          // pointer to the output element we'll accumulate entropy to
          output_data_itrv = (double *) PyArray_GETPTR3(output_data_array, i, t, response_var_num);

          // iterate thru count_array
          for (jc = 0; jc < num_bins; jc++) {
              count_val = IIND1(count_array, jc);
              if (count_val >= 1) {
                  p = (double)(count_val) / (double)(t+1);  // typecasting is key here!
                  plogp = p * log2(p / q); // single entropy term in summation

                  *output_data_itrv += plogp;
              }
          }
      }
  }

  Py_END_ALLOW_THREADS

  /* Clean up. */
  Py_XDECREF(count_array);

  // return output_data_obj;
  return Py_BuildValue("");  // we're editing the output_data_array in place, so return none
}


static PyObject *mutual_information_for_replication(PyObject *self, PyObject *args) {
  // works! vs what I use for data_processors.py:mutual_information_for_replication,
  // this method is 93% faster (for 10Kx500, 20s vs 5min).
  // I do need to prepare the data by digitizing it and build edge arrays
  //
  // note: for N=100, I achieved isclose().all() matches for default atol. however, for N=10,000, I got differences in
  // 0.02 % of values; differences ranged from 2e-6 to 2e-3. I think this is negligible (less than 0.1% "error" in the [-1, 1] interval)
  // but intend to revisit it.
  PyArrayObject *count_array, *count_2d_array, *marginal_entropy_array;
  PyArrayObject *input_data_array, *output_data_array, *edges_array, *outdegrees_array;

  npy_intp dims[1], dims2[2];
  int N, t_max, response_var_num, num_bins, num_edges;
  int jc, i, t, count_val;
  int bin_i, bin_j, *count_i;
  int edge_i, edge_j, *count_xy, *outdegree_i;
  int e, xc, yc;

  double p, plogp, cur_sum_plogp;
  double *marginal_entropy_it, *marginal_entropy_jt, *output_data_itrv;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!iO!iO!O!:relative_entropy_for_replication",
        &PyArray_Type, &input_data_array,
        &num_bins,
        &PyArray_Type, &output_data_array,
        &response_var_num,
        &PyArray_Type, &edges_array,
        &PyArray_Type, &outdegrees_array)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* How many data points are there? */
  N = (int)PyArray_DIM(input_data_array, 0);
  t_max = (int)PyArray_DIM(input_data_array, 1);
  num_edges = (int)PyArray_DIM(edges_array, 0);  // network edges

  /* Build arrays */
  dims[0] = num_bins;
  count_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT);  // 1d pmf/counts

  dims2[0] = num_bins;
  dims2[1] = num_bins;
  count_2d_array = (PyArrayObject *) PyArray_SimpleNew(2, dims2, NPY_INT);

  // create one for marginal entropy; use existing output_data_array to accumulate mutual information (py:164)
  dims2[0] = N;
  dims2[1] = t_max;
  marginal_entropy_array = (PyArrayObject *) PyArray_SimpleNew(2, dims2, NPY_DOUBLE);

  // error trapping on array creation
  if (count_array == NULL || count_2d_array == NULL || marginal_entropy_array == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build an array");
    Py_XDECREF(count_array);
    Py_XDECREF(count_2d_array);
    Py_XDECREF(marginal_entropy_array);
    return NULL;
  }

  // make array interfaces to objects and zeroize
  PyArray_FILLWBYTE(count_array, 0);
  PyArray_FILLWBYTE(count_2d_array, 0);
  PyArray_FILLWBYTE(marginal_entropy_array, 0);

  // consistency checks before calculations begin
  NDIM_CHECK(input_data_array, 2);  // N, t
  TYPE_CHECK(input_data_array, NPY_INT32);

  NDIM_CHECK(output_data_array, 3);  // N, t, RV
  TYPE_CHECK(output_data_array, NPY_DOUBLE);
  DIM_CHECK(output_data_array, 0, N);
  DIM_CHECK(output_data_array, 1, t_max);

  NDIM_CHECK(outdegrees_array, 1);
  TYPE_CHECK(outdegrees_array, NPY_INT32);
  DIM_CHECK(outdegrees_array, 0, N);

  NDIM_CHECK(edges_array, 2);
  TYPE_CHECK(edges_array, NPY_INT32);
  DIM_CHECK(edges_array, 1, 2);  // the 0th dim is unknown (|E|)

  NDIM_CHECK(count_array, 1);
  DIM_CHECK(count_array, 0, num_bins);

  NDIM_CHECK(count_2d_array, 2);
  DIM_CHECK(count_2d_array, 0, num_bins);

  NDIM_CHECK(marginal_entropy_array, 2);
  DIM_CHECK(marginal_entropy_array, 0, N);
  DIM_CHECK(marginal_entropy_array, 1, t_max);

  // the actual function
  Py_BEGIN_ALLOW_THREADS

  for (i = 0; i < N; i++) {
      PyArray_FILLWBYTE(count_array, 0);  // reusing count_array for each agent

      for (t = 0; t < t_max; t++) {
          bin_i = IIND2(input_data_array, i, t);

          count_i = (int *) PyArray_GETPTR1(count_array, bin_i);
          *count_i += 1;

          // pointer to the output element we'll accumulate entropy to
          marginal_entropy_it = (double *) PyArray_GETPTR2(marginal_entropy_array, i, t);

          // iterate thru count_array
          for (jc = 0; jc < num_bins; jc++) {
              count_val = IIND1(count_array, jc);
              if (count_val >= 1) {
                  p = (double)(count_val) / (double)(t+1);  // typecasting is key here!
                  plogp = p * log2(p); // single entropy term in summation

                  *marginal_entropy_it += plogp;
              }
          }
      }
  }

  // joint entropy term + MI calculation

  for (e = 0; e < num_edges; e++) {
      // the two nodes on either end of the current edge e
      PyArray_FILLWBYTE(count_2d_array, 0);

      edge_i = IIND2(edges_array, e, 0);
      edge_j = IIND2(edges_array, e, 1);

      for (t = 0; t < t_max; t++) {
          bin_i = IIND2(input_data_array, edge_i, t);
          bin_j = IIND2(input_data_array, edge_j, t);

          count_xy = (int *) PyArray_GETPTR2(count_2d_array, bin_i, bin_j);
          *count_xy += 1;

          cur_sum_plogp = 0;  // reset accumulator

          // // iterate thru count_2d_array
          for (xc = 0; xc < num_bins; xc++) {
              for (yc = 0; yc < num_bins; yc++) {
                  count_val = IIND2(count_2d_array, xc, yc);
                  if (count_val >= 1) {
                      p = (double)(count_val) / (double)(t+1);
                      plogp = p * log2(p);

                      cur_sum_plogp += plogp;  // np.sum(p_ijt * log_p_ijt)
                  }
              }
          }
          output_data_itrv = (double *) PyArray_GETPTR3(output_data_array, edge_i, t, response_var_num);
          marginal_entropy_it = (double *) PyArray_GETPTR2(marginal_entropy_array, edge_i, t);
          marginal_entropy_jt = (double *) PyArray_GETPTR2(marginal_entropy_array, edge_j, t);

          // MI calculation; accumulate for now, then divide by outdegree at the end
          *output_data_itrv += cur_sum_plogp - (*marginal_entropy_it) - (*marginal_entropy_jt);
      }
  }

  // average MI per agent based on their outdegree
  for (i = 0; i < N; i++) {
      for (t = 0; t < t_max; t++) {
          outdegree_i = (int *) PyArray_GETPTR1(outdegrees_array, i);
          if (*outdegree_i < 1) { continue; }  // avoid div by zero for nodes w/ no out-neighbors

          output_data_itrv = (double *) PyArray_GETPTR3(output_data_array, i, t, response_var_num);
          *output_data_itrv /= (double)(*outdegree_i);  // TOOD maybe change all doubles to floats to match int32
      }
  }

  Py_END_ALLOW_THREADS

  /* Clean up. */
  Py_XDECREF(count_array);
  Py_XDECREF(count_2d_array);
  Py_XDECREF(marginal_entropy_array);

  // return output_data_obj;
  return Py_BuildValue("");  // we're editing the output_data_array in place, so return none
}


static PyObject *transfer_entropy_for_replication(PyObject *self, PyObject *args) {
// works! vs what I use for data_processors.py:transfer_entropy_for_replication,
// this method is 88% faster (for 10Kx500, 1.6min vs 14.25min). interesting that the % faster is decreasing with complexity
// I do need to prepare the data by digitizing it and build edge arrays
//
// note: for N=100, I achieved isclose().all() matches for default atol. however, for N=10,000, I got differences in
// 420 out of 5e6 values; differences ranged from 8e-4 to 2e-2. These "errors" are somewhat more concerning
// than those from mutual information notes, but interestingly, the errors affected only 1 agent.
// the correlation between time index and the "error" is 0.73 for one particular test case.
// Will revisit, but this is less than 1% error in the [-1, 1] interval.
  PyArrayObject *count_array, *count_2d_array, *count_3d_array, *marginal_entropy_array, *i1it_entropy_array;
  PyArrayObject *input_data_array, *output_data_array, *edges_array, *outdegrees_array;

  npy_intp dims[1], dims2[2], dims3[3];
  int N, t_max, response_var_num, num_bins, num_edges;

  int jc, i, t, count_val;
  int bin_i, bin_i1, bin_j, *count_i;
  int e, xc, yc, zc;
  int edge_i, edge_j, *count_yz, *count_xyz, *outdegree_i;
  int *count_i1i;

  double p, plogp, cur_sum_plogp_2d, cur_sum_plogp_3d;
  double *marginal_entropy_it, *i1it_entropy_it, *output_data_itrv;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "O!iO!iO!O!:relative_entropy_for_replication",
        &PyArray_Type, &input_data_array,
        &num_bins,
        &PyArray_Type, &output_data_array,
        &response_var_num,
        &PyArray_Type, &edges_array,
        &PyArray_Type, &outdegrees_array)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* How many data points are there? */
  N = (int)PyArray_DIM(input_data_array, 0);
  t_max = (int)PyArray_DIM(input_data_array, 1);
  num_edges = (int)PyArray_DIM(edges_array, 0);  // network edges

  /* Build arrays */
  dims[0] = num_bins;
  count_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT);  // 1d pmf/counts

  dims2[0] = num_bins;
  dims2[1] = num_bins;
  count_2d_array = (PyArrayObject *) PyArray_SimpleNew(2, dims2, NPY_INT);

  dims3[0] = num_bins;
  dims3[1] = num_bins;
  dims3[2] = num_bins;
  count_3d_array = (PyArrayObject *) PyArray_SimpleNew(3, dims3, NPY_INT);

  dims2[0] = N;
  dims2[1] = t_max;
  marginal_entropy_array = (PyArrayObject *) PyArray_SimpleNew(2, dims2, NPY_DOUBLE);

  dims2[0] = N;
  dims2[1] = t_max-1;  // -1 because of lag values
  i1it_entropy_array = (PyArrayObject *) PyArray_SimpleNew(2, dims2, NPY_DOUBLE);

  // error trapping on array creation
  if (count_array == NULL || count_2d_array == NULL || marginal_entropy_array == NULL ||
        count_3d_array == NULL || i1it_entropy_array == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build an array");
    Py_XDECREF(count_array);
    Py_XDECREF(count_2d_array);
    Py_XDECREF(count_3d_array);
    Py_XDECREF(marginal_entropy_array);
    Py_XDECREF(i1it_entropy_array);
    return NULL;
  }

  // make array interfaces to objects and zeroize
  PyArray_FILLWBYTE(count_array, 0);
  PyArray_FILLWBYTE(count_2d_array, 0);
  PyArray_FILLWBYTE(count_3d_array, 0);
  PyArray_FILLWBYTE(marginal_entropy_array, 0);
  PyArray_FILLWBYTE(i1it_entropy_array, 0);

  // consistency checks before calculations begin
  NDIM_CHECK(input_data_array, 2);  // N, t
  TYPE_CHECK(input_data_array, NPY_INT32);

  NDIM_CHECK(output_data_array, 3);  // N, t, RV
  TYPE_CHECK(output_data_array, NPY_DOUBLE);
  DIM_CHECK(output_data_array, 0, N);
  DIM_CHECK(output_data_array, 1, t_max);

  NDIM_CHECK(outdegrees_array, 1);
  TYPE_CHECK(outdegrees_array, NPY_INT32);
  DIM_CHECK(outdegrees_array, 0, N);

  NDIM_CHECK(edges_array, 2);
  TYPE_CHECK(edges_array, NPY_INT32);
  DIM_CHECK(edges_array, 1, 2);  // the 0th dim is unknown (|E|)

  NDIM_CHECK(count_array, 1);
  DIM_CHECK(count_array, 0, num_bins);

  NDIM_CHECK(count_2d_array, 2);
  DIM_CHECK(count_2d_array, 0, num_bins);

  NDIM_CHECK(count_3d_array, 3);
  DIM_CHECK(count_3d_array, 0, num_bins);

  NDIM_CHECK(marginal_entropy_array, 2);
  DIM_CHECK(marginal_entropy_array, 0, N);
  DIM_CHECK(marginal_entropy_array, 1, t_max);

  NDIM_CHECK(i1it_entropy_array, 2);
  DIM_CHECK(i1it_entropy_array, 0, N);
  DIM_CHECK(i1it_entropy_array, 1, t_max-1);

   // the actual function
  Py_BEGIN_ALLOW_THREADS

  for (i = 0; i < N; i++) {
      PyArray_FILLWBYTE(count_array, 0);  // reusing count_array for each agent

      for (t = 0; t < t_max; t++) {
          bin_i = IIND2(input_data_array, i, t);

          count_i = (int *) PyArray_GETPTR1(count_array, bin_i);
          *count_i += 1;

          // pointer to the output element we'll accumulate entropy to
          marginal_entropy_it = (double *) PyArray_GETPTR2(marginal_entropy_array, i, t);

          // // iterate thru count_array
          for (jc = 0; jc < num_bins; jc++) {
              count_val = IIND1(count_array, jc);
              if (count_val >= 1) {
                  p = (double)(count_val) / (double)(t+1);  // typecasting is key here!
                  plogp = p * log2(p);

                  *marginal_entropy_it += plogp;
              }
          }
      }
  }
  // i1it entropy terms
  for (i = 0; i < N; i++) {
      PyArray_FILLWBYTE(count_2d_array, 0);

      for (t = 0; t < t_max-1; t++) { // -1 because lag term
          bin_i1 = IIND2(input_data_array, i, t+1);
          bin_i = IIND2(input_data_array, i, t);

          count_i1i = (int *) PyArray_GETPTR2(count_2d_array, bin_i1, bin_i);
          *count_i1i += 1;

          // pointer to the output element we'll accumulate entropy to
          i1it_entropy_it = (double *) PyArray_GETPTR2(i1it_entropy_array, i, t);

          // iterate thru count_2d_array
          for (xc = 0; xc < num_bins; xc++) {
              for (yc = 0; yc < num_bins; yc++) {
                  count_val = IIND2(count_2d_array, xc, yc);
                  if (count_val >= 1) {
                      p = (double)(count_val) / (double)(t+1);
                      plogp = p * log2(p);

                      *i1it_entropy_it += plogp;  // np.sum(p_i1it * log_p_i1it)
                  }
              }
          }
      }
  }

  // 3d joint entropy term + TE calculation

  for (e = 0; e < num_edges; e++) {
      PyArray_FILLWBYTE(count_2d_array, 0);
      PyArray_FILLWBYTE(count_3d_array, 0);

      // the two nodes on either end of the current edge e
      edge_i = IIND2(edges_array, e, 0);
      edge_j = IIND2(edges_array, e, 1);

      for (t = 0; t < t_max-1; t++) {  // -1 again because of the lag term
          bin_i1 = IIND2(input_data_array, edge_i, t+1);
          bin_i = IIND2(input_data_array, edge_i, t);
          bin_j = IIND2(input_data_array, edge_j, t);

          count_xyz = (int *) PyArray_GETPTR3(count_3d_array, bin_i1, bin_i, bin_j);
          *count_xyz += 1;

          count_yz = (int *) PyArray_GETPTR2(count_2d_array, bin_i, bin_j);
          *count_yz += 1;

          cur_sum_plogp_2d = 0; // np.sum(p_ijt * np.log2(p_ijt))
          cur_sum_plogp_3d = 0; // np.sum(p_i1ijt * np.log2(p_i1ijt))

          // iterate thru count_2d_array
          for (xc = 0; xc < num_bins; xc++) {
              for (yc = 0; yc < num_bins; yc++) {
                  count_val = IIND2(count_2d_array, xc, yc);
                  if (count_val >= 1) {
                      p = (double)(count_val) / (double)(t+1);  // typecasting is key here!
                      plogp = p * log2(p);

                      cur_sum_plogp_2d += plogp;  // np.sum(p_ijt * log_p_ijt)
                  }
              }
          }

          // iterate thru count_3d_array
          for (xc = 0; xc < num_bins; xc++) {
              for (yc = 0; yc < num_bins; yc++) {
                  for (zc = 0; zc < num_bins; zc++) {
                      count_val = IIND3(count_3d_array, xc, yc, zc);
                      if (count_val >= 1) {
                          p = (double)(count_val) / (double)(t+1);  // typecasting is key here!
                          plogp = p * log2(p);

                          cur_sum_plogp_3d += plogp;  // np.sum(p_i1ijt * np.log2(p_i1ijt))
                      }
                  }
              }
          }
          output_data_itrv = (double *) PyArray_GETPTR3(output_data_array, edge_i, t, response_var_num);
          marginal_entropy_it = (double *) PyArray_GETPTR2(marginal_entropy_array, edge_i, t);
          i1it_entropy_it = (double *) PyArray_GETPTR2(i1it_entropy_array, edge_i, t);

          // TE calculation; accumulate for now, then divide by outdegree at the end
          *output_data_itrv += cur_sum_plogp_3d - (*i1it_entropy_it) - cur_sum_plogp_2d + (*marginal_entropy_it);
      }
  }

  // average MI per agent based on their outdegree
  for (i = 0; i < N; i++) {
      for (t = 0; t < t_max; t++) {
          outdegree_i = (int *) PyArray_GETPTR1(outdegrees_array, i);
          if (*outdegree_i < 1) { continue; }  // avoid div by zero for nodes w/ no out-neighbors

          output_data_itrv = (double *) PyArray_GETPTR3(output_data_array, i, t, response_var_num);
          *output_data_itrv /= (float)(*outdegree_i);
      }
  }

  Py_END_ALLOW_THREADS

  /* Clean up. */
  Py_XDECREF(count_array);
  Py_XDECREF(count_2d_array);
  Py_XDECREF(count_3d_array);
  Py_XDECREF(marginal_entropy_array);
  Py_XDECREF(i1it_entropy_array);

  // return output_data_obj;
  return Py_BuildValue("");  // we're editing the output_data_array in place, so return none
}
