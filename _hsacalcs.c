/*
  C helper module for intensive calcs.
#=============================================================================================
# VERSION CONTROL INFORMATION
#=============================================================================================
__version__ = "$Revision: $ $Date: $"
# $Date: $
# $Revision: $
# $LastChangedBy: $
# $HeadURL: $
# $Id: $

#=============================================================================================

#=============================================================================================
# INSTALLATION INSTRUCTIONS
#=============================================================================================

  To compile on Linux:

  gcc -O3 -lm -fPIC -shared -I(directory with Python.h) -I(directory with numpy/arrayobject.h) -o _hsacalcs.so _hsacalcs.c

  For a desmond installation of python 2.5 (change path up to desmond directory, rest should be the same):
  
  gcc -O3 -lm -fPIC -shared -I /opt/schrodinger2014-2/mmshare-v26017/lib/Linux-x86_64/include/python2.7/ -I /opt/schrodinger2014-2/mmshare-v26017/lib/Linux-x86_64/lib/python2.7/site-packages/numpy/core/include/ -o _hsacalcs.so _hsacalcs.c
  
  For a default installation of python 2.5:
  
  gcc -O3 -lm -fPIC -shared -I/usr/local/include/python2.5 -I/usr/local/lib/python2.5/site-packages/numpy/core/include -o _hsacalcs.so _hsacalcs.c

*/

#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>

double dist_mic(double x1, double x2, double x3, double y1, double y2, double y3, double b1, double b2, double b3) {
    /* Method for obtaining inter atom distance using minimum image convention
     */
    double dx, dy, dz;
    dx = x1-y1;
    dy = x2-y2;
    dz = x3-y3;
    if (dx > b1/2.0) dx -= b1; 
    else if (dx < -b1/2.0) dx += b1; 
    if (dy > b2/2.0) dy -= b2;
    else if (dy < -b2/2.0) dy += b2;
    if (dz > b3/2.0) dz -= b3; 
    else if (dz < -b3/2.0) dz += b3;

    return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
    }
    
double dist(double x1, double x2, double x3, double y1, double y2, double y3) {
    /* Method for Euclidean distance between two points
     */
    double dx, dy, dz;
    dx = x1-y1;
    dy = x2-y2;
    dz = x3-y3;
    return sqrt(pow(dx, 2)+ pow(dy, 2)+ pow(dz, 2));
    }


PyObject *_gistcalcs_elecE( PyObject *self, PyObject *args) // we don't have a class here but still need *self argument
    /* Method for calculation electrostatic energy between a water molecule and a set of other molecules
    */
    {
    // First we declare variables to be used in this function
    npy_intp m, n;
    int i, j;
    int *wat, *other;
    double *b_x, *b_y, *b_z;
    double *wx, *wy, *wz, *wc;
    double *sx, *sy, *sz, *sc;
    double d;
    double e_elec = 0.0;
    // These variables are declared to store Python objects, which will arrive here wrapped in args tuple
    PyArrayObject *wat_at_ids, *other_at_ids, *coords, *charges, *box;
    // Here we parse the args tuple as it contains all the variables sent from Python
    // In this case Python sent five Numpy arrays (hence five times O!, O means a python object, ! is a pointer check)
    // name of the function as called from Python is also in the string, will be used in locating errors
    // for each O!, we need to supply a pointer type and a pointer
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!:elecE",
        &PyArray_Type, &wat_at_ids,
        &PyArray_Type, &other_at_ids,
        &PyArray_Type, &coords,
        &PyArray_Type, &charges,
        &PyArray_Type, &box))
        {
            return NULL; /* raise argument parsing exception*/
        }
    
    // A consistency check for correct data type being sent from Python
    if (PyArray_NDIM(coords) != 2)
        {
            PyErr_Format(PyExc_ValueError, "coordinate array is not of correct dimensions or type");
            return NULL;
            
        }
        
    m = PyArray_DIM(wat_at_ids, 0); // m is indec over water atoms
    n = PyArray_DIM(other_at_ids, 0); // n is indec over all other atoms
    // periodic box information
    b_x = (double *) PyArray_GETPTR1(box, 0); 
    b_y = (double *) PyArray_GETPTR1(box, 1);
    b_z = (double *) PyArray_GETPTR1(box, 2);
    
    // loop over each water atom and over each other atom
    for (i = 0; i < m; i++) {
        wat = (int *) PyArray_GETPTR1(wat_at_ids, i); // obtain index for this atom (this is not array index, this is unique atom id)
        wx = PyArray_GETPTR2(coords, *wat-1, 0); // use wat to get the correct x, y, z coordinates from coord array
        wy = PyArray_GETPTR2(coords, *wat-1, 1); 
        wz = PyArray_GETPTR2(coords, *wat-1, 2);
        wc = PyArray_GETPTR1(charges, *wat); // use wat to get the correct charge from charge array
        for (j = 0; j < n; j++) {
            other = (int *) PyArray_GETPTR1(other_at_ids, j);
            sx = PyArray_GETPTR2(coords, *other-1, 0); // obtain index of this atom
            sy = PyArray_GETPTR2(coords, *other-1, 1); // obtain x, y, z
            sz = PyArray_GETPTR2(coords, *other-1, 2);
            sc = PyArray_GETPTR1(charges, *other); // charge on this atom
            d = dist_mic(*wx, *wy, *wz, *sx, *sy, *sz, *b_x, *b_y, *b_z); // distance (based on minimum image convention)
            e_elec += ((*wc)*(*sc))/d; // Coulombic interaction calculation
        }
        
    }
    return Py_BuildValue("f", e_elec);
}

PyObject *_gistcalcs_vdwE( PyObject *self, PyObject *args)
    {
    npy_intp m, n;
    int i, j;
    int *wat, *other;
    double *b_x, *b_y, *b_z;
    double *wx, *wy, *wz, *wc, *w_sig, *w_eps;
    double *sx, *sy, *sz, *sc, *s_sig, *s_eps;
    double comb_sig, comb_eps;
    double d, dist6, dist12, aij, bij;
    double e_vdw = 0.0;

    PyArrayObject *wat_at_ids, *other_at_ids, *coords, *vdwparms, *box;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!:elecE",
        &PyArray_Type, &wat_at_ids,
        &PyArray_Type, &other_at_ids,
        &PyArray_Type, &coords,
        &PyArray_Type, &vdwparms,
        &PyArray_Type, &box))
        {
            return NULL; /* raise argument parsing exception*/
        }

    if (PyArray_NDIM(coords) != 2)
        {
            PyErr_Format(PyExc_ValueError, "coordinate array is not of correct dimensions or type");
            return NULL;
            
        }
    if (PyArray_NDIM(vdwparms) != 2)
        {
            PyErr_Format(PyExc_ValueError, "vdw parmeter array is not of correct dimensions or type");
            return NULL;
            
        }

    m = PyArray_DIM(wat_at_ids, 0);
    n = PyArray_DIM(other_at_ids, 0);
    
    b_x = (double *) PyArray_GETPTR1(box, 0);
    b_y = (double *) PyArray_GETPTR1(box, 1);
    b_z = (double *) PyArray_GETPTR1(box, 2);
    
    for (i = 0; i < m; i++) {
        wat = (int *) PyArray_GETPTR1(wat_at_ids, i);
        wx = PyArray_GETPTR2(coords, *wat-1, 0);
        wy = PyArray_GETPTR2(coords, *wat-1, 1);
        wz = PyArray_GETPTR2(coords, *wat-1, 2);
        w_sig = PyArray_GETPTR2(vdwparms, *wat, 0);
        w_eps = PyArray_GETPTR2(vdwparms, *wat, 1);
        
        for (j = 0; j < n; j++) {
            other = (int *) PyArray_GETPTR1(other_at_ids, j);
            sx = PyArray_GETPTR2(coords, *other-1, 0);
            sy = PyArray_GETPTR2(coords, *other-1, 1);
            sz = PyArray_GETPTR2(coords, *other-1, 2);
            s_sig = PyArray_GETPTR2(vdwparms, *other, 0);
            s_eps = PyArray_GETPTR2(vdwparms, *other, 1);
            comb_sig = (*w_sig + *s_sig)/2;
            comb_eps = sqrt((*w_eps)*(*s_eps));
            d = dist_mic(*wx, *wy, *wz, *sx, *sy, *sz, *b_x, *b_y, *b_z);
            dist6 = pow(d, 6);
            dist12 = dist6 * dist6;
            aij = 4*comb_eps*pow(comb_sig, 12);
            bij = -4*comb_eps*pow(comb_sig, 6);
            e_vdw +=  (aij/dist12)+(bij/dist6);
        }
        
    }
    return Py_BuildValue("f", e_vdw);
}

PyObject *_gistcalcs_nbr_E_ww( PyObject *self, PyObject *args)
    {

    int i, m, o, p;
    int wat_O;
    double *b_x, *b_y, *b_z; // variables to store box info
    double *w1x, *w1y, *w1z, *h1x, *h1y, *h1z, *w1_sig, *w1_eps; // variables to store water molecule's coordinates (separately for O and H-atoms)
    double *w1c, *h1c; // variables to store water molecule's non-bonded params 
    double *w2x, *w2y, *w2z, *h2x, *h2y, *h2z, *w2_sig, *w2_eps; // variables to store water molecule's coordinates (separately for O and H-atoms)
    double *w2c, *h2c; // variables to store water molecule's non-bonded params 
    double comb_eps, comb_sig, aij, bij, dist6, dist12, d_oo, d_oh, d_hh; // distances 

    //const double aij_wat = 581935.564;
    //const double bij_wat = -594.825035;

    PyArrayObject *other_at_ids, *coords, *vdwparms, *charges, *box, *nbr_energies;
    
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!:nbr_E_ww",
        &wat_O,
        &PyArray_Type, &other_at_ids,
        &PyArray_Type, &coords,
        &PyArray_Type, &vdwparms,
        &PyArray_Type, &charges,
        &PyArray_Type, &box,
        &PyArray_Type, &nbr_energies))
        {
            return NULL; /* raise argument parsing exception*/
        }

    //m = PyArray_DIM(wat_at_ids, 0);
    m = PyArray_DIM(other_at_ids, 0);
    
    b_x = (double *) PyArray_GETPTR1(box, 0);
    b_y = (double *) PyArray_GETPTR1(box, 1);
    b_z = (double *) PyArray_GETPTR1(box, 2);
    // iterate over each neighbor oxygen atom
    
    w1x = PyArray_GETPTR2(coords, wat_O-1, 0); //
    w1y = PyArray_GETPTR2(coords, wat_O-1, 1); // obtain x, y, z
    w1z = PyArray_GETPTR2(coords, wat_O-1, 2);
    w1c = PyArray_GETPTR1(charges, wat_O); // charge on this atom
    w1_sig = PyArray_GETPTR2(vdwparms, wat_O, 0);
    w1_eps = PyArray_GETPTR2(vdwparms, wat_O, 1);

    //printf("water atom ID x y z charge: %i %5.3f %5.3f %5.3f %5.3f \n", wat_O, *w1x, *w1y, *w1z, *w1c);
    // for each neighbor water
    for (i = 0; i < m; i++) {
        // initialize variable to hold it's total energy
        double E_wat = 0.0;
        int *target_O_at, *target_H_at;

        // obtain its index
        target_O_at = (int *) PyArray_GETPTR1(other_at_ids, i);
        // obtain its coords and charge
        w2x = PyArray_GETPTR2(coords, *target_O_at-1, 0); //
        w2y = PyArray_GETPTR2(coords, *target_O_at-1, 1); // obtain x, y, z
        w2z = PyArray_GETPTR2(coords, *target_O_at-1, 2);
        w2c = PyArray_GETPTR1(charges, *target_O_at); // charge on this atom
        w2_sig = PyArray_GETPTR2(vdwparms, *target_O_at, 0);
        w2_eps = PyArray_GETPTR2(vdwparms, *target_O_at, 1);
        //printf("water atom ID x y z charge: %i %5.3f %5.3f %5.3f %5.3f \n", *target_O_at, *w2x, *w2y, *w2z, *w2c);

        // we first calculate this neighbor water molecule's oxygen non-bonded interactions (both vdw and elec) with current water atom
        d_oo = dist_mic(*w1x, *w1y, *w1z, *w2x, *w2y, *w2z, *b_x, *b_y, *b_z); // distance (based on minimum image convention)
        dist6 = pow(d_oo, 6);
        dist12 = dist6 * dist6;
        
        // Coulombic interaction calculation
        // Coulombic interaction calculation
        comb_sig = (*w1_sig + *w2_sig)/2.0;
        comb_eps = sqrt((*w1_eps)*(*w2_eps));
        aij = 4*comb_eps*pow(comb_sig, 12);
        bij = -4*comb_eps*pow(comb_sig, 6);
        //*(double *)PyArray_GETPTR2(voxel_data, v_id, 13) += ((*w1c)*(*w2c))/d_oo; 
        //*(double *)PyArray_GETPTR2(voxel_data, v_id, 13) += (aij_wat/dist12)+(bij_wat/dist6);
        
        E_wat += ((*w1c)*(*w2c))/d_oo;
        //E_wat += (aij_wat/dist12)+(bij_wat/dist6);
        E_wat += (aij/dist12)+(bij/dist6);
        // for each nbr oxygen go over its hydrogen
        for (p = wat_O+1; p <= wat_O+2; p++){
            //printf("processing water atom: %i\n", m);
            h1x = PyArray_GETPTR2(coords, p-1, 0); // use wat to get the correct x, y, z coordinates from coord array
            h1y = PyArray_GETPTR2(coords, p-1, 1); 
            h1z = PyArray_GETPTR2(coords, p-1, 2);
            h1c = PyArray_GETPTR1(charges, p); // use wat to get the correct charge from charge array
            //printf("processing hydrogen %i of water %i x y z charge: %5.3f %5.3f %5.3f %5.3f\n", p, wat_O, *h1x, *h1y, *h1z, *h1c);
            d_oh = dist_mic(*h1x, *h1y, *h1z, *w2x, *w2y, *w2z, *b_x, *b_y, *b_z); // distance (based on minimum image convention)
            //*(double *)PyArray_GETPTR2(voxel_data, v_id, 13) += ((*h1c)*(*w2c))/d_oh; // Coulombic interaction calculation
            E_wat += ((*h1c)*(*w2c))/d_oh;
            } // end looping over water hydrogen atoms
            
        //printf("Hydrogen atom indices belonging to this water %i are %i %i\n", *target_O_at, *target_O_at+1, *target_O_at+2);
        // now we go over hydrogen atoms (since they only contribute to electrostatic interactions)
        for (o = *target_O_at+1; o <= *target_O_at+2; o++){
            //printf("processing hydrogen atom: %i\n", o);
            target_H_at = (int *) PyArray_GETPTR1(other_at_ids, o);
            h2x = PyArray_GETPTR2(coords, o-1, 0); // use wat to get the correct x, y, z coordinates from coord array
            h2y = PyArray_GETPTR2(coords, o-1, 1); 
            h2z = PyArray_GETPTR2(coords, o-1, 2);
            h2c = PyArray_GETPTR1(charges, o); // use wat to get the correct charge from charge array
            //printf("processing hydrogen %i of water %i for target water atom %i\n", o, *target_O_at, wat_O);
            d_oh = dist_mic(*h2x, *h2y, *h2z, *w1x, *w1y, *w1z, *b_x, *b_y, *b_z); 
            //*(double *)PyArray_GETPTR2(voxel_data, v_id, 13) += ((*w1c)*(*h2c))/d_oh;
            E_wat += ((*w1c)*(*h2c))/d_oh;
            for (p = wat_O + 1; p <= wat_O + 2; p++){
                //printf("processing water atom: %i\n", m);
                h1x = PyArray_GETPTR2(coords, p-1, 0); // use wat to get the correct x, y, z coordinates from coord array
                h1y = PyArray_GETPTR2(coords, p-1, 1); 
                h1z = PyArray_GETPTR2(coords, p-1, 2);
                h1c = PyArray_GETPTR1(charges, p); // use wat to get the correct charge from charge array
                //printf("processing hydrogen %i of water %i for hydrogen %i of target water atom %i\n", p, wat_O, o, *target_O_at);
                d_hh = dist_mic(*h1x, *h1y, *h1z, *h2x, *h2y, *h2z, *b_x, *b_y, *b_z); // distance (based on minimum image convention)
                //*(double *)PyArray_GETPTR2(voxel_data, v_id, 13) += ((*h1c)*(*h2c))/d_hh; // Coulombic interaction calculation
                E_wat += ((*h1c)*(*h2c))/d_hh;
                } // end looping over water hydrogen atoms
            } // end looping over target water hydrogen atoms
        //printf("target water atom ID x y z energy: %i %5.3f %5.3f %5.3f %5.3f\n", *target_at, *w2x, *w2y, *w2z, E_wat/2.0);
        *(double *)PyArray_GETPTR1(nbr_energies, i) += E_wat;
        
    }
    return Py_BuildValue("i", 1);
}






/* Method Table
 * Registering all the functions that will be called from Python
 */

static PyMethodDef _hsacalcs_methods[] = {
    {
        "elecE",                           // name of the fucntion called from Python
        (PyCFunction)_gistcalcs_elecE,     // corresponding C++ function
        METH_VARARGS,
        "compute electrostatic energies"   // doc string
    },
    {
        "vdwE",
        (PyCFunction)_gistcalcs_vdwE,
        METH_VARARGS,
        "compute LJ energies"
    },
    {
        "nbr_E_ww",
        (PyCFunction)_gistcalcs_nbr_E_ww,
        METH_VARARGS,
        "compute LJ energies"
    },
    {NULL, NULL}
};

/* Initialization function for this module
 */

PyMODINIT_FUNC init_hsacalcs() // init function has the same name as module, except with init prefix
{
    // we prodice name of the module, method table and a doc string
    Py_InitModule3("_hsacalcs", _hsacalcs_methods, "compute energies.\n");
    import_array(); // required for Numpy initialization
}