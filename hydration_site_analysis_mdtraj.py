from __future__ import division
__doc__='''

#===============================================================================
#
#          FILE:  Main classes and functions implementing hydrogen bond calculations on a 3-D grid
                  In general this grid coincides with a GIST grid from a previously run GIST calculation.
#         USAGE:  a tester script will be provided as an example 
# 
#   DESCRIPTION:  
# 
#       OPTIONS:  ---
#  REQUIREMENTS:  MDTraj
#          BUGS:  ---
#         NOTES:  ---
#        AUTHOR:  Kamran Haider
#   Contibutors:  
#     COPYRIGHT:  
#       COMPANY:  
#       VERSION:  1.0
#       CREATED:  
#      REVISION:  ---
#===============================================================================

'''
_version = "$Revision: 1.0 $"

# import other python modules
import numpy as np
import mdtraj as md
from scipy import stats

import os, sys, time
import re
import copy
import _hsacalcs_v2 as quick



#################################################################################################################
# Main GIST class                                                                                               #
#################################################################################################################



class HSAcalcs:
#*********************************************************************************************#
    # Initializer function
    def __init__(self, input_prmtop, input_trj, clustercenter_file):
        """
        Initializes  an object of HSAcalcs class
        """
        print "Reading in topology and trajectory ..."
        self.trj = md.load(input_trj, top=input_prmtop)
        self.top = self.trj.topology
        self._indexGenerator()
        self.pbc = self.trj.unitcell_lengths[0]*10.0
        self._getParams(input_prmtop)
        self.hsa_data = self._initializeHSADict(clustercenter_file)

#*********************************************************************************************#
    def _indexGenerator(self):
        # obtain total number of atoms
        '''
        Returns atom indices as numpy arrays for different categories of atoms
        Returned as memeber attributes to HSACalcs class
        '''
        self.all_atom_ids = self.top.select("all")
        self.wat_atom_ids = self.top.select("water")
        self.wat_oxygen_atom_ids = self.top.select("water and name O")
        self.non_water_atom_ids = self.top.select("not water")

#*********************************************************************************************#
    def _initializeHSADict(self, clust_center_file):
        '''
        Returns a dictionary with hydration site indices as keys and their properties as values.
        '''
        clusters_pdb_file = md.load_pdb(clust_center_file)
        c_count = 0
        data_fields = 12
        self.data_titles = ["wat", "occ", "gO", "Esw", "EswLJ", "EswElec", "Eww", "EwwLJ", "EwwElec", "Etot", 
                        "Enbr", "nbrs", "pair_ene"]
        hs_dict = {}
        for chain in clusters_pdb_file.xyz*10:
            for h in chain: 
                hs_dict[c_count] = [tuple(h)] # create a dictionary key-value pair with voxel index as key and it's coords as
                hs_dict[c_count].append(np.zeros(data_fields, dtype="float64"))
                hs_dict[c_count].append([]) # to store E_nbr distribution
                for i in range(data_fields+1): hs_dict[c_count][2].append([]) 
                hs_dict[c_count].append(np.zeros(data_fields+1, dtype="float64")) # to store error info on each timeseries
                c_count += 1
        return hs_dict
#*********************************************************************************************#
    #def _getPBC(self):
        '''
        Stores box information for the trajectory
        '''
        #IMPORTANT: Only trajectories with NVT should be processed
        #self.trj.unitcell_lengths[0]*10.0
        
#*********************************************************************************************#
    def _getParams(self, input_prmtop):
        '''
        Returns van der Waal parameters and charges from the topology file
        '''
        prmtop_file = open(input_prmtop)
        # read all lines in prmtop into a list
        lines = prmtop_file.readlines()
        # create a dictionary
        blks = {}
        # creayomh regexp to identify sections of the file
        re_flag = re.compile(r'\%FLAG\s+(\S+)', re.IGNORECASE)
        re_format = re.compile(r'\%FORMAT\s*\((\d+)\w(\d+)\S*\)', re.IGNORECASE)
        re_comment = re.compile(r'\%\S+', re.IGNORECASE)
        flag_name = ''
        flag_length = 0
        for l in lines:
            flag_match = re_flag.search(l)
            format_match = re_format.search(l)
            comment_match = re_comment.search(l)
            if flag_match:
                flag_name = flag_match.group(1)
                flag_size = 0
                flag_length = 0
                blks[flag_name] = []
            elif format_match:
                flag_size = int(format_match.group(1))
                flag_length = int(format_match.group(2))
            elif comment_match:
                continue
            else:
                if not (flag_name and flag_size and flag_length):
                    print 'cannot recognize flag.'
                for i in range(flag_size):
                    start = i*flag_length
                    end = start + flag_length
                    if end > len(l):
                        end = len(l)
                    element = l[start:end].strip()
                    if element:
                        blks[flag_name].append(element)
                    else:
                        break

        type_index = blks['ATOM_TYPE_INDEX']
        ntype = len(set(type_index))
        vdw_type_symbol = range(1, ntype+1)
        acoeff = blks['LENNARD_JONES_ACOEF']
        bcoeff = blks['LENNARD_JONES_BCOEF']
        nonbonded = blks['NONBONDED_PARM_INDEX']
        vdwtype = {}
        k = 0
        for i in range(ntype):
            for j in range(ntype):
                index = int(nonbonded[k]) - 1
                A = float(acoeff[index])
                B = float(bcoeff[index])
                sigma = 0.0
                epsilon = 0.0
                if A != 0 or B != 0:
                    epsilon = B*B/4.0/A
                    sigma = (A/B)**(1.0/6.0)
                #print i, j, k, type_list[i], type_list[j], acoeff[index], bcoeff[index]
                #print k, sigma, epsilon
                if i == j:
                    #print i+1, sigma, epsilon
                    vdwtype[i+1] = (sigma, epsilon)
                k += 1
        vdw = []
        for at in self.all_atom_ids:
            vdw.append(vdwtype[int(type_index[at])])
        self.vdw = np.asarray(vdw)
        self.chg  = np.asarray(blks['CHARGE'], dtype=float)

#*********************************************************************************************#
    def hsEnergyCalculation(self, n_frame, start_frame):
        '''
        Returns energetic quantities for each hydration site
        '''
        for i in xrange(start_frame, start_frame + n_frame):
            print "Processing frame: ", i+1, "..." 
            pos = self.trj[i].xyz[0,:,:]*10.0
            oxygen_pos = pos[self.wat_oxygen_atom_ids] # obtain coords of O-atoms
            d_clust = _DistanceCell(oxygen_pos, 1.0)
            d_nbrs = _DistanceCell(oxygen_pos, 3.5)
            for cluster in self.hsa_data:
                #print "processin cluster: ", cluster
                nbr_indices = d_clust.query_nbrs(self.hsa_data[cluster][0])
                cluster_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                # begin iterating over water oxygens found in this cluster in current frame
                for wat_O in cluster_wat_oxygens:
                    self.hsa_data[cluster][1][0] += 1 # raise water population by 1
                    cluster_water_all_atoms = np.asarray([wat_O, wat_O+1, wat_O+2])
                    rest_wat_at_ids = np.setxor1d(cluster_water_all_atoms, self.wat_atom_ids) # at indices for rest of solvent water atoms
                    rest_wat_oxygen_at_ids = np.setxor1d(wat_O, self.wat_oxygen_atom_ids) # at indices for rest of solvent water O-atoms
                    Etot = 0
                    if self.non_water_atom_ids.size != 0:
                        #print "Getting energies: "
                        Elec_sw = quick.elecE(cluster_water_all_atoms, self.non_water_atom_ids, pos, self.chg, self.pbc)*0.5
                        LJ_sw = quick.vdwE(np.asarray([wat_O]), self.non_water_atom_ids, pos, self.vdw, self.pbc)*0.5
                        #print Elec_sw*0.5, LJ_sw*0.5
                        self.hsa_data[cluster][1][3] += Elec_sw + LJ_sw
                        self.hsa_data[cluster][2][3].append(Elec_sw + LJ_sw)
                        self.hsa_data[cluster][1][4] += LJ_sw
                        self.hsa_data[cluster][2][4].append(LJ_sw)
                        self.hsa_data[cluster][1][5] += Elec_sw
                        self.hsa_data[cluster][2][5].append(Elec_sw)
                        Etot += Elec_sw + LJ_sw 
                    Elec_ww = quick.elecE(cluster_water_all_atoms, rest_wat_at_ids, pos, self.chg, self.pbc)*0.5
                    LJ_ww = quick.vdwE(np.asarray([wat_O]), rest_wat_oxygen_at_ids, pos, self.vdw, self.pbc)*0.5
                    Etot += Elec_ww + LJ_ww
                    self.hsa_data[cluster][1][6] += Elec_ww + LJ_ww
                    self.hsa_data[cluster][2][6].append(Elec_ww + LJ_ww)
                    self.hsa_data[cluster][1][7] += LJ_ww
                    self.hsa_data[cluster][2][7].append(LJ_ww)
                    self.hsa_data[cluster][1][8] += Elec_ww
                    self.hsa_data[cluster][2][8].append(Elec_ww)
                    self.hsa_data[cluster][1][9] += Etot
                    self.hsa_data[cluster][2][9].append(Etot)

                    nbr_indices = d_nbrs.query_nbrs(pos[wat_O])
                    firstshell_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                    self.hsa_data[cluster][1][11] += len(firstshell_wat_oxygens) # add  to cumulative sum
                    self.hsa_data[cluster][2][11].append(len(firstshell_wat_oxygens)) # add nbrs to nbr timeseries
                    #print Etot
                    if len(firstshell_wat_oxygens) != 0:
                        nbr_energy_array = np.zeros(len(firstshell_wat_oxygens), dtype="float64")
                        quick.nbr_E_ww(wat_O, np.asarray(firstshell_wat_oxygens), pos, self.vdw, self.chg, self.pbc, nbr_energy_array)
                        self.hsa_data[cluster][1][10] += (np.sum(nbr_energy_array)/len(firstshell_wat_oxygens))*0.5
                        self.hsa_data[cluster][2][10].append((np.sum(nbr_energy_array)/len(firstshell_wat_oxygens))*0.5)
                        for ene in nbr_energy_array:
                            self.hsa_data[cluster][2][12].append(ene/2.0)
#*********************************************************************************************#

    def normalizeClusterQuantities(self, n_frame):
        rho_bulk = 0.0329 #molecules/A^3 # 0.0329
        sphere_vol = (4/3)*np.pi*1.0
        bulkwaterpersite = rho_bulk*n_frame*sphere_vol
        for cluster in self.hsa_data:
            if self.hsa_data[cluster][1][0] != 0:
                #print cluster
                # occupancy of the cluster
                self.hsa_data[cluster][1][1] = self.hsa_data[cluster][1][0]/n_frame
                # gO of the cluster
                self.hsa_data[cluster][1][2] = self.hsa_data[cluster][1][0]/(bulkwaterpersite)
                if self.non_water_atom_ids.size != 0:
                    # normalized Esw
                    self.hsa_data[cluster][1][3] /= self.hsa_data[cluster][1][0]
                    # normalized EswLJ
                    self.hsa_data[cluster][1][4] /= self.hsa_data[cluster][1][0]
                    # normalized EswElec
                    self.hsa_data[cluster][1][5] /= self.hsa_data[cluster][1][0]

                # normalized Eww
                self.hsa_data[cluster][1][6] /= self.hsa_data[cluster][1][0]
                # normalized EwwLJ
                self.hsa_data[cluster][1][7] /= self.hsa_data[cluster][1][0]
                # normalized EwwElec
                self.hsa_data[cluster][1][8] /= self.hsa_data[cluster][1][0]
                # normalized Etot
                self.hsa_data[cluster][1][9] /= self.hsa_data[cluster][1][0]
                # Normalized Nbr and Ewwnbr
                if self.hsa_data[cluster][1][11] != 0:
                    self.hsa_data[cluster][1][10] /= self.hsa_data[cluster][1][0]
                    self.hsa_data[cluster][1][11] /= self.hsa_data[cluster][1][0]

#*********************************************************************************************#

    def writeHBsummary(self, prefix):
        f = open(prefix+"_hsa_ene_summary.txt", "w")
        header_2 = "index x y z wat occ gO Esw EswLJ EswElec Eww EwwLJ EwwElec Etot Enbr nbrs\n"
        f.write(header_2)
        for cluster in self.hsa_data:
            d = self.hsa_data[cluster]
            l = "%d %.2f %.2f %.2f %d %f %f %f %f %f %f %f %f %f %f %f\n" % \
                ( cluster, d[0][0], d[0][1], d[0][2], \
                d[1][0], d[1][1], d[1][2], \
                d[1][3], d[1][4], d[1][5], d[1][6], d[1][7], d[1][8], d[1][9],\
                d[1][10], d[1][11])
            f.write(l)
        f.close()
        # writing standard deviations
        e = open(prefix+"_hsa_ene_stats.txt", "w")
        header_3 = "index Esw EswLJ EswElec Eww EwwLJ EwwElec Etot Enbr nbrs\n"
        e.write(header_3)
        for cluster in self.hsa_data:
            d = self.hsa_data[cluster]
            l = "%d %f %f %f %f %f %f %f %f %f\n" % \
                ( cluster, d[3][3], d[3][4], d[3][5], d[3][6], d[3][7], d[3][8], d[3][9],\
                d[3][10], d[3][11])
            #print l
            e.write(l)
        e.close()

#*********************************************************************************************#

    def writeTimeSeries(self, prefix):
        cwd = os.getcwd()
        # create directory to store detailed data for individual columns in HSA
        directory = cwd + "/" + prefix+"_cluster_ene_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
        # for each cluster, go through time series data
        for cluster in self.hsa_data:
            cluster_index = "%03d_" % cluster
            #print cluster_index#, self.hsa_data[cluster][2]
            for index, data_field in enumerate(self.hsa_data[cluster][2]):
                # only write timeseries data that was stored during calculation
                if len(data_field) != 0:
                    # create appropriate file name
                    data_file = cluster_index + prefix + "_" + self.data_titles[index] 
                    #print index, self.data_titles[index]
                    f =  open(data_file, "w")
                    # write each value from the timeseries into the file
                    for value in data_field:
                    #    print value
                        f.write(str(value)+"\n")
                    f.close()
                    if self.data_titles[index] == "Enbr":
                        self.hsa_data[cluster][3][index] += stats.sem(np.asarray(data_field), axis=None, ddof=0)
                        #print self.hsa_data[cluster][3][index]
                    else:
                        self.hsa_data[cluster][3][index] += np.std(np.asarray(data_field))
                        #print self.hsa_data[cluster][3][index]

        os.chdir("../")
#*********************************************************************************************#

#################################################################################################################
# Class and methods for 'efficient' neighbor search                                                             #
#################################################################################################################

class _DistanceCell:
    def __init__(self, xyz, dist):
        """
        Class for fast queries of coordinates that are within distance <dist>
        of specified coordinate. This class must first be initialized from an
        array of all available coordinates, and a distance threshold. The
        query() method can then be used to get a list of points that are within
        the threshold distance from the specified point.
        """
        # create an array of indices around a cubic grid
        self.neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    self.neighbors.append((i,j,k))
        self.neighbor_array = np.array(self.neighbors, np.int)

        self.min_ = np.min(xyz, axis=0)
        self.cell_size = np.array([dist, dist, dist], np.float)
        cell = np.array((xyz - self.min_) / self.cell_size)#, dtype=np.int)
        # create a dictionary with keys corresponding to integer representation of transformed XYZ's
        self.cells = {}
        for ix, assignment in enumerate(cell):
            # convert transformed xyz coord into integer index (so coords like 1.1 or 1.9 will go to 1)
            indices =  assignment.astype(int)
            # create interger indices
            t = tuple(indices)
            # NOTE: a single index can have multiple coords associated with it
            # if this integer index is already present
            if t in self.cells:
                # obtain its value (which is a list, see below)
                xyz_list, trans_coords, ix_list = self.cells[t]
                # append new xyz to xyz list associated with this entry
                xyz_list.append(xyz[ix])
                # append new transformed xyz to transformed xyz list associated with this entry
                trans_coords.append(assignment)
                # append new array index 
                ix_list.append(ix)
            # if this integer index is encountered for the first time
            else:
                # create a dictionary key value pair,
                # key: integer index
                # value: [[list of x,y,z], [list of transformed x,y,z], [list of array indices]]
                self.cells[t] = ([xyz[ix]], [assignment], [ix])

        self.dist_squared = dist * dist



    def query_nbrs(self, point):
        """
        Given a coordinate point, return all point indexes (0-indexed) that
        are within the threshold distance from it.
        """
        cell0 = np.array((point - self.min_) / self.cell_size, 
                                     dtype=np.int)
        tuple0 = tuple(cell0)
        near = []
        for index_array in tuple0 + self.neighbor_array:
            t = tuple(index_array)
            if t in self.cells:
                xyz_list, trans_xyz_list, ix_list = self.cells[t]
                for (xyz, ix) in zip(xyz_list, ix_list):
                    diff = xyz - point
                    if np.dot(diff, diff) <= self.dist_squared and float(np.dot(diff, diff)) > 0.0:
                        #near.append(ix)
                        #print ix, np.dot(diff, diff)
                        near.append(ix)
        return near
#*********************************************************************************************#




if (__name__ == '__main__') :

    _version = "$Revision: 0.0 $"
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input_prmtop", dest="prmtop", type="string", help="Input amber prmtop file")
    parser.add_option("-t", "--input_trajectory", dest="trjname", type="string", help="Input trajectory file")
    parser.add_option("-c", "--cluster_centers", dest="clusters", type="string", help="Cluster center file")
    parser.add_option("-f", "--frames", dest="frames", type="int", help="Number of frames")
    parser.add_option("-s", "--starting frame", dest="start_frame", type="int", help="Starting frame")
    parser.add_option("-o", "--output_name", dest="prefix", type="string", help="Output log file")
    (options, args) = parser.parse_args()
    print "Setting things up..."
    h = HSAcalcs(options.prmtop, options.trjname, options.clusters)
    print "Running calculations ..."
    t = time.time()
    h.hsEnergyCalculation(options.frames, options.start_frame)
    h.normalizeClusterQuantities(options.frames)
    print "Done! took %8.3f seconds." % (time.time() - t)
    print "Writing timeseries data ..."
    h.writeTimeSeries(options.prefix)
    print "Writing summary..."
    h.writeHBsummary(options.prefix)
