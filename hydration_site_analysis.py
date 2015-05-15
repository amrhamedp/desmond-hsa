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
#  REQUIREMENTS:  Desmond, Schrodinger Python API
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
# import shrodinger modules
from schrodinger import structure
from schrodinger.application.desmond.cms import Vdw
from schrodinger.trajectory.desmondsimulation import create_simulation
from schrodinger.trajectory.atomselection import select_component
from schrodinger.trajectory.atomselection import FrameAslSelection as FAS
import schrodinger.application.desmond.ffiostructure as ffiostructure

# import other python modules
import numpy as np
from scipy import stats

#from scipy.spatial import KDTree, cKDTree
import os, sys, time

import _hsacalcs as quick

#################################################################################################################
# Main GIST class                                                                                               #
#################################################################################################################



class HSAcalcs:
#*********************************************************************************************#
    # Initializer function
    def __init__(self, input_cmsname, input_trjname, clustercenter_file):
        """
        Data members
        """
        self.cmsname = input_cmsname
        self.dsim = create_simulation(input_cmsname, input_trjname)
        self._indexGenerator()
        self.hsa_data = self._initializeHSADict(clustercenter_file)
        self.pbc = self._initializePBC()
        self.charges, self.vdw = self.getParams()


#*********************************************************************************************#
    # index generator function
    def _indexGenerator(self):

        frame = self.dsim.getFrame(0)
        # atom and global indices of all atoms in the system
        self.all_atom_ids = np.arange(len(self.dsim.cst.atom))+1
        self.all_atom_gids = np.arange(len(self.dsim.cst.atom)+self.dsim.cst._pseudo_total)
        # obtain oxygen atom and global indices for all water molecules
        oxygen_sel = FAS('atom.ele O')
        all_oxygen_atoms = oxygen_sel.getAtomIndices(frame)
        water_sel = select_component(self.dsim.cst, ['solvent'])
        solvent_atoms = water_sel.getAtomIndices(frame)
        solvent_oxygen_atoms = list(set(solvent_atoms).intersection(set(all_oxygen_atoms)))
        solvent_oxygen_atoms.sort()
        self.wat_oxygen_atom_ids = np.array(solvent_oxygen_atoms, dtype=np.int)
        self.wat_oxygen_atom_gids = self.wat_oxygen_atom_ids - 1
        # obtain atom indices for all water atoms
        self.wat_atom_ids = self.getWaterIndices(self.wat_oxygen_atom_ids)
        # obtain atom and global indices for all water atoms
        #wat_id_list = self.getWaterIndices(self.wat_oxygen_atom_ids)
        #self.wat_atom_ids = wat_id_list[0]
        #self.wat_atom_gids = wat_id_list[1]        
        # obtain all non-water atom and global indices
        self.non_water_atom_ids = np.setxor1d(self.all_atom_ids, self.wat_atom_ids).astype(int)
        #self.non_water_gids = np.setxor1d(self.all_atom_gids,self.wat_atom_gids)
        # These lists define the search space for solute water Hbond calculation
 
#*********************************************************************************************#
    # retrieve water atom indices for selected oxygen indices 
    def getWaterIndices(self, oxygen_atids):
        # here we will get data that is required to create previous index mapper object
        # first step is to obtain solvent forcefield structure
        solvent_ffst = None
        # obtain solvent fsst by iterating over all 'types' of forcefield (i.e., solute, solvent, ion)
        for ffst in self.dsim.cst.ffsts:
            if ffst.parent_structure.property['s_ffio_ct_type'] == 'solvent':
                if solvent_ffst is not None:
                    raise Exception("does not support multiple solvent ct.")
                solvent_ffst = ffst

        # set oxygen index to none
        oxygen_index = None
        # set types of pseudo particles to 0
        npseudo_sites = 0
        # set number of solvent atom types to 0
        natom_sites = 0
       # for each forcefield site (which is any 'site' on the structure to which params are assigned)
        for i, site in enumerate(solvent_ffst.ffsite):
            # check if this site belongs to Oxygen atoms
            if site.vdwtype.upper().startswith('O'):
                # if oxygen index is already defined, raise exception otherwise set oxygen index to this site
                if oxygen_index is not None:
                    raise Exception("water molecule has more than two oxygen atoms")
                oxygen_index = i
            # check if this site belongs to pseudoparticle, if yes raise corresponding number
            if site.type.lower() == 'pseudo':
                npseudo_sites += 1
            # check if this site belongs to an atom, if yes raise corresponding number
            elif site.type.lower() == 'atom':
                natom_sites += 1
        # at the end of this loop we have checked all possible forcefield sites to get the correst index for oxygen
        # in addition we get total number of atoms and pseudopartciles on a solvent site (water in this case) 
        if oxygen_index is None:
            raise Exception("can not locate oxygen atom.")
        if natom_sites == 0:
            raise Exception("number of atoms is zero.")
        # here we totall number of atoms in solvent 
        nmols = len(solvent_ffst.parent_structure.atom)/natom_sites
        #print oxygen_index
        # this is atid for the first oxygen atom in water oxygen atom array
        wat_begin_atid = oxygen_atids[0]
        # gid in this case is atid - 1
        wat_begin_gid = wat_begin_atid - 1
        oxygen_gids = oxygen_atids - 1
        pseudo_begin_gid = wat_begin_gid + natom_sites*nmols
        id_list = []
        #return atids of atoms of selected water molecules.
        water_atids = []
        for oxygen_atid in oxygen_atids:
            for i in range(natom_sites):
                atid = oxygen_atid + i - oxygen_index
                water_atids.append(atid)
        #id_list.append(np.array(water_atids))
        #return gids of particles (including pseudo sites) of selected water molecules.
        # For now we will ignore GIDs but these are important when water model has pseudoatoms
        """
        water_gids = []
        for oxygen_gid in oxygen_gids:
            for i in range(natom_sites):
                gid = oxygen_gid + i - oxygen_index
                water_gids.append(gid)
            # pseudo atoms are placed right after real atoms
            offset = (oxygen_gid - wat_begin_gid) / natom_sites
            for i in range(npseudo_sites):
                gid = pseudo_begin_gid + offset*npseudo_sites + i
                water_gids.append(gid)
        water_gids.sort()
        id_list.append(np.array(water_gids, dtype=np.int))
        """
        self.oxygen_index = oxygen_index
        self.n_atom_sites = natom_sites
        self.n_pseudo_sites = npseudo_sites
        self.wat_begin_gid = wat_begin_gid
        self.pseudo_begin_gid = pseudo_begin_gid
        return np.array(water_atids)

#*********************************************************************************************#
    def _initializeHSADict(self, clust_center_file):
        clusters = structure.StructureReader(clust_center_file).next()
        hs_dict = {}
        cluster_centers = clusters.getXYZ()
        c_count = 0
        data_fields = 12
        self.data_titles = ["wat", "occ", "gO", "Esw", "EswLJ", "EswElec", "Eww", "EwwLJ", "EwwElec", "Etot", 
                        "Enbr", "nbrs", "pair_ene"]

        for h in cluster_centers: 
            hs_dict[c_count] = [tuple(h)] # create a dictionary key-value pair with voxel index as key and it's coords as
            hs_dict[c_count].append(np.zeros(data_fields, dtype="float64"))
            hs_dict[c_count].append([]) # to store E_nbr distribution
            for i in range(data_fields+1): hs_dict[c_count][2].append([]) 
            hs_dict[c_count].append(np.zeros(data_fields+1, dtype="float64")) # to store error info on each timeseries
            c_count += 1
        return hs_dict


#*********************************************************************************************#
    def _initializePBC(self):
        # for minimum image convention
        box_vectors = self.dsim.getFrame(0).box
        if box_vectors[0] == 0.0 or box_vectors[4] == 0.0 or box_vectors[8] == 0.0:
            print "Warning: Periodic Boundary Conditions unspecified!"
        else:
            box = np.asarray([box_vectors[0], box_vectors[4], box_vectors[8]])
        return box
#*********************************************************************************************#

#*********************************************************************************************#
    def getNeighborAtoms(self, xyz, dist, point):
        """
        An efficint routine for neighbor search
        """
        # create an array of indices around a cubic grid
        dist_squared = dist * dist
        neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    neighbors.append((i,j,k))
        neighbor_array = np.array(neighbors, np.int)
        min_ = np.min(xyz, axis=0)
        cell_size = np.array([dist, dist, dist], np.float)
        cell = np.array((xyz - min_) / cell_size)#, dtype=np.int)
        # create a dictionary with keys corresponding to integer representation of transformed XYZ's
        cells = {}
        for ix, assignment in enumerate(cell):
            # convert transformed xyz coord into integer index (so coords like 1.1 or 1.9 will go to 1)
            indices =  assignment.astype(int)
            # create interger indices
            t = tuple(indices)
            # NOTE: a single index can have multiple coords associated with it
            # if this integer index is already present
            if t in cells:
                # obtain its value (which is a list, see below)
                xyz_list, trans_coords, ix_list = cells[t]
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
                cells[t] = ([xyz[ix]], [assignment], [ix])

        cell0 = np.array((point - min_) / cell_size, dtype=np.int)
        tuple0 = tuple(cell0)
        near = []
        for index_array in tuple0 + neighbor_array:
            t = tuple(index_array)
            if t in cells:
                xyz_list, trans_xyz_list, ix_list = cells[t]
                for (xyz, ix) in zip(xyz_list, ix_list):
                    diff = xyz - point
                    if np.dot(diff, diff) <= dist_squared and float(np.dot(diff, diff)) > 0.0:
                        #near.append(ix)
                        #print ix, np.dot(diff, diff)
                        near.append(ix)
        return near

#*********************************************************************************************#

#*********************************************************************************************#
    def getParams(self):
        # obtain LJ and Elec params
        #*********************************************************************************#
        vdw = [None,] # combined list of all vdw params from all ct's
        chg = [0.0,] # combined list of all charges from all ct's
        ct_list = [e for e in ffiostructure.CMSReader(self.cmsname)]
        struct_ct = ct_list[1:] # this means this works only on cms files with separate CT blocks
        # get tota number of solute atoms
        
        for ct in struct_ct:
            ct_chg = []
            ct_vdw = []
            vdw_type = {} # dict of vdw types, Vdw object in this list are uninitialized
            for e in ct.ffio.vdwtype :
                vdw_type[e.name] = Vdw( (e.name,), e.funct, (e.c1, e.c2,) )
                #print (e.name,), e.funct, (e.c1, e.c2,)
            for e in ct.ffio.site: # for each site (i.e., an atom in most cases)
                ct_vdw.append( vdw_type[e.vdwtype] ) # add to vdw list for this ct
                ct_chg.append(e.charge)
                #print e.index, e.charge
            ct_vdw *= int(ct.atom_total / len( ct.ffio.site ))
            ct_chg *= int(ct.atom_total / len( ct.ffio.site ))
            #print int(ct.atom_total / len( ct.ffio.site ))
            vdw.extend( ct_vdw )
            chg.extend( ct_chg)
            
        chg = np.asarray(chg)*18.2223
        vdw_params = [[0,0],]
        #print len(chg)
        #print len(all_at_ids)
        for v in vdw[1:]:
            vdw_params.extend([v.c])
        vdw_params = np.asarray(vdw_params)
        return (chg, vdw_params)
#*********************************************************************************************#
                   
    def hsEnergyCalculation(self, n_frame, start_frame):
        # first step is to iterate over each frame
        for i in xrange(start_frame, start_frame + n_frame):
            print "Processing frame: ", i+1, "..."
            # get frame structure, position array
            frame = self.dsim.getFrame(i)
            #measure_manager = PBCMeasureMananger(frame)
            frame_st = self.dsim.getFrameStructure(i)
            pos = frame.position
            oxygen_pos = pos[self.wat_oxygen_atom_ids-1] # obtain coords of O-atoms
            for cluster in self.hsa_data:
                #print "processin cluster: ", cluster
                nbr_indices = self.getNeighborAtoms(oxygen_pos, 1.0, self.hsa_data[cluster][0])
                cluster_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                # begin iterating over water oxygens found in this cluster in current frame
                for wat_O in cluster_wat_oxygens:
                    self.hsa_data[cluster][1][0] += 1 # raise water population by 1
                    cluster_water_all_atoms = self.getWaterIndices(np.asarray([wat_O]))
                    rest_wat_at_ids = np.setxor1d(cluster_water_all_atoms, self.wat_atom_ids) # at indices for rest of solvent water atoms
                    rest_wat_oxygen_at_ids = np.setxor1d(wat_O, self.wat_oxygen_atom_ids) # at indices for rest of solvent water O-atoms
                    Etot = 0
                    # solute-solvent energy calcs
                    if self.non_water_atom_ids.size != 0:
                        Elec_sw = quick.elecE(cluster_water_all_atoms, self.non_water_atom_ids, pos, self.charges, self.pbc)*0.5
                        LJ_sw = quick.vdwE(np.asarray([wat_O]), self.non_water_atom_ids, pos, self.vdw, self.pbc)*0.5
                        #print Elec_sw*0.5, LJ_sw*0.5
                        self.hsa_data[cluster][1][3] += Elec_sw + LJ_sw
                        self.hsa_data[cluster][2][3].append(Elec_sw + LJ_sw)
                        self.hsa_data[cluster][1][4] += LJ_sw
                        self.hsa_data[cluster][2][4].append(LJ_sw)
                        self.hsa_data[cluster][1][5] += Elec_sw
                        self.hsa_data[cluster][2][5].append(Elec_sw)
                        Etot += Elec_sw + LJ_sw 
                    #********************************************************************************#
                    # solvent-solvent energy calcs
                    # obtain this water's electrostatic interaction with all other atoms
                    # and also collect all its nbr's electrostatic interactions into index 7 of and total number of neighbours into index 8
                    Elec_ww = quick.elecE(cluster_water_all_atoms, rest_wat_at_ids, pos, self.charges, self.pbc)*0.5
                    LJ_ww = quick.vdwE(np.asarray([wat_O]), rest_wat_oxygen_at_ids, pos, self.vdw, self.pbc)*0.5
                    Etot += Elec_ww + LJ_ww
                    self.hsa_data[cluster][1][6] += Elec_ww + LJ_ww
                    self.hsa_data[cluster][2][6].append(Elec_ww + LJ_ww)
                    self.hsa_data[cluster][1][7] += LJ_ww
                    self.hsa_data[cluster][2][7].append(LJ_ww)
                    self.hsa_data[cluster][1][8] += Elec_ww
                    self.hsa_data[cluster][2][8].append(Elec_ww)
                    self.hsa_data[cluster][1][9] += Elec_ww + LJ_ww
                    self.hsa_data[cluster][2][9].append(Etot)

                    nbr_indices = self.getNeighborAtoms(oxygen_pos, 3.5, pos[wat_O-1])
                    firstshell_wat_oxygens = [self.wat_oxygen_atom_ids[nbr_index] for nbr_index in nbr_indices]
                    self.hsa_data[cluster][1][11] += len(firstshell_wat_oxygens) # add  to cumulative sum
                    self.hsa_data[cluster][2][11].append(len(firstshell_wat_oxygens)) # add nbrs to nbr timeseries
                    
                    if len(firstshell_wat_oxygens) != 0:
                        nbr_energy_array = np.zeros(len(firstshell_wat_oxygens), dtype="float64")
                        quick.nbr_E_ww(wat_O, np.asarray(firstshell_wat_oxygens), pos, self.vdw, self.charges, self.pbc, nbr_energy_array)
                        self.hsa_data[cluster][1][10] += (np.sum(nbr_energy_array)/len(firstshell_wat_oxygens))*0.5
                        self.hsa_data[cluster][2][10].append((np.sum(nbr_energy_array)/len(firstshell_wat_oxygens))*0.5)
                        for ene in nbr_energy_array:
                            self.hsa_data[cluster][2][12].append(ene)

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





if (__name__ == '__main__') :

    _version = "$Revision: 0.0 $"
    
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input_cms", dest="cmsname", type="string", help="Input CMS file")
    parser.add_option("-t", "--input_trajectory", dest="trjname", type="string", help="Input trajectory directory")
    parser.add_option("-c", "--cluster_centers", dest="clusters", type="string", help="Cluster center file")
    parser.add_option("-f", "--frames", dest="frames", type="int", help="Number of frames")
    parser.add_option("-s", "--starting frame", dest="start_frame", type="int", help="Starting frame")
    parser.add_option("-o", "--output_name", dest="prefix", type="string", help="Output log file")
    (options, args) = parser.parse_args()
    print "Setting things up..."
    h = HSAcalcs(options.cmsname, options.trjname, options.clusters)
    print "Running calculations ..."
    t = time.time()
    h.hsEnergyCalculation(options.frames, options.start_frame)
    h.normalizeClusterQuantities(options.frames)
    print "Done! took %8.3f seconds." % (time.time() - t)
    print "Writing timeseries data ..."
    h.writeTimeSeries(options.prefix)
    print "Writing summary..."
    h.writeHBsummary(options.prefix)

    