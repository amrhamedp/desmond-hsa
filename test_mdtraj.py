import re
import copy

import mdtraj as md
#t = md.load("../wfh/kh.58935/md10.nc", top="../wfh/kh.58935/arg_solvated.prmtop")
#find ways to access atoms via indices
# for now we will consider simple cases with no pseudo atoms

#for a in t.topology.atoms:
#    print a.charge

def parsePrmtop(ifname):
    f = open(ifname)
    # read all lines in prmtop into a list
    lines = f.readlines()
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
    return blks

prmtop_info = parsePrmtop("../wfh/kh.58935/meth.prmtop")
charges  = prmtop_info['CHARGE']

type_index = prmtop_info['ATOM_TYPE_INDEX']
ntype = len(set(type_index))
print ntype
vdw_type_symbol = range(ntype)

acoeff = prmtop_info['LENNARD_JONES_ACOEF']
bcoeff = prmtop_info['LENNARD_JONES_BCOEF']
nonbonded = prmtop_info['NONBONDED_PARM_INDEX']
#print nonbonded
vdwtype = []
vdwtype_combined = []
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
        vdwtype_combined.append((vdw_type_symbol[i], vdw_type_symbol[j], sigma, epsilon))
        if i == j:
            vdwtype.append((vdw_type_symbol[i], sigma, epsilon))
        print index, type_index[i], type_index[j], A, B
        print ntype*(int(type_index[i]) - 1) + int(type_index[j]) - 1

        #print 4*(int(type_index[i])-1)+ int(type_index[j]) + 4
        k += 1
print len(vdwtype)
#for vdw in vdwtype:
#    print vdw
#print charges