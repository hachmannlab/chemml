#!/usr/bin/env python

_LIB_NAME = "lib_jcode"
_LIB_VERSION = "v1.18.0"
_REVISION_DATE = "2012-11-09"
_AUTHOR = "Johannes Hachmann (jh@chemistry.harvard.edu)"
_DESCRIPTION = "This is the a library for general purpose functions."

# Version history timeline:
# v0.001 (2007-??-??) 
# v0.002 (2010-02-16): added tot_exec_time_str 
# v0.100 (2010-07-25,08-06): complete overhaul 
# v0.101 (2010-08-10): added intermed_exec_timing 
# v0.200 (2010-08-16): added std_datetime_str 
# v0.300 (2010-08-31): added chk_rmdir 
# v1.4.0 (2010-10-25): added wc_all; changed version format
# v1.5.0 (2010-10-31): added mksubdir_struct; modified chk_mkdir
# v1.6.0 (2010-11-01): added bin_file_format_change
# v1.7.0 (2011-04-04): added isFloat
# v1.7.1 (2011-04-04): modified isFloat, such that NaN, Nan, nan are excluded 
# v1.8.0 (2011-07-06): added target_dir_struct 
# v1.9.0 (2011-09-07): added line_count 
# v1.10.0 (2011-10-07): improve print_invoked_opts
# v1.10.1 (2011-10-13): improve formatting of print_invoked_opts
# v1.11.0 (2011-11-30): improve formatting of print_invoked_opts
# v1.12.0 (2012-04-10): add md5checksum
# v1.13.0 (2012-06-20): add filelinecount
# v1.14.0 (2012-06-20): add revdict_lookup
# v1.15.0 (2012-07-25): add intermed_process_timing
# v1.15.1 (2012-08-21): modify linecount for empty files, intermed_process_timing for zero count
# v1.15.2 (2012-10-03): modify idFloat by excluding infinity via '#IND'
# v1.16.0 (2012-10-22): add queryset_iterator
# v1.17.0 (2012-11-07,08): extend queryset_iterator with id_only and values
# v1.18.0 (2012-11-09): added list_chunks

###################################################################################################
# TASKS OF THIS LIBRARY:
# -provides general purpose functions
###################################################################################################

###################################################################################################
# TODO:
# -print lists/tuples in formatted fashion
# -review all the older routines
###################################################################################################

import sys
import os
import struct
import time
import datetime
import subprocess
import hashlib
import mmap
from numpy import fromstring

###################################################################################################

def banner(logfile, SCRIPT_NAME, SCRIPT_VERSION, REVISION_DATE, AUTHOR, DESCRIPTION,):
    """(banner):
        Banner for this little script.
    """
    str = []
    str.append("============================================================================== ")
    str.append(SCRIPT_NAME + " " + SCRIPT_VERSION + " (" + REVISION_DATE + ")")
    str.append(AUTHOR)
    str.append("============================================================================== ")
    str.append(time.ctime())
    str.append("")    
    str.append(DESCRIPTION)
    str.append("")

    print 
    for line in str:
        print line
        logfile.write(line + '\n')

##################################################################################################

def print_invoked_opts(logfile,opts,commline_list=[]):
    """(print_invoked_opts):
        Prints the invoked options to stdout and the logfile.
    """    
    if len(commline_list) != 0:
        tmp_str = "Invoked command line: "
        print tmp_str
        logfile.write(tmp_str + '\n')
        tmp_str = ' '.join(commline_list)
        print tmp_str
        print 
        logfile.write(tmp_str + '\n\n')
        
    tmp_str = "Invoked options: "
    print tmp_str
    logfile.write(tmp_str + '\n')
    for key, value in opts.__dict__.items():
        tmp_str = '   ' + key + ': ' + str(value)   
        print tmp_str    
        logfile.write(tmp_str + '\n')
    print
    logfile.write('\n')

##################################################################################################

def wc_dir(dir):
    """(wc_dir):
        Returns the number of dirs in a given dir via ls -1d | wc -l. 
        Not that this becomes a rather expensive function call when dir contains many subdirs.
    """    
#TODO: take care of error for empty dirs
    tmp_str = "ls -1d " + dir + "/*/ | wc -l"
    # this is a quite new python feature and may is only available in 2.6 or so 
    # n = subprocess.getoutput(tmp_str)
    # ... and for older python versions     
    return int(subprocess.Popen(tmp_str,shell=True,stdout=subprocess.PIPE).stdout.read())

##################################################################################################

def wc_all(dir):
    """(wc_all):
        Returns the number of files and dirs in a given dir via ls -1 | wc -l. 
        Not that this becomes a rather expensive function call when dir contains many entries.
    """    
#TODO: take care of error for empty dirs
    tmp_str = "ls -1 " + dir + " | wc -l"
    # this is a quite new python feature and may is only available in 2.6 or so 
    # n = subprocess.getoutput(tmp_str)
    # ... and for older python versions     
    return int(subprocess.Popen(tmp_str,shell=True,stdout=subprocess.PIPE).stdout.read())

##################################################################################################

def line_count(file_namestr):
    """(line_count):
        Returns the number of lines in a file.
    """    
    if os.path.getsize(file_namestr) == 0:
        return 0
    with open(file_namestr) as file:
        for i, l in enumerate(file):
            pass
    return i + 1

##################################################################################################

def mksubdir_struct(dir,max_n_entries=10000,run_always=False):
    """(mksubdir_struct):
        This function takes the content of a dir and makes numbered substructure dirs with each n_entries of the original dir.
        The motivation was to have a function with limits the number of entries in a directory to a certain threshold
        (e.g., 10,000 or 30,000) in order to avoid performance issues with the OS/filesystem. 
    """
    entry_list = []
    for entry in os.listdir(dir):
        entry_list.append(entry)
    entry_list.sort()
    
    n_entries = len(entry_list)
    
    if n_entries >= max_n_entries or run_always:
        subdir_counter = 0
        subdir_entry_counter = 0
        subdir_pathstr = dir + "/%05d"  %(subdir_counter)
        
        if chk_mkdir(subdir_pathstr,True) == False:
            sys.exit("Naming conflict!")
        
        for entry in entry_list:
            tmp_str = "mv " + entry + " " + subdir_pathstr + "/." 
            os.system(tmp_str)
            subdir_entry_counter +=1
            if subdir_entry_counter >= max_n_entries:
                subdir_counter += 1
                subdir_entry_counter = 0
                subdir_pathstr = dir + "/%05d"  %(subdir_counter)
                if chk_mkdir(subdir_pathstr,True) == False:
                    sys.exit("Naming conflict!")
                
##################################################################################################

def chk_mkdir(dir,warning=False):
    """(chk_mkdir):
        This function checks whether a directory exists and if not creates it.
    """
    if not os.path.isdir(dir):
        tmp_str = "mkdir -p " + dir
        os.system(tmp_str)
    elif warning:
        return False

##################################################################################################

def chk_rmdir(dir,check='any'):
    """(chk_rmdir):
        This function checks whether a directory exists and removes it, if it is empty.
    """
    if os.path.isdir(dir):
        n_dirs = 0
        n_files = 0
        for i in os.listdir(dir):
            if os.path.isdir(dir + '/' + i):
                n_dirs += 1
            elif os.path.isfile(dir + '/' + i):
                n_files += 1
        if n_dirs == 0 and n_files == 0:
            tmp_str = "rm -rf " + dir
        elif n_dirs == 0 and check=='dirs':
            tmp_str = "rm -rf " + dir
        elif n_files == 0 and check=='files':
            tmp_str = "rm -rf " + dir
        else:
            tmp_str = " "
        os.system(tmp_str)

##################################################################################################

def chk_rmfile(file_namestr):
    """(chk_rmfile):
        This function checks whether a file is empty and if yes deletes it.
    """
    file = open(file_namestr,'r')
    test_str = file.read()
    file.close()
    if len(test_str) == 0:
        os.remove(file_namestr)
    
##################################################################################################

def target_dir_struct(target_dir_path, maxitems = 10000, digits=5):
    """(target_dir_struct):
        This function checks whether a target dir exists and establishes/checks the subdir structure.
    """
    # check if target_dir exists and if not create it
    chk_mkdir(target_dir_path)
    # establish target_dir structure
    # 1) get all the present subdirs
    target_subdir_list = [] # fill with all present subfolders
    for i in os.listdir(target_dir_path):
        if os.path.isdir(target_dir_path + '/' + i) and i not in target_subdir_list:
            target_subdir_list.append(i)
    # 2a) if there are no subfolders present
    if len(target_subdir_list)==0:
        target_subdir = 0   # this is the highest folder
        target_subdir_n = 0 # this is the number of items in it
    # 2b) if there are subfolders present    
    else:
        target_subdir_list.sort()
        target_subdir = int(target_subdir_list[-1]) # pick the highest folder
        target_subdir_n = wc_all(target_dir_path + '/' + target_subdir_list[-1])
        if target_subdir_n >= maxitems:     # this limit is more important for folders rather than files (in this case tarballs); but we do it anyways
            target_subdir += 1
            target_subdir_n = 0

    target_subdir_pathstr = target_dir_path + '/' + '{num:{fill}{width}}'.format(num=target_subdir, fill='0', width=digits)
#    target_subdir_pathstr = target_dir_path + "/%05d"  %(target_subdir)
    chk_mkdir(target_subdir_pathstr)
    return target_subdir, target_subdir_n, target_subdir_pathstr
    
##################################################################################################

def mv2subdir_struct(source_dir_pathstr, target_subdir, target_subdir_n, target_subdir_pathstr, maxitems = 10000):
    """(mv2subdir_struct):
        This function moves a source folder into a target subdir structure and updates it.
    """
    # move
    tmp_str = 'mv ' + source_dir_pathstr + ' ' + target_subdir_pathstr + '/. ' 
    os.system(tmp_str)
    target_subdir_n += 1

    # check if limit is reached
    if target_subdir_n >= maxitems:     # this limit is more important for folders rather than files (in this case tarballs); but we do it anyways
        target_subdir += 1
        target_subdir_n = 0
        
        # make new target subdir
        tmp_str = target_subdir_pathstr.split('/')[-1]
        digits = len(tmp_str)
        target_subdir_pathstr = target_subdir_pathstr[:-digits] + '{num:{fill}{width}}'.format(num=target_subdir, fill='0', width=digits)
        chk_mkdir(target_subdir_pathstr)
    return target_subdir, target_subdir_n, target_subdir_pathstr
    
##################################################################################################

def std_datetime_str(mode='datetime'):
    """(std_time_str):
        This function gives out the formatted time as a standard string, i.e., YYYY-MM-DD hh:mm:ss.
    """
    if mode == 'datetime':
        return str(datetime.datetime.now())[:19]
    elif mode == 'date':
        return str(datetime.datetime.now())[:10]
    elif mode == 'time':
        return str(datetime.datetime.now())[11:19]
    elif mode == 'datetime_ms':
        return str(datetime.datetime.now())
    elif mode == 'time_ms':
        return str(datetime.datetime.now())[11:]
    else:
        sys.exit("Invalid mode!")

##################################################################################################
def tot_exec_time_str(time_start):
    """(tot_exec_time_str):
        This function gives out the formatted time string.
    """
    time_end = time.time()
    exec_time = time_end-time_start
    tmp_str = "execution time: %0.2fs (%dh %dm %0.2fs)" %(exec_time, exec_time/3600, (exec_time%3600)/60,(exec_time%3600)%60)
    return tmp_str

##################################################################################################

def intermed_exec_timing(time_start,intermed_n,total_n,n_str="n"):
    """(intermed_exec_timing):
        This function gives out the intermediate timing, speed, pace, projected remaining and end time.
    """
    tmp_time = time.time()
    tmp_exec_time = tmp_time-time_start
    sec_per_n = 1.0*tmp_exec_time/intermed_n
    n_per_hour = 3600.0/sec_per_n
    proj_rest_sec = sec_per_n*(total_n-intermed_n)
    proj_end_time = int(round(tmp_time + proj_rest_sec))
    tmp_str = "   Current speed: %0.2f " %(n_per_hour)
    tmp_str += n_str + "'s/hour; current pace: %0.3f " %(sec_per_n)
    tmp_str += "sec/" + n_str + "\n" 
#    tmp_str +="   Projected remaining time: %0.2fs (%dh %dm %0.2fs) " %(proj_rest_sec, proj_rest_sec/3600, (proj_rest_sec%3600)/60,(proj_rest_sec%3600)%60)
    tmp_str +="   Projected remaining time: %0.2fs (%dh %dm %0.2fs) \n" %(proj_rest_sec, proj_rest_sec/3600, (proj_rest_sec%3600)/60,(proj_rest_sec%3600)%60)
    tmp_str +="   Projected end time: " + time.ctime(proj_end_time) 
    return tmp_str

##################################################################################################

def intermed_process_timing(time_start,process_n,intermed_n,total_n,n_str="n"):
    """(intermed_process_timing):
        This function gives out the intermediate timing, speed, pace, projected remaining and end time of a particular process with restarted time.
    """
    tmp_time = time.time()
    tmp_exec_time = tmp_time-time_start
    if process_n == 0:
        return ''
    
    sec_per_n = 1.0*tmp_exec_time/process_n
    n_per_hour = 3600.0/sec_per_n
    proj_rest_sec = sec_per_n*(total_n-intermed_n)
    proj_end_time = int(round(tmp_time + proj_rest_sec))
    tmp_str = "   Current speed: %0.2f " %(n_per_hour)
    tmp_str += n_str + "'s/hour; current pace: %0.3f " %(sec_per_n)
    tmp_str += "sec/" + n_str + "\n" 
#    tmp_str +="   Projected remaining time: %0.2fs (%dh %dm %0.2fs) " %(proj_rest_sec, proj_rest_sec/3600, (proj_rest_sec%3600)/60,(proj_rest_sec%3600)%60)
    tmp_str +="   Projected remaining time: %0.2fs (%dh %dm %0.2fs) \n" %(proj_rest_sec, proj_rest_sec/3600, (proj_rest_sec%3600)/60,(proj_rest_sec%3600)%60)
    tmp_str +="   Projected end time: " + time.ctime(proj_end_time) 
    return tmp_str

##################################################################################################

def timeit(func):
    """(timeit):
        Annotate a function with its elapsed execution time.
    """
    def timed_func(*args, **kwargs):
        t1 = time.time()
        
        try:
            func(*args, **kwargs)
        finally:
            t2 = time.time()

        timed_func.func_time = ((t2 - t1) / 60.0, t2 - t1)

        if __debug__:
            sys.stdout.write("%s took %0.3fm %0.3fs %0.3fms\n" % (
                func.func_name,
                timed_func.func_time[0],
                timed_func.func_time[1],
            ))

    return timed_func

###################################################################################################

def dsu_sort(list, index, reverse=False):
    """(dsu_sort):
    """
# TODO: infoline
    for i, e in enumerate(list):
        list[i] = (e[index], e)
    if reverse:
        list.sort(reverse=True)
    else:
        list.sort()
    for i, e in enumerate(list):
        list[i] = e[1]
    return list

###################################################################################################

def dsu_sort2(list, index, reverse=False):
    """(dsu_sort2):
        This function sorts only based on the primary element, not on secondary elements in case of equality.
    """
    for i, e in enumerate(list):
        list[i] = e[index]
    if reverse:
        list.sort(reverse=True)
    else:
        list.sort()
    for i, e in enumerate(list):
        list[i] = e[1]
    return list

###################################################################################################

def bin_file_format_change(infile_namestr,outfile_namestr,mode):
    """(bin_file_format_change):
        This function reads in a binary file of a certain format, converts it, and gives out a binary file of the new format.
    """
# TODO: open file and read in
# TODO: this needs to be a binary read
    infile = open(infile_namestr,'rb',0)
    in_bin_str = infile.read()
    infile.close()
# TODO: change binary format
# TODO: this needs to be a binary write
    outfile = open(outfile_namestr,'wb',0)    
    if mode == 'sp2dp':
        in_bin = fromstring(in_bin_str,float32)
        sys.exit()
    elif mode == 'dp2sp':
        in_bin = fromstring(in_bin_str,float64)
        sys.exit()
    else:
        sys.exit("Unknown binary format conversion mode.")
# TODO: make new file and dump
    
    outfile.close()

###################################################################################################

def isFloat(x):
    if x in ('NAN','NaN','Nan','nan'):
        return 0
    elif '#IND' in x:
        return 0
    try:
        float(x)
        return 1
    except:
        return 0

###################################################################################################

def md5checksum(file_path, blocksize=8192): # 4kB blocks
    """(md5checksum):
        Compute md5 hash of the specified file.
    """
    file = open(file_path, 'rb')
    md5sum = hashlib.md5()
    while True:
        data = file.read(blocksize)
        if not data:
            break
        md5sum.update(data)
    file.close()
    return md5sum.hexdigest()

###################################################################################################

def filelinecount(filename):
    """(filelinecount):
        Counts the number of lines in a file.
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

###################################################################################################

def revdict_lookup(dict,lookup_val):
    """(revdict_lookup):
        Performs a reverse dictionary lookup. Careful: only returns first match, but there may be others.
    """    
    key = (key for key,value in dict.items() if value==lookup_val).next()
    return key


###################################################################################################

def queryset_iterator(queryset, chunksize=1000, reverse=False, id_only=False, values= False):
    """(queryset_iterator):
        Django incremental queryset iterator.
        Found on: http://www.poeschko.com/2012/02/memory-efficient-django-queries/
    """    
    ordering = '-' if reverse else ''
    queryset = queryset.order_by(ordering + 'pk')
    last_pk = None
    new_items = True
    while new_items:
        new_items = False
        chunk = queryset
        if last_pk is not None:
            func = 'lt' if reverse else 'gt'
            chunk = chunk.filter(**{'pk__' + func: last_pk})
        chunk = chunk[:chunksize]
        if id_only:
            chunk = chunk.values('pk')            
        row = None
        for row in chunk:
            yield row
        if row is not None:
            if id_only or values:
                last_pk = row['pk']                
            else:
                last_pk = row.pk
            new_items = True

###################################################################################################

def list_chunks(l, n):
    """ Yield successive n-sized chunks from l.
        Found on: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]