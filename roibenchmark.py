'''ROI benchmark - calculate benchmark statistics for some storage strategies

1 lzw-encoded tiff plane per ROI

1 lzw-encoded tiff plane per image, each ROI has unique integer value

run-length encoding + zip deflate

encode as N x 2 pixel coordinates, shuffled, zip deflate encoded

skeleton reconstruction, shuffled, zip deflate encoded

also record size without compression

\\iodine-cifs\imaging_analysis\2008_04_15_Lithium_Neurons_JenPan\ChristinaWright\Image Processing\07_20_2010_PIPE_16D_Output\outlines
'''

import numpy as np
from scipy.ndimage import distance_transform_edt
import libtiff
import zlib

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import skeletonize_labels, color_labels

C_ROI_BENCHMARK = "ROIBenchmark"
FTR_BINARY_PLANAR = "BinaryPlanar"
FTR_INTEGER_PLANAR = "IntegerPlanar"
FTR_RLE = "RLE"
FTR_IJ = "IJ"
FTR_SKEL = "Skeleton"

M_BINARY_PLANAR, M_INTEGER_PLANAR, M_RLE, M_IJ, M_SKEL = [
    "_".join((C_ROI_BENCHMARK, x)) for x in 
    (FTR_BINARY_PLANAR, FTR_INTEGER_PLANAR, FTR_RLE, FTR_IJ, FTR_SKEL)]

M_BINARY_PLANAR_RAW, M_INTEGER_PLANAR_RAW, M_RLE_RAW, M_IJ_RAW, M_SKEL_RAW = [
    x+"Raw" for x in (M_BINARY_PLANAR, M_INTEGER_PLANAR, M_RLE, M_IJ, M_SKEL)]

M_RLE_COMPOSITE = M_RLE+"Composite"
M_IJ_COMPOSITE = M_IJ+"Composite"
M_SKEL_COMPOSITE = M_SKEL+"Composite"

M_ALL = (M_BINARY_PLANAR, M_INTEGER_PLANAR, M_RLE, M_IJ, M_SKEL,
         M_BINARY_PLANAR_RAW, M_INTEGER_PLANAR_RAW, M_RLE_RAW, 
         M_IJ_RAW, M_SKEL_RAW, M_IJ_COMPOSITE, M_RLE_COMPOSITE, M_SKEL_COMPOSITE)

class ROIBenchmark(cpm.CPModule):
    variable_revision_number = 1
    module_name = "ROIBenchmark"
    category = "Measurements"
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber(
            "Objects name", "None",
            doc = "Name of the objects to benchmark")
        
    def settings(self):
        return [self.objects_name]
    
    @staticmethod
    def compress(vectors):
        '''For each vector, encode the difference between vector elements
        
        Seed the machine with the value, 0, then record the difference between
        the last element and the next element for each vector. Concatenate
        and use zlib.compress to compress.
        '''
        if len(vectors[0]) == 0:
            return ""
        elif len(vectors[0]) == 1:
            vv = vectors
        else:
            vv = [np.hstack(([v[0]], v[1:] - v[:-1])) for v in vectors]
        return zlib.compress("".join([v.tostring() for v in vv]))
    
    def run(self, workspace):
        m = workspace.measurements
        objects = workspace.object_set.get_objects(self.objects_name.value)
        assert isinstance(objects, cpo.Objects)
        labels = objects.segmented
        
        benchmarks = dict([(x, np.zeros(objects.count, int))
                           for x in M_ALL])
        ii, jj = np.mgrid[0: objects.shape[0], 0:objects.shape[1]]
        benchmarks[M_INTEGER_PLANAR_RAW][:] = np.prod(labels.shape) * 2 / objects.count
        b = libtiff.tif_lzw.encode(labels)
        benchmarks[M_INTEGER_PLANAR][:] = len(b) / objects.count
        #
        # Do distance transform in pieces in case objects touch
        #
        distance = np.zeros(objects.shape, np.float32)
        clabels = color_labels(labels)
        for i in range(1, np.max(clabels)+1):
            mask = clabels == i
            distance[mask] = distance_transform_edt(mask)[mask]
        maskj = np.zeros((objects.shape[0], objects.shape[1]+1), bool)
        skel = skeletonize_labels(labels)
        skel_composite = []
        rle_composite = []
        ij_composite = []
        for i in range(1, objects.count+1):
            mask = labels == i
            maskj[:, :-1] = mask
            (imin, imax), (jmin, jmax) = [[f(d) for f in (np.min, np.max)]
                                          for d in (ii[mask], jj[mask])]
            #
            ####################
            #
            # Binary planar
            #
            benchmarks[M_BINARY_PLANAR_RAW][i-1] = (imax - imin + 1) * (jmax - jmin + 1)
            b = libtiff.tif_lzw.encode(mask[imin:(imax+1), jmin:(jmax+1)])
            #
            # Compressed + 16-bit origin & size
            #
            benchmarks[M_BINARY_PLANAR][i-1] = len(b) + 8
            #
            ####################
            #
            # RLE encoded
            #
            fm = (~ maskj[:, :-1]) & maskj[:, 1:]
            lm = maskj[:, :-1] & ~ maskj[:, 1:]
            iif = ii[fm]
            jjf = jj[fm] + 1
            iil = ii[lm]
            jjl = jj[lm] + 1
            order = np.lexsort((iif, jjf))
            iif, jjf = iif[order], jjf[order]
            order = np.lexsort((iil, jjl))
            iil, jjl = iil[order], jjl[order]
            data = np.column_stack((iif, jjf, jjl - jjf + 1))
            benchmarks[M_RLE_RAW][i-1] = np.prod(data.shape) * 2
            rle_composite.append(data)
            b = self.compress((data[:, 0], data[:, 1], data[:, 2]))
            benchmarks[M_RLE][i-1] = len(b)
            #
            ###################
            #
            # IJ
            #
            data = np.column_stack((ii[mask], jj[mask]))
            order = np.lexsort((jj[mask], ii[mask]))
            data = data[order, :]
            ij_composite.append(data)
            benchmarks[M_IJ_RAW][i-1] = np.prod(data.shape) * 2
            b = self.compress((data[:, 0], data[:, 1]))
            benchmarks[M_IJ][i-1] = len(b)
            
            skmask = skel == i
            iii = ii[skmask]
            jjj = jj[skmask]
            ddd = distance[skmask]
            order = np.lexsort((jjj, iii))
            data = np.rec.array((iii[order], jjj[order], ddd[order]),
                                dtype = [("i", np.uint16), ("j", np.uint16),
                                         ("d", np.float32)])
            benchmarks[M_SKEL_RAW][i-1] = len(data.tostring())
            b = self.compress((data["i"], data["j"], data["d"]))
            benchmarks[M_SKEL][i-1] = len(b)
            skel_composite.append(data)
            
        for data, idxs, ftr in (
            (np.vstack(ij_composite).transpose(), (0, 1), M_IJ_COMPOSITE),
            (np.vstack(rle_composite).transpose(), (0, 1, 2), M_RLE_COMPOSITE),
            (np.hstack(skel_composite), ("i", "j", "d"), M_SKEL_COMPOSITE)):
            b = self.compress([data[idx] for idx in idxs])
            benchmarks[ftr][:] = len(b) / objects.count + 4
        for key in benchmarks:
            m.add_measurement(self.objects_name.value, key, benchmarks[key])
            
    def get_measurement_columns(self, pipeline):
        return [(self.objects_name.value, ftr, cpmeas.COLTYPE_INTEGER)
                for ftr in M_ALL]

if __name__=="__main__":
    from matplotlib.font_manager import FontProperties
    import pylab
    import sys
    import csv
    
    fd = open(sys.argv[1], "r")
    rdr = csv.reader(fd)
    header = rdr.next()
    rows = np.atleast_2d([[float(x) for x in row] for row in rdr])
    fd.close()
    hi = dict([(x, i) for i, x in enumerate(header)
               if x in M_ALL])
    boxplots = pylab.boxplot([rows[:, hi[x]] for x in M_ALL])
    pylab.gca().set_yscale('log')
    names = [x[(len(C_ROI_BENCHMARK)+1):] for x in M_ALL]
    ticklabels = pylab.setp(pylab.gca(), xticklabels = names)
    pylab.setp(ticklabels, rotation=45, fontsize='x-small', horizontalalignment='right')
    pylab.subplots_adjust(bottom = .15)
    if len(sys.argv) > 2:
        pylab.title(sys.argv[2])
    pylab.show()
    