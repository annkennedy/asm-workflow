import os, sys
import numpy as np
import PIL
from PIL import Image
import io
import struct
from datetime import datetime, timedelta,date
import time
from matplotlib.dates import date2num, num2date
import colour_demosaicing
import skvideo.io
import re
import copy
import pickle
import pdb
import cv2
import progressbar as pb
import array


class S3File(io.RawIOBase):
    # added a new class to support read/write/seek from S3 files
    def __init__(self, s3_object):
        self.s3_object = s3_object
        self.position = 0

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        return self.s3_object.content_length

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence (%r, should be %d, %d, %d)" % (
                whence, io.SEEK_SET, io.SEEK_CUR, io.SEEK_END
            ))

        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        if size == -1:
            # Read to the end of the file
            range_header = "bytes=%d-" % self.position
            self.seek(offset=0, whence=io.SEEK_END)
        else:
            new_position = self.position + size

            # If we're going to read beyond the end of the object, return
            # the entire object.
            if new_position >= self.size:
                return self.read()

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)

        return self.s3_object.get(Range=range_header)["Body"].read()

    def readable(self):
        return True

# Create interface sr for reading seq files.
#   sr = seqIo_reader( fName )
# Create interface sw for writing seq files.
#   sw = seqIo_Writer( fName, header )
# Crop sub-sequence from seq file.
#   seqIo_crop( fName, 'crop', tName, frames )
# Extract images from seq file to target directory or array.
#   Is = seqIo_toImgs( fName, tDir=[],skip=1,f0=0,f1=np.inf,ext='' )
# Create seq file from an array or directory of images or from an AVI file. DONE
#   seqIo_frImgs( fName, fName,header,aviName=[],Is=[],sDir=[],name='I',ndig=5,f0=0,f1=1e6 )
# Convert seq file by applying imgFun(I) to each frame I.
#   seqIo( fName, 'convert', tName, imgFun, varargin )
# Replace header of seq file with provided info.
#   seqIo( fName, 'newHeader', info )
# Create interface sr for reading dual seq files.
#   sr = seqIo( fNames, 'readerDual', [cache] )

FRAME_FORMAT_RAW_GRAY = 100 #RAW
FRAME_FORMAT_RAW_COLOR = 200 #RAW
FRAME_FORMAT_JPEG_GRAY = 102 #JPG
FRAME_FORMAT_JPEG_COLOR = 201 #JPG
FRAME_FORMAT_MONOB = 101 #BRGB8
FRAME_FORMAT_MONOB_JPEG = 103 #JBRGB
FRAME_FORMAT_PNG_GRAY = 0o1 #PNG
FRAME_FORMAT_PNG_COLOR =0o2 #PNG

def fread(fid, nelements, dtype):

    """Equivalent to Matlab fread function"""

    data_array = array.array(dtype)    
    data_array.fromfile(fid, nelements)
    if len(data_array)==1:
        data_array = data_array[0]
    return data_array

def fwrite(fid,a,dtype='B'):
    # assuming 8but ASCII for string
    if dtype is np.str:
        dt = np.uint8
    else:
        dt = dtype
    if isinstance(a,np.ndarray):
        data_array = a.astype(dtype)
    else:
        data_array = np.array(a).astype(dttype)
    data_array.tofile(fid)

class seqIo_reader():
    def __init__(self,filename,s3_resource,info=[]):
        self.filename = filename
        
        #new!
        bucket,movie_file = os.path.split(filename)
        bucket = bucket.replace('s3://','')
        s3_obj = s3_resource.Object(bucket_name=bucket, key=movie_file)
        
        try: self.file = S3File(s3_obj)
        except EnvironmentError as e: print(os.strerror(e.errno))
        self.header={}
        self.seek_table=None
        self.frames_read=-1
        self.timestamp_length = 10
        if info==[]:
            self.readHeader()
        else:
            info.numFrames=0
        self.buildSeekTable(False)


    def readHeader(self):
        #make sure we do this at the beginning of the file
        assert self.frames_read == -1, "Can only read header from beginning of file"
        self.file.seek(0,0)
        # pdb.set_trace()

        # Read 1024 bytes (len of header)
        tmp = fread(self.file,1024,'B')        
        #check that the header is not all 0's
        n=len(tmp)
        if n<1024:raise ValueError('no header')
        
        self.file.seek(0,0)
        #first 4 bytes stor 0XFEED next 24 store 'Norpix seq '
        magic_number = fread(self.file,1,'I')
        name = fread(self.file,10,'H')
        name = ''.join(map(chr,name))
        if not '{0:X}'.format(magic_number)=='FEED' or not name=='Norpix seq':raise ValueError('invalid header')
        self.file.seek(4,1)
        #next 8 bytes for version and header size (1024) then 512 for desc
        version = int(fread(self.file,1,'i'))
        hsize =int(fread(self.file,1,'I'))
        assert(hsize)==1024 ,"incorrect header size"
        # d = self.file.read(512)
        descr=fread(self.file,256,'H')
        # descr = ''.join(map(chr,descr))
        # descr = ''.join(map(unichr,descr)).replace('\x00',' ')
        descr = ''.join([chr(x) for x in descr]).replace('\x00',' ')
        descr = descr.encode('utf-8')
        #read more info
        tmp = fread(self.file,9,'I')
        assert tmp[7]==0, "incorrect origin"
        fps = fread(self.file,1,'d')
        codec = 'imageFormat' + '%03d'%tmp[5]
        desc_format = fread(self.file,1,'I')
        padding = fread(self.file,428,'B')
        padding = ''.join(map(chr,padding))
        #store info
        self.header={'magicNumber':magic_number,
                     'name':name,
                     'seqVersion': version,
                     'headerSize':hsize,
                     'descr': descr,
                     'width':int(tmp[0]),
                     'height':int(tmp[1]),
                     'imageBitDepth':int(tmp[2]),
                     'imageBitDepthReal':int(tmp[3]),
                     'imageSizeBytes':int(tmp[4]),
                     'imageFormat':int(tmp[5]),
                     'numFrames':int(tmp[6]),
                     'origin':int(tmp[7]),
                     'trueImageSize':int(tmp[8]),
                     'fps':fps,
                     'codec':codec,
                     'descFormat':desc_format,
                     'padding':padding,
                     'nHiddenFinalFrames':0
                     }
        assert(self.header['imageBitDepthReal']==8)
        # seek to end fo header
        self.file.seek(432,1)
        self.frames_read += 1

        self.imageFormat = self.header['imageFormat']
        if self.imageFormat in (100,200):   self.ext = 'raw'
        elif self.imageFormat in (102,201): self.ext = 'jpg'
        elif self.imageFormat in(0o1,0o2):  self.ext = 'png'
        elif self.imageFormat == 101:       self.ext = 'brgb8'
        elif self.imageFormat == 103:       self.ext = 'jbrgb'
        else:                              raise ValueError('uknown format')

        self.compressed = True if self.ext in ['jpg','jbrgb','png','brgb8'] else False
        self.bit_depth = self.header['imageBitDepth']

        # My code uses a timestamp_length of 10 bytes, old uses 8. Check if not 10
        if self.bit_depth / 8 * (self.header['height'] * self.header['width']) + self.timestamp_length \
                != self.header['trueImageSize']:
            # If not 10, adjust to actual (likely 8) and print message
            self.timestamp_length = int(self.header['trueImageSize'] \
                                        - (self.bit_depth / 8 * (self.header['height'] * self.header['width'])))

    def buildSeekTable(self,memoize=False):
        """Build a seek table containing the offset and frame size for every frame in the video."""
        pickle_name = self.filename.strip(".seq") + ".seek"
        if memoize:
            if os.path.isfile(pickle_name):
                self.seek_table = pickle.load(open(pickle_name, 'rb'))
                return

        # assert self.header['numFrames']>0
        n=self.header['numFrames']
        if n==0:n=1e7

        seek_table = np.zeros((n)).astype(int)
        seek_table[0]=1024
        extra = 8 # extra bytes after image data , 8 for ts then 0 or 8 empty
        self.file.seek(1024,0)
        #compressed case

        if self.compressed:
            i=1
            print('building seek table for %d frames (this takes a while) ...' % n)
            while (True):
                if(i%1000==0):
                    print('frame %d' % i)
                try:
                    # size = fread(self.file,1,np.uint32)
                    # offset = seek_table[i-1] + size +extra
                    # seek_table[i]=offset
                    # # seek_table[i-1,1]=size
                    # self.file.seek(size-4+extra,1)

                    size = fread(self.file, 1, 'I')
                    offset = seek_table[i - 1] + size + extra
                    # self.file.seek(size-4+extra,1)
                    self.file.seek(offset, 0)
                    if i == 1:
                        if fread(self.file, 1, 'I') != 0:
                            self.file.seek(-4, 1)
                        else:
                            extra += 8;
                            offset += 8
                            self.file.seek(offset, 0)

                    seek_table[i] = offset
                    # seek_table[i-1,1]=size
                    i+=1
                except:
                    break
                    #most likely EOF
        else:
            #uncompressed case
            assert (self.header['numFrames']>0)
            frames = range(0, self.header["numFrames"])
            offsets = [x * self.header["trueImageSize"] + 1024 for x in frames]
            for i,offset in enumerate(offsets):
                seek_table[i]=offset
                # seek_table[i,1]=self.header["imageSize"]
        if n==1e7:
            n = np.minimum(n,i)
            self.seek_table=seek_table[:n]
            self.header['numFrames']=n
        else:
            self.seek_table=seek_table
        if memoize:
            pickle.dump(seek_table,open(pickle_name,'wb'))

        #compute frame rate from timestamps as stored fps may be incorrect
        # if n==1: return
        # ds = self.ts[1:100]-self.ts[:99]
        # ds = ds[abs(ds-np.median(ds))<.005]
        # if bool(np.prod(ds)): self.header['fps']=1/np.mean(ds)

    def getFrame(self,index,decode=True):
        #get frame image (I) and timestamp (ts) at which frame was recorded
        nch = self.header['imageBitDepth']/8
        if self.ext in ['raw','brgb8']: #read in an uncompressed iamge( assume imageBitDepthReal==8)
            shape = (self.header['height'], self.header['width'])
            self.file.seek(1024 + index*self.header['trueImageSize'],0)
            I = fread(self.file,self.header['imageSizeBytes'],'B')

            if decode:
                if nch==1:
                    I=np.reshape(I,shape)
                else:
                    I=np.reshape(I,(shape,nch))
                if nch==3:
                    t=I[:,:,2]; I[:,:,2]=I[:,:,0]; I[:,:,1]=t
                if self.ext=='brgb8':
                    I= colour_demosaicing.demosaicing_CFA_Bayer_bilinear(I,'BGGR')

        elif self.ext in ['jpg','jbrgb']:
            if decode:
                self.file.seek(self.seek_table[index],0)
                nBytes = fread(self.file,1,'I')
                data = fread(self.file,nBytes-4,'B')
                I = PIL.Image.open(io.BytesIO(data))
                if self.ext == 'jbrgb':
                    I=colour_demosaicing.demosaicing_CFA_Bayer_bilinear(I,'BGGR')

        elif self.ext=='png':
            self.file.seek(self.seek_table[index],0)
            nBytes = fread(self.file,1,'I')
            I= fread(self.file,nBytes-4,'B')
            if decode:
                I= np.array(I).transpose(range(I.shape,-1,-1))
        else: assert(False)
        return np.array(I)

    # Close the file
    def close(self):
        self.file.close()


#mimium header
# header = {'width': IM_TOP_W,
#           'height': IM_TOP_H,
#           'fps': fps,
#           'codec': 'imageFormat102'}
# filename= '/media/cristina/MARS_data/mice_project/teresa/Mouse156_20161017_17-22-09/Mouse156_20161017_17-22-09_Top_J85.seq'
# filename_out = filename[:-4] + '_new.seq'
# reader = seqIo_reader(filename)
# reader.header
# Initialize a SEQ writer
# writer = seqIo_writer(filename_out,reader.header)
# I,ts = reader.getFrame(0)
# writer.addFrame(I,ts)
# for f in range(8):
#     I,ts = reader.getFrame(f)
#     print writer.file.tell()
#     writer.addFrame(I,ts)
# writer.close()
# reader.close()



