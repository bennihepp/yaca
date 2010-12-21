import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy
import struct
import Image
import ImageChops
import ImageEnhance
from libtiff import TIFF


TMP_IMAGE_FILENAME_TEMPLATE = '/dev/shm/adc-tmp-image-file-%d.tiff'



class CellFeatureTextFactory(object):

    def __init__(self, adc, objMask=None, mahalFeatures=None):
        self.adc = adc
        if objMask != None:
            self.features = self.adc.objFeatures[ objMask ]
        else:
            self.features = self.adc.objFeatures[ : ]
        self.mahalFeatures = mahalFeatures
        self.mahalFeatureIdOffset = len( self.adc.objFeatureIds )

    def createFeatureText(self, index, featureId):

        #print 'index=%s, featureId=%s' % (str(index),str(featureId))

        if featureId < self.mahalFeatureIdOffset or self.mahalFeatures == None:
            if featureId == self.adc.objTreatmentFeatureId:
                return self.adc.treatments[ int( self.features[ index, featureId ] ) ].name
            else:
                return self.features[ index , featureId ]

        else:
            featureId = featureId - self.mahalFeatureIdOffset
            return self.mahalFeatures[ index, featureId ]

class ImageFeatureTextFactory(object):

    def __init__(self, adc, imgMask=None):
        self.adc = adc
        if imgMask != None:
            self.features = self.adc.imgFeatures[ imgMask ]
        else:
            self.features = self.adc.imgFeatures[ : ]

    def createFeatureText(self, index, featureId):

        #print 'index=%s, featureId=%s' % (str(index),str(featureId))

        if featureId == self.adc.imgTreatmentFeatureId:
            return self.adc.treatments[ int( self.features[ index, featureId ] ) ].name
        else:
            return self.features[ index , featureId ]



class ImagePixmapFactory(object):

    def __init__(self, adc, channelMapping ):
        self.adc = adc
        self.channelMapping = channelMapping

    def getObjIndexMultiplier(self, img):

        pixel_arr = self.convertImageToShortArray( img )
        mask = pixel_arr > 0
        multiplier = numpy.min( pixel_arr[ mask ] )

        pixel_arr /= multiplier
        d = {}
        for p in pixel_arr:
            if p not in d:
                d[p] = 0
            d[p] += 1

        return multiplier

    def maskPixelArray(self, pixel_arr, objIndex):

        #pixel_arr = pixel_arr / multiplier

        mask = numpy.logical_or( pixel_arr > objIndex, pixel_arr < objIndex )
        #mask = pixel_arr < objIndex
        pixel_arr[ mask ] = 0
        pixel_arr[ numpy.invert( mask ) ] = 255

        return pixel_arr

    def adjustImage(self, pixel_arr, black, white, brightness, invert, binary):

        mask = pixel_arr > white
        pixel_arr[ mask ] = 255
        mask = pixel_arr < black
        pixel_arr[ mask ] = 0

        if invert:
            pixel_arr = 255 - pixel_arr

        if binary:
            mask = pixel_arr > 0
            pixel_arr[ mask ] = 255
            #pixel_arr[ numpy.invert( mask ) ] = 0

        pixel_arr = pixel_arr * brightness

        if brightness > 1.0:
            mask = pixel_arr > 255
            pixel_arr[ mask ] = 255

        return pixel_arr

    def loadImage(self, path):

        return Image.open( path )

    def cropImage(self, img, rect):

        return img.crop ( rect )

    def convertImageToShortArray(self, img, img_type=None):

        pixel_arr = numpy.array( img.getdata() )

        if img_type == None:
            img_type = self.getImageType( img )

        # Transform 16bit images to 8bit
        if img_type == 'I;16' or img_type == 'Scan-R':

            # Check for Scan-R images
            if img_type == 'Scan-R': # or ( ( pixel_arr < 0 ).any() and ( pixel_arr <= 0 ).all() ):
                pixel_arr = pixel_arr + 2**15
                #pixel_arr = pixel_arr * (2**8-1.0)/(2**12-1.0)

            else:
                mask = pixel_arr < 0
                pixel_arr[ mask ] += 2**16
                #pixel_arr = pixel_arr * (2**8-1.0)/(2**16-1.0)

        return pixel_arr

    def convertImageToByteArray(self, img, img_type=None):

        pixel_arr = numpy.array( img.getdata() )

        if img_type == None:
            img_type = self.getImageType( img )

        # Transform 16bit images to 8bit
        if img_type == 'I;16' or img_type == 'Scan-R':

            # Check for Scan-R images
            if img_type == 'Scan-R': # or ( ( pixel_arr < 0 ).any() and ( pixel_arr <= 0 ).all() ):
                pixel_arr = pixel_arr + 2**15
                pixel_arr = pixel_arr * (2**8-1.0)/(2**12-1.0)

            else:
                mask = pixel_arr < 0
                pixel_arr[ mask ] += 2**16
                pixel_arr = pixel_arr * (2**8-1.0)/(2**16-1.0)

        return pixel_arr

    def convertByteArrayToImage(self, pixel_arr, img_size):
        tmp_str = struct.pack( '@%dB' % len( pixel_arr ), *pixel_arr)
        return Image.fromstring( 'L', img_size, tmp_str, "raw", "L", 0, 1 )

    def getImageType(self, img):

        img_type = img.mode
        if img.format == 'TIFF':
            for k in img.tag.keys():
                if 'National Instruments IMAQ' in img.tag[ k ]:
                    img_type = 'Scan-R'
                    break

        return img_type


    def createImageMask(self, index, channel, left, top, width, height, tmp_image_filename=None, imageCache=None, cacheId=None):

        use_cache = imageCache != None

        if cacheId == None:
            cacheId = int( index )

        if not tmp_image_filename:
            tmp_image_filename = self.TMP_IMAGE_FILENAME_TEMPLATE % \
                                 ( str( numpy.random.randint(sys.maxint) ), str( time.time() ) )

        imageFiles = self.adc.images[ index ].imageFiles

        path = None
        for n,p in imageFiles:
            if n == channel:
                path = p
                break

        if path == None:
            raise Exception( 'The specified channel is not valid: %s' % channel )

        #objIndex = int( self.features[ index , self.adc.objFeatureIds[ 'Cells_Number_Object_Number' ] ] )

        rect = None

        img_size = None

        if use_cache:

            if not cacheId in imageCache:
                imageCache[ cacheId ] = {}

            files_cached = ( 'files' in imageCache[ cacheId ] )

            region = ( left, top, width, height )

            if files_cached:
                oldRegion = imageCache[ cacheId ][ 'region' ]
                if oldRegion != region:
                    files_cached = False

            if not files_cached:
                imageCache[ cacheId ][ 'files' ] = {}
                imageCache[ cacheId ][ 'region' ] = region

        else:
            files_cached = False

        img = None

        if files_cached and ( channel in imageCache[ cacheId ][ 'files' ] ):
            file_cached = True
        else:
            file_cached = False

        adjustment_changed = True

        if not file_cached:

            tmp_img = self.loadImage( path )

            img_type = self.getImageType( tmp_img )

            xc = int( left + width/2.0 + 0.5 )
            yc = int( top + height/2.0 + 0.5 )
            objIndex = tmp_img.getpixel( (xc, yc ) )

            #multiplier = self.getObjIndexMultiplier( tmp_img )

            if rect == None:

                if width < 0:
                    width = tmp_img.size[ 0 ] - left
                if height < 0:
                    height = tmp_img.size[ 1 ] - top
                rect = ( left, top, left + width, top + height )

            img = self.cropImage( tmp_img, rect )
            #del tmp_img

            img_size = img.size

            pixel_arr = self.convertImageToShortArray( img, img_type )

            if use_cache:
                imageCache[ cacheId ][ 'files' ][ channel ] = {}
                imageCache[ cacheId ][ 'files' ][ channel ][ 'raw_16bit_pixels' ] = pixel_arr.copy()
                imageCache[ cacheId ][ 'files' ][ channel ][ 'image_size' ] = img_size
                imageCache[ cacheId ][ 'files' ][ channel ][ 'objIndex' ] = objIndex

        else:

            pixel_arr     = imageCache[ cacheId ][ 'files' ][ channel ][ 'raw_16bit_pixels' ]
            img_size      = imageCache[ cacheId ][ 'files' ][ channel ][ 'image_size' ]
            objIndex    = imageCache[ cacheId ][ 'files' ][ channel ][ 'objIndex' ]

            pixel_arr = pixel_arr.copy()

            adjustment_changed = False


        if adjustment_changed:

            pixel_arr = self.maskPixelArray( pixel_arr, objIndex )

            img = self.convertByteArrayToImage( pixel_arr, img_size )

            #del pixel_arr

            if use_cache:
                imageCache[ cacheId ][ 'files' ][ channel ][ 'processed_mask' ] = img.copy()

        else:

            img = imageCache[ cacheId ][ 'files' ][ channel ][ 'processed_mask' ]

        return img


    def createImage(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):

            use_cache = imageCache != None

            if cacheId == None:
                cacheId = int( index )

            if not tmp_image_filename:
                tmp_image_filename = self.TMP_IMAGE_FILENAME_TEMPLATE % \
                                     ( str( numpy.random.randint(sys.maxint) ), str( time.time() ) )

            imageFiles = self.adc.images[ index ].imageFiles

            id_to_channel_map = { 0:'R', 1:'G', 2:'B' }

            for c in channelAdjustment.keys():
                if not c in 'RGB':
                    i = len( id_to_channel_map )
                    id_to_channel_map[ i ] = c

            img_dict = {}

            rect = None

            img_size = None

            any_rgb_adjustment_changed = False

            if use_cache:

                if not cacheId in imageCache:
                    imageCache[ cacheId ] = {}

                files_cached = ( 'files' in imageCache[ cacheId ] )

                region = ( left, top, width, height )

                if files_cached:
                    oldRegion = imageCache[ cacheId ][ 'region' ]
                    if oldRegion != region:
                        files_cached = False

                if not files_cached:
                    imageCache[ cacheId ][ 'files' ] = {}
                    imageCache[ cacheId ][ 'region' ] = region

            else:
                files_cached = False

            for name,path in imageFiles:

                use_image = False

                for c,n in self.channelMapping.iteritems():
                    if n != None:
                        if name == n:
                            use_image = True
                            channel = c
                            break

                if use_image and ( channel in channelAdjustment ):

                    black,white,brightness,invert,binary = channelAdjustment[ channel ]
                    adjustment = ( black, white, brightness, invert, binary )

                    img = None

                    if files_cached and ( channel in imageCache[ cacheId ][ 'files' ] ):
                        file_cached = True
                    else:
                        file_cached = False

                    adjustment_changed = True

                    if not file_cached:

                        tmp_img = self.loadImage( path )

                        img_type = self.getImageType( tmp_img )

                        if rect == None:

                            if width < 0:
                                width = tmp_img.size[ 0 ] - left
                            if height < 0:
                                height = tmp_img.size[ 1 ] - top
                            rect = ( left, top, left + width, top + height )

                        img = self.cropImage( tmp_img, rect )
                        #del tmp_img

                        img_size = img.size

                        pixel_arr = self.convertImageToByteArray( img, img_type )

                        if use_cache:
                            imageCache[ cacheId ][ 'files' ][ channel ] = {}
                            imageCache[ cacheId ][ 'files' ][ channel ][ 'raw_pixels' ] = pixel_arr.copy()
                            imageCache[ cacheId ][ 'files' ][ channel ][ 'image_size' ] = img_size

                    else:

                        pixel_arr     = imageCache[ cacheId ][ 'files' ][ channel ][ 'raw_pixels' ]
                        img_size      = imageCache[ cacheId ][ 'files' ][ channel ][ 'image_size' ]
                        oldAdjustment = imageCache[ cacheId ][ 'files' ][ channel ][ 'adjustment' ]

                        pixel_arr = pixel_arr.copy()

                        if oldAdjustment == adjustment:

                            adjustment_changed = False

                    if adjustment_changed:

                        imageCache[ cacheId ][ 'files' ][ channel ][ 'adjustment' ] = adjustment

                        if channel in 'RGB':
                            any_rgb_adjustment_changed = True

                        pixel_arr = self.adjustImage( pixel_arr, black, white, brightness, invert, binary )

                        img = self.convertByteArrayToImage( pixel_arr, img_size )

                        #del pixel_arr

                        if use_cache:
                            imageCache[ cacheId ][ 'files' ][ channel ][ 'processed_img' ] = img.copy()

                    else:

                        img = imageCache[ cacheId ][ 'files' ][ channel ][ 'processed_img' ]

                    img_dict[ channel ] = img
                    #del img


            if img_size == None:
                if width < 0 or height < 0:
                    width = 0
                    height = 0
                img_size = ( width, height )

            imgs = []

            #if color:

            for i in xrange( len( id_to_channel_map ) ):
                c = id_to_channel_map[ i ]
                if not c in img_dict:
                    img_dict[ c ] = Image.new( 'L', img_size )
                imgs.append( img_dict[ c ] )

            #else:

            #    if len( img_dict ) > 0:
            #        imgs.append( img_dict.values()[0] )
            #    else:
            #        imgs.append( Image.new( 'L', img_size ) )

            #del img_dict


            channels = channelAdjustment.keys()
            #channels.sort()

            rgb_channels_changed = True

            if files_cached and not any_rgb_adjustment_changed:

                oldChannels = imageCache[ cacheId ][ 'channels' ]
                bool_array = [ ( c in oldChannels ) == ( c in channels ) for c in 'RGB' ]
                if numpy.all( bool_array ):
                    rgb_channels_changed = False

            if files_cached:
                oldColor = imageCache[ cacheId ][ 'color' ]
                if color != oldColor:
                    rgb_channels_changed = True

            if use_cache:
                imageCache[ cacheId ][ 'channels' ] = channels
                imageCache[ cacheId ][ 'color' ] = color

            if not rgb_channels_changed:

                merged_img = imageCache[ cacheId ][ 'merged_image' ]

            else:

                if color:
                    merged_img = Image.merge( 'RGB', imgs[ :3 ] )

                else:
                    merged_img = None
                    for c in 'RGB':
                        if c in channelAdjustment:
                            merged_img = img_dict[ c ]
                            break
                    if merged_img == None:
                        merged_img = Image.new( 'L', img_size )

                    #print '  merging gray-scale image...'
                    #print '  ', imgs[0].mode, imgs[1].mode, imgs[2].mode
                    #ie = ImageEnhance.Contrast( merged_img )
                    #merged_img = ie.enhance( 0.0 )

                if use_cache:
                    imageCache[ cacheId ][ 'merged_image' ] = merged_img.copy()

            del imgs[ :3 ]

            if len(imgs) > 0:
                #del imgs[:3]
                while len(imgs) > 1:
                    img1,img2 = imgs[0],imgs[1]
                    tmp = ImageChops.lighter( img1, img2 )
                    del imgs[0]#, img1, img2
                    imgs[0] = tmp
                #imgs[0].save('/home/benjamin/tmp.tif')
                if imgs[0].mode != merged_img.mode:
                    imgs[0] = imgs[0].convert( merged_img.mode )
                #imgs[0].save('/home/benjamin/tmp_rgb.tif')
                #merged_img.save('/home/benjamin/merged.tif')
                tmp = ImageChops.lighter( merged_img, imgs[0])
                #tmp.save('/home/benjamin/merged_tmp.tif')
                del merged_img
                merged_img = tmp

            #del imgs[:]
            #del imgs
    
            #if merged_img.mode != 'RGB':
            #    merged_img = merged_img.convert( 'RGB' )
            #merged_img.save('/home/benjamin/rgb.tif')
    
            return merged_img


    def createPixmap(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):
        img = self.createImage( index, left, top, width, height, channelAdjustment, color, tmp_image_filename, imageCache, cacheId )

        img.save( tmp_image_filename )

        pix = QPixmap( tmp_image_filename )

        os.remove( tmp_image_filename )

        return pix


    def createPixmap2(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):

            use_cache = imageCache != None

            if cacheId == None:
                cacheId = int( index )

            """if imageCache != None:
                imageCacheUsage = {}
                for path in imageCache.keys():
                    imageCacheUsage[ path ] = False"""

            if not tmp_image_filename:
                tmp_image_filename = self.TMP_IMAGE_FILENAME_TEMPLATE % \
                                     ( str( numpy.random.randint(sys.maxint) ), str( time.time() ) )

            imageFiles = self.adc.images[ index ].imageFiles

            if color:
                id_to_channel_map = { 0:'R', 1:'G', 2:'B' }

                for c in channelAdjustment.keys():
                    if not c in 'RGB':
                        i = len( id_to_channel_map )
                        id_to_channel_map[ i ] = c

            """channel_to_id_map = { 'R':0, 'G':1, 'B':2 }

            for c in channelAdjustment.keys():
                if not c in channel_to_id_map:
                    channel_to_id_map[c] = len(channel_to_id_map)"""

            img_dict = {}

            rect = None

            #xc = int( float( self.adc.objects[objId].position_x ) + 0.5 )
            #yc = int( float( self.adc.objects[objId].position_y ) + 0.5 )
            #rect =  (xc - width/2, yc - height/2, xc + width/2, yc + height/2)

            img_size = None

            any_rgb_adjustment_changed = False

            if use_cache:

                if not cacheId in imageCache:
                    imageCache[ cacheId ] = {}

                index_cached = ( 'files' in imageCache[ cacheId ] )

                if not index_cached:
                    imageCache[ cacheId ][ 'files' ] = {}

            else:
                index_cached = False

            for name,path in imageFiles:

                use_image = False

                for c,n in self.channelMapping.iteritems():
                    if n != None:
                        if name == n:
                            use_image = True
                            channel = c
                            break

                if use_image and ( channel in channelAdjustment ):

                    black,white,brightness,invert,binary = channelAdjustment[ channel ]
                    adjustment = ( left, top, width, height, color, black, white, brightness, invert,binary )

                    img = None

                    if use_cache:
                        img_cached = index_cached and ( name in imageCache[ cacheId ][ 'files' ] )
                    else:
                        img_cached = False

                    adjustment_changed = True

                    if not img_cached:

                        tmp = Image.open( path )

                        #img = TIFF.open( sys.argv[1] )
                        #pixel_arr = img.read_image()

                        if rect == None:
                            if width < 0:
                                width = tmp.size[ 0 ] - left
                            if height < 0:
                                height = tmp.size[ 1 ] - top
                            rect = ( left, top, left + width, top + height )
    
                        img_mode = tmp.mode
                        img = tmp.crop ( rect )
                        img_size = img.size
                        del tmp
                        arr = numpy.array(img.getdata())
                        del img
                        if img_mode == 'I;16':
                            #tmp_str = struct.pack('@%dh' % len(arr), *arr)
                            #arr = numpy.array(struct.unpack('@%dH' % len(arr), tmp_str))
                            #del tmp_str
                            #print '<3>'
                            #print arr
                            #if ( arr >= 2**15 ).any():
                            #    arr = arr - 2**15
                            if ( arr < 0 ).any() and ( arr <= 0 ).all():
                                arr = arr + 2**15
                                arr = arr * (2**8-1.0)/(2**12-1.0)
                            else:
                                mask = arr < 0
                                arr[ mask ] += 2**16
                                arr = arr * (2**8-1.0)/(2**16-1.0)

                        if use_cache:
                            imageCache[ cacheId ][ 'files' ][ name ] = {}
                            imageCache[ cacheId ][ 'files' ][ name ][ 'raw' ] = ( arr.copy(), img_size )
                            imageCache[ cacheId ][ 'files' ][ name ][ 'adjustment' ] = adjustment

                    else:

                        arr, img_size = imageCache[ cacheId ][ 'files' ][ name ][ 'raw' ]

                        oldAdjustment = imageCache[ cacheId ][ 'files' ][ name ][ 'adjustment' ]

                        if oldAdjustment == adjustment:

                            adjustment_changed = False

                        else:

                            #del imageCache[ cacheId ][ 'files' ][ name ][ 'processed' ]
                            imageCache[ cacheId ][ 'files' ][ name ][ 'adjustment' ] = adjustment

                    if adjustment_changed:

                        if channel in 'RGB':
                            any_rgb_adjustment_changed = True

                        mask = arr > white
                        arr[ mask ] = 255
                        mask = arr < black
                        arr[ mask ] = 0

                        if invert:
                            arr = 255 - arr
    
                        if binary:
                            mask = arr > 0
                            arr[ mask ] = 255
                            #arr[ numpy.invert( mask ) ] = 0

                        arr = arr * brightness

                        if brightness > 1.0:
                            mask = arr > 255
                            arr[ mask ] = 255

                        tmp_str = struct.pack('@%dB' % len(arr), *arr)
                        del arr
                        img = Image.fromstring('L',img_size,tmp_str,"raw","L",0,1)
                        del tmp_str

                        if use_cache:
                            imageCache[ cacheId ][ 'files' ][ name ][ 'processed' ] = img.copy()

                    else:

                        img = imageCache[ cacheId ][ 'files' ][ name ][ 'processed' ]

                    img_dict[ channel ] = img
                    del img


            #if rect == None:
            #    rect = ( 0, 0, width, height )

            if img_size == None:
                if width < 0 or height < 0:
                    width = 0
                    height = 0
                img_size = ( width, height )

            imgs = []

            #for path,used in imageCacheUsage.iteritems():
            #    if not used:
            #        del imageCache[ path ]

            if color:
                for i in xrange( len( id_to_channel_map ) ):
                    c = id_to_channel_map[ i ]
                    if not c in img_dict:
                        img_dict[ c ] = Image.new( 'L', img_size )
                    imgs.append( img_dict[ c ] )
            else:
                if len( img_dict ) > 0:
                    imgs.append( img_dict.values()[0] )
                else:
                    imgs.append( Image.new( 'L', im_size ) )

            """for i in xrange()
            for c,i in channel_to_id_map.iteritems():
                imgs[i] = img_dict[ c ]"""

            del img_dict
    
            """for i in xrange( len( imgs ) ):
                if not imgs[i]:
                    imgs[i] = Image.new( 'L', ( (rect[2]-rect[0]) , (rect[3]-rect[1]) ), 0)"""


            channels = channelAdjustment.keys()
            #channels.sort()

            rgb_channels_changed = True

            if index_cached and not any_rgb_adjustment_changed:

                oldChannels = imageCache[ cacheId ][ 'channels' ]
                bool_array = [ ( c in oldChannels ) == ( c in channels ) for c in 'RGB' ]
                if numpy.all( bool_array ):
                    rgb_channels_changed = False

            if use_cache:
                imageCache[ cacheId ][ 'channels' ] = channels

            if not rgb_channels_changed:

                merged_img = imageCache[ cacheId ][ 'merged_image' ]
                mode = imageCache[ cacheId ][ 'mode' ]

            else:

                if color:
                    mode = 'RGB'
                    merged_img = Image.merge( mode, imgs[:3] )
                else:
                    mode = 'L'
                    merged_img = imgs[0]

                if use_cache:
                    imageCache[ cacheId ][ 'merged_image' ] = merged_img.copy()
                    imageCache[ cacheId ][ 'mode' ] = mode

            img = merged_img

            if len(imgs) > 3:
                del imgs[:3]
                while len(imgs) > 1:
                    img1,img2 = imgs[0],imgs[1]
                    tmp = ImageChops.lighter( img1, img2 )
                    del imgs[0], img1, img2
                    imgs[0] = tmp
                imgs[0].save('/home/benjamin/tmp.tif')
                imgs[0] = imgs[0].convert( mode )
                imgs[0].save('/home/benjamin/tmp_rgb.tif')
                tmp = ImageChops.lighter( img, imgs[0])
                del img
                img = tmp
    
            del imgs[:]
            del imgs
    
            img.save( tmp_image_filename )
            del img
    
            #qtimg = QImage( tmp_image_filename )
            #pix = QPixmap.fromImage( qtimg )
    
            #del qtimg
    
            pix = QPixmap( tmp_image_filename )
   
            os.remove( tmp_image_filename )

            #imageCache[ int( index ) ][ 'pixmap' ] = pix

            return pix



class CellPixmapFactory(ImagePixmapFactory):

    def __init__(self, adc, channelMapping, objMask=None, ):
        ImagePixmapFactory.__init__( self, adc, channelMapping )
        if objMask != None:
            self.features = self.adc.objFeatures[ objMask ]
        else:
            self.features = self.adc.objFeatures[ : ]

    def createImageMask(self, index, channel, left, top, width, height, tmp_image_filename=None, imageCache=None, cacheId=None):

            objId = int( self.features[ index , self.adc.objObjectFeatureId ] )
            imgId = int( self.features[ index , self.adc.objImageFeatureId ] )

            if cacheId == None:
                cacheId = int( index )

            if left < 0 or top < 0:
                xc = int( float( self.adc.objects[objId].position_x ) + 0.5 )
                yc = int( float( self.adc.objects[objId].position_y ) + 0.5 )

                left = xc - width / 2
                top = yc - width / 2

            return ImagePixmapFactory.createImageMask( self, imgId, channel, left, top, width, height, tmp_image_filename, imageCache, cacheId )

    def createImage(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):

            objId = int( self.features[ index , self.adc.objObjectFeatureId ] )
            imgId = int( self.features[ index , self.adc.objImageFeatureId ] )

            if cacheId == None:
                cacheId = int( index )

            if left < 0 or top < 0:
                xc = int( float( self.adc.objects[objId].position_x ) + 0.5 )
                yc = int( float( self.adc.objects[objId].position_y ) + 0.5 )

                left = xc - width / 2
                top = yc - width / 2

            return ImagePixmapFactory.createImage( self, imgId, left, top, width, height, channelAdjustment, color, tmp_image_filename, imageCache, cacheId )


class CustomPixmapFactory(object):

    def __init__(self, images):
        self.images = images

    def createImage(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):
        return self.images[ index ]

    def createPixmap(self, index, left, top, width, height, channelAdjustment, color, tmp_image_filename=None, imageCache=None, cacheId=None):
        img = self.createImage( index, left, top, width, height, channelAdjustment, color, tmp_image_filename, imageCache, cacheId )

        img.save( tmp_image_filename )

        pix = QPixmap( tmp_image_filename )

        os.remove( tmp_image_filename )

        return pix

