import sys, os, random

import numpy
import cStringIO
#import bz2
import Image
import ImageChops

class bz2:
	@staticmethod
	def compress(x):
		return x



class ImageComparer(object):

    def __init__(self):
        pass

    def concatenate_imgs(self, img1, img2, mode='side-by-side'):
        img_mode = img1.mode
        img_size = img1.size
        new_img_size = ( 2*img_size[0], 2*img_size[1] )
        img3 = Image.new( img_mode, new_img_size )
        img3.paste( img1, ( 0,0 ) )
        img3.paste( img2, ( img_size[0],0 ) )
        img3.paste( img2, ( 0,img_size[1] ) )
        img3.paste( img1, ( img_size[0],img_size[1] ) )
        return img3

    def compress_and_measure_img(self, img, format='JPEG', **kwargs):
        if format == 'JPEG' or 'PNG':
            sio = cStringIO.StringIO()
            img.save( sio, format, **kwargs)
            s = sio.getvalue()
            sio.close()
            s = bz2.compress( s )
            return float( len( s ) )
        elif format == 'pgm':
            sio = cStringIO.StringIO()
            arr = numpy.array( img.getdata() )
            for c in arr:
                sio.write('%d ') % c
            s = sio.getvalue()
            sio.close()
            s = bz2.compress( s )
            return float( len( s ) )

    def measure_optimized_img_mse(self, img1, img2):
        DIV=32
        a1 = numpy.array( img1.getdata() )
        a1 = a1.reshape( img1.size )
        a2 = numpy.array( img2.getdata() )
        a2 = a2.reshape( img2.size )
        width,height = img1.size
        mse = numpy.zeros( ( ( width / DIV ) * 2 + 1, ( height / DIV) * 2 + 1 ) )
        mse[ : ] = -1
        for i in xrange( mse.shape[0] ):
            x = i - width/DIV
            if x < 0:
                tmp_a1 = a1[ -x: , : ]
                tmp_a2 = a2[  :x , : ]
            elif x > 0:
                tmp_a1 = a1[  x: , : ]
                tmp_a2 = a2[ :-x , : ]
            else:
                tmp_a1 = a1
                tmp_a2 = a2
            for j in xrange( mse.shape[1] ):
                y = j - height/DIV
                if y < 0:
                    tmp2_a1 = tmp_a1[ : , -y: ]
                    tmp2_a2 = tmp_a2[ : ,  :y ]
                elif y > 0:
                    tmp2_a1 = tmp_a1[ : ,  y: ]
                    tmp2_a2 = tmp_a2[ : , :-y ]
                else:
                    tmp2_a1 = tmp_a1
                    tmp2_a2 = tmp_a2
                mse[ i, j ] = self.measure_array_mse( tmp2_a1, tmp2_a2 )
        mse = mse.reshape( ( mse.shape[0] * mse.shape[1], ) )
        min_i = numpy.argmin( mse )
        min_ix = min_i % ( width/DIV )
        min_iy = ( min_i - min_ix ) / ( width/DIV )
        min_x = min_ix - width/DIV
        min_y = min_iy - height/DIV
        return mse[ min_i ], min_x, min_y

    def measure_img_mse(self, img1, img2):
        a1 = numpy.array( img1.getdata() )
        a2 = numpy.array( img2.getdata() )
        return self.measure_array_mse( a1, a2 )

    def measure_array_mse(self, arr1, arr2):
        length = 1
        for d in xrange( len( arr1.shape ) ):
            length *= arr1.shape[ d ]
        a1 = arr1.reshape( ( length, ) )
        a2 = arr2.reshape( ( length, ) )
        MSE = numpy.sum( ( a1 - a2 ) ** 2 )
        m = numpy.sum( numpy.logical_or( a1 > 0, a2 > 0 ) )
        MSE = MSE / float( m )
        return MSE

    def comp_imgs(self, img1, imgs, format='JPEG', **kwargs):

        if 'method' not in kwargs:
            kwargs[ 'method' ] = 'MSE'
        method = kwargs[ 'method' ]

        if method == 'NCD':

            C1 = self.compress_and_measure_img( img1, format, **kwargs )
            C2s = []
            for img2 in imgs:
                C2s.append( self.compress_and_measure_img( img2, format, **kwargs) )
            C2s = numpy.array( C2s )
            C12s = []
            for img2 in imgs:
                img3 = self.concatenate_imgs( img1, img2 )
                C12 = self.compress_and_measure_img( img3, format, **kwargs )
                img4 = self.concatenate_imgs( img2, img1 )
                C12 += self.compress_and_measure_img( img4, format, **kwargs )
                C12 /= 4.0
                C12s.append( C12 )
            C12s = numpy.array( C12s )
    
            NCDs = []
            for i in xrange( len( C2s ) ):
                NCD_numerator = C12s[ i ] - min( C1, C2s[ i ] )
                NCD_denominator = max( C1, C2s[ i ] )
                NCDs.append( NCD_numerator / NCD_denominator )
    
            return numpy.array( NCDs )

        elif method == 'MSE':

            results = []
            for img2 in imgs:
                results.append( self.measure_img_mse( img1, img2 ) )

            return numpy.array( results )

        elif method == 'MSE_optimized':

            results = []
            shifts = []
            for img2 in imgs:
                mse,min_x,min_y = self.measure_optimized_img_mse( img1, img2 )
                results.append( mse )
                shifts.append( ( min_x, min_y ) )

            return numpy.array( results ), shifts

