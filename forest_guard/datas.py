import ee
from forest_guard.params import *

def maskL8sr(image):
    '''
    Cloud masking function from GEE
    input ee.image and return ee.image
    '''
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
    mask2 = image.mask().reduce('min')
    mask3 = image.select(OPTICAL_BANDS).gt(0).And(
            image.select(OPTICAL_BANDS).lt(10000)).reduce('min')
    mask = mask1.And(mask2).And(mask3)
    return image.select(OPTICAL_BANDS).divide(10000).addBands(
            image.select(THERMAL_BANDS).divide(10).clamp(273.15, 373.15)
            .subtract(273.15).divide(100)).updateMask(mask)

def get_image_landsat(start_date, end_date):  
    '''
    return landsat8 surface reflectance image
    from 'LANDSAT/LC08/C01/T1_SR' after Cloud masking
    input start_date = '2015-01-01', end_date='2017-12-31' and return ee.Image
    '''
    # Use Landsat 8 surface reflectance data.
    l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    # The image input data is a cloud-masked median composite.
    image = l8sr.filterDate('2015-01-01', '2017-12-31').map(maskL8sr).median()
    return image

def get_jaxa(start_date, end_date):
    '''
    return jaxa fnf
    from 'JAXA/ALOS/PALSAR/YEARLY/FNF'
    input start_date = '2017-01-01', end_date='2017-12-31' and return ee.Image
    '''
    jaxa = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF').filterDate(start_date, end_date).median()
    return jaxa

def get_stacked_arrays(feature, target, bands, response, kernel_size):
    '''
    stack feature and target
    select the bands and response
    return arrays of neighborhoodToArray with kernel_size
    Stack the 2D images (Landsat composite and NLCD impervious surface) to create a single image from which samples can be taken. 
    Convert the image into an array image in which each pixel stores 256x256 patches of pixels for each band. 
    This is a key step that bears emphasis: to export training patches, 
    convert a multi-band image to an array image using neighborhoodToArray(), then sample the image at points.
    '''
    featureStack = ee.Image.cat([
                                feature.select(bands),
                                target.select(response)
                                ]).float()

    list = ee.List.repeat(1, kernel_size)
    lists = ee.List.repeat(list, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)

    arrays = featureStack.neighborhoodToArray(kernel)
    return arrays


def export_to_cloud_sampling_train_eval(trainingPolys, evalPolys, arrays, n, N, folder, scale=30):
    '''
    trainingPolys, evalPolys : ee.FeatureCollection
    Use some pre-made geometries to sample the stack in strategic locations. 
    Specifically, these are hand-made polygons in which to take the kernelshape samples.
    
    arrays : ee.Image containing the features and target
    The mapped data look reasonable so take a sample from each polygon and merge the results into a single export. 
    The key step is sampling the array image at points, to get all the pixels in a kernelshape neighborhood at each point.
    It's worth noting that to build the training and testing data for the FCNN, you export a single TFRecord file that 
    contains patches of pixel values in each record. You do NOT need to export each training/testing patch to a different image.
    Since each record potentially contains a lot of data (especially with big patches or many input bands), 
    some manual sharding of the computation is necessary to avoid the computed value too large error. 
    Specifically, the following code takes multiple (smaller) samples within each geometry, merging the results to get a single export.
    
    n, N : 2 integers
    n = # Number of shards in each polygon. typically 200
    N =  # Total sample size in each polygon.typically 2000

    scale : integers
    Smallest number 30
    '''
        # Convert the feature collections to lists for iteration.
    trainingPolysList = trainingPolys.toList(trainingPolys.size())
    evalPolysList = evalPolys.toList(evalPolys.size())

    # Export all the training data (in many pieces), with one task 
    # per geometry.
    for g in range(trainingPolys.size().getInfo()):
        geomSample = ee.FeatureCollection([])
        for i in range(n):
            sample = arrays.sample(
            region = ee.Feature(trainingPolysList.get(g)).geometry(), 
            scale = scale,
            numPixels = N / n, # Size of the shard.
            seed = i,
            tileScale = 8
            )
            geomSample = geomSample.merge(sample)

        desc = TRAINING_BASE + '_g' + str(g)

        
        task = ee.batch.Export.table.toCloudStorage(
                                                    collection = geomSample,
                                                    description = desc,
                                                    bucket = BUCKET,
                                                    fileNamePrefix = folder + '/' + desc,
                                                    fileFormat = 'TFRecord',
                                                    selectors = BANDS + [RESPONSE], 
                                                    )
        task.start()  
    
    # Export all the evaluation data.
    for g in range(evalPolys.size().getInfo()):
        geomSample = ee.FeatureCollection([])
        for i in range(n):
            sample = arrays.sample(
                                region = ee.Feature(evalPolysList.get(g)).geometry(), 
                                scale = scale,
                                numPixels = N / n,
                                seed = i,
                                tileScale = 8
                                )
            geomSample = geomSample.merge(sample)

    desc = EVAL_BASE + '_g' + str(g)
    task = ee.batch.Export.table.toCloudStorage(
                                            collection = geomSample,
                                            description = desc,
                                            bucket = BUCKET,
                                            fileNamePrefix = folder + '/' + desc,
                                            fileFormat = 'TFRecord',
                                            selectors = BANDS + [RESPONSE]
                                            )
    task.start()
    
    print(task.status())
    # Monitor task progress
    # Code Extracted here:
    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/TF_demo1_keras.ipynb
    import time 
    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(5)
    print(task.status())
    return None