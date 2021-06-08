'''
Module allowing to visualise prediction in folium
Assumptions :
- model is eefied 
- model is put on ai google platform

'''
import folium
import ee
from forest_guard.params import BANDS, OPTICAL_BANDS, THERMAL_BANDS, PROJECT



def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
      tiles=map_id_dict['tile_fetcher'].url_format,
      attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
      name=name,
      overlay=True,
      control=True,
      show=False,
      opacity=0.7
  ).add_to(self)


# Cloud masking function.
def maskL8sr(image):
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


folium.Map.add_ee_layer = add_ee_layer

def map_pred(model_name, model_version, model_region, year1=2017, year2=2020, center_point=[48.47, 6.92], zoom_start=13):
    '''
    plot on follium landsat on year1 and year2
    '''
    threshold = 0.80
    #max year is 2020
    if year2>2020:
        year2=2020
    if year1>2020:
        year1=2020

    #min year is 2013
    if year2<2013:
        year2=2013
    if year1<2013:
        year1=2013

    # Use Landsat 8 surface reflectance data.
    l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    # The image input data is a cloud-masked median composite.
    image20 = l8sr.filterDate(
        f'{year2}-01-01', f'{year2}-12-31').map(maskL8sr).median().select(BANDS).float()
    image17 = l8sr.filterDate(
        f'{year1}-01-01', f'{year1}-12-31').map(maskL8sr).median().select(BANDS).float()


    # Load the trained model and use it for prediction.  If you specified a region 
    # other than the default (us-central1) at model creation, specify it here.
    model = ee.Model.fromAiPlatformPredictor(
        projectId = PROJECT,
        modelName = model_name,
        version = model_version,
        region= model_region,
        inputTileSize = [144, 144],
        inputOverlapSize = [8, 8],
        proj = ee.Projection('EPSG:4326').atScale(30),
        fixInputProj = True,
        outputBands = {'fnf': {
            'type': ee.PixelType.float()
        }
        }
    )



    # Use folium to visualize the input imagery and the predictions.
    map = folium.Map(location=center_point, zoom_start=zoom_start)



    ## landsat
    mapid = image17.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3})
    folium.TileLayer(
        tiles=mapid['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        overlay=True,
        name=f'landsat{year1}',
        show=False
    ).add_to(map)

    mapid = image20.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3})
    folium.TileLayer(
        tiles=mapid['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        overlay=True,
        name=f'landsat{year2}',
        show=False
    ).add_to(map)

    ### Predictions
    palette = [ '#FEFF99',
            '006400',
            ]
    #Pred year1
    predictions17 = model.predictImage(image17.toArray())
    predictions17=predictions17.where(predictions17.gt(threshold), 1)
    predictions17=predictions17.where(predictions17.lt(threshold), 0)

    mask_pred17 = predictions17.updateMask(predictions17.gte(threshold))

    map.add_ee_layer(mask_pred17,
                    {'bands': ['fnf'],
                        'min': 1, 
                        'max': 1, 
                        'palette':palette
                        }, f'predictions_{year1}'
                    )

    #PRED year2
    predictions20 = model.predictImage(image20.toArray())
    predictions20=predictions20.where(predictions20.gt(threshold), 1)
    predictions20=predictions20.where(predictions20.lt(threshold), 0)

    mask_pred20 = predictions20.updateMask(predictions20.gte(threshold))

    map.add_ee_layer(mask_pred20,
                    {'bands': ['fnf'],
                        'min': 1, 
                        'max': 1, 
                        'palette':palette
                        }, f'predictions_{year2}'
                    )
    #Diff of prediction
    palette_diff = [ 'red','blue']


    diff = predictions20.subtract(predictions17)
    d_map = predictions20.multiply(0)
    d_map = d_map.where(diff.gt(0.6), 2)      # All pos diffs are now labeled 2.
    d_map = d_map.where(diff.lt(-0.6), 1)      # Label all neg to 1.
    mask_d_map = d_map.updateMask(d_map.gte(0.5)) #mask the zeros


    map.add_ee_layer(mask_d_map,
                    {'bands': ['fnf'],
                        'min': 1, 
                        'max': 2, 
                        'palette':palette_diff
                        }, f'predictions_diff_{year2}_{year1}'
                    )

    ###JAXA
    from forest_guard.datas import get_jaxa
    jaxa = get_jaxa('2017-01-01', '2017-12-31')

    palette = ['006400',
            'FEFF99',
                '0000FF'
            ]
    mask_pred_jaxa = jaxa.updateMask(jaxa.lte(1.5))

    map.add_ee_layer(mask_pred_jaxa,
                    {'bands': ['fnf'],
                        'min': 1, 
                        'max': 3, 
                        'palette':palette
                        }, 'jaxa_masked'
                    )

    #### SHOW MAP
    map.add_child(folium.LayerControl())


    return map