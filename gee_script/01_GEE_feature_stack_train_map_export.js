// CLEAN PAPER-1 WORKFLOW FILE
// Run first in Google Earth Engine. Export all CSV tables and the classified asset before running Python.
// =============================================================================
// Dak Lak Coffee Mapping 2024, CLEAN WORKFLOW STEP 01: feature stack, RF training, map/table export
// Sentinel-2 + Landsat 8/9 + Sentinel-1 + DEM
// RF feature ranking -> Pearson correlation filter -> selected Top 25 -> RF map
// Includes: validation predictions, raw + row-normalized confusion matrix,
// Table 4 class-wise PA/UA/F1, area statistics, district coffee area,
// optional probability/uncertainty layers. FULL97 candidate features preserved.
// =============================================================================

// =============================================================================
// 1) SETTINGS
// =============================================================================
var region = ee.FeatureCollection('users/ntduc11/daklak');
var trainingTable = ee.FeatureCollection('users/ntduc11/ROI1');  // 3000 points expected
var aoi = region.geometry();
Map.centerObject(region, 8);

var SCALE = 10;
var SEED = 2024;
var EXPORT_FOLDER = 'GEE_Exports_R3000';
var EXPORT_CLASSIFIED_ASSET = true;
var CLASSIFIED_ASSET_ID = 'users/ntduc11/DakLak_2024_10class_RF_corrTop25';  // change if asset already exists

// Export switches
var EXPORT_MAPS = true;
var EXPORT_TABLES = true;
var EXPORT_PREDICTOR_STACK = false;   // heavy; set true only if needed
var EXPORT_AREA_STATS = false;      // clean workflow: run Step 02 area script after asset export
var EXPORT_DISTRICT_AREA = false;  // clean workflow: run Step 02 area script after asset export
var EXPORT_PROBABILITY = false;       // heavy; run after final map is confirmed

// District boundary for area validation
// Recommended: use your own official district asset.
var USE_CUSTOM_DISTRICTS = true;
var DISTRICT_ASSET = 'users/ntduc11/daklak_districts';
var DISTRICT_NAME_PROP = 'District'; // change to 'District' or actual field if needed

// Reference geometry mode.
// true  = rebuild point geometry from lon/lat columns.
// false = use existing point geometry in the ROI asset.
// If your ROI table has lon and lat columns, keep true. If not, set false.
var USE_LON_LAT_COLUMNS = true;

// Split mode: 'use_split_field' | 'random' | 'spatial_blocks'
var SPLIT_MODE = 'use_split_field';
var BLOCK_SIZE_M = 1000;

// Missing pixels after cloud masking/compositing.
// TRUE prevents losing validation samples in sampleRegions. Check histograms after sampling.
var FILL_MISSING_PIXELS = true;
var FILL_VALUE = 0;

// Feature selection
var USE_CORR_FILTER = true;
var CORR_THRESH = 0.90;
var MAX_FEATURES = 25;
var CORR_SAMPLE_SIZE = 1500;
var RFRANK_TREES = 250;

// RF config: keep consistent with manuscript
var RF_TREES_FINAL = 2000;
var RF_VARS_PER_SPLIT = 3;
var RF_BAG_FRACTION = 0.65;
var RF_MIN_LEAF = 1;

// Time windows
var dryStart = ee.Date('2023-11-01');
var dryEnd   = ee.Date('2024-04-30');
var wetStart = ee.Date('2024-05-01');
var wetEnd   = ee.Date('2024-10-31');
var yearStart = ee.Date('2024-01-01');
var yearEnd   = ee.Date('2024-12-31');

// =============================================================================
// 2) CLASS TABLE
// =============================================================================
var classTable = [
  {id:1,  name:'Sun coffee',           color:'#8c3b00'},
  {id:2,  name:'Intercrop coffee',     color:'#ff8080'},
  {id:3,  name:'Newly planted coffee', color:'#ffb000'},
  {id:4,  name:'Rubber',               color:'#4caf82'},
  {id:5,  name:'Partially vegetative', color:'#d4c86a'},
  {id:6,  name:'Rice',                 color:'#0087a8'},
  {id:7,  name:'Other upland crops',   color:'#f2e6b8'},
  {id:8,  name:'Forest',               color:'#1f4d2b'},
  {id:9,  name:'Water',                color:'#00c8ff'},
  {id:10, name:'Built',                color:'#9a9a9a'}
];

var classIds = ee.List(classTable.map(function(d){ return d.id; }));
var classNames = ee.List(classTable.map(function(d){ return d.name; }));
var palette = classTable.map(function(d){ return d.color; });

// =============================================================================
// 3) GENERAL HELPERS
// =============================================================================
function safeZeroImage(bands){
  return ee.Image.constant(ee.List.repeat(0, bands.length)).rename(bands);
}

function renameWithPrefix(img, prefix){
  var oldNames = img.bandNames();
  var newNames = oldNames.map(function(b){
    return ee.String(prefix).cat('_').cat(ee.String(b));
  });
  return img.rename(newNames);
}

function featureHasProperty(f, prop){
  return ee.Feature(f).propertyNames().contains(prop);
}

// Robust reference-point preparation.
// Expected columns: class_id; optional lon, lat; optional split.
// If lon/lat are present, geometry is rebuilt from them. Otherwise original geometry is retained.
function prepareReferencePoints(fc){
  return fc.map(function(f){
    f = ee.Feature(f);

    // IMPORTANT:
    // Use a client-side switch instead of server-side boolean chaining.
    // This avoids GEE errors from server-side Boolean methods.
    var geom;
    if (USE_LON_LAT_COLUMNS) {
      geom = ee.Geometry.Point([
        ee.Number.parse(ee.String(f.get('lon'))),
        ee.Number.parse(ee.String(f.get('lat')))
      ]);
    } else {
      geom = f.geometry();
    }

    var cid = ee.Number.parse(ee.String(f.get('class_id')));
    var split = ee.String(ee.Algorithms.If(featureHasProperty(f, 'split'), f.get('split'), ''));

    return ee.Feature(geom, f.toDictionary()).set({class_id: cid, split: split});
  })
  .filterBounds(aoi)
  .filter(ee.Filter.inList('class_id', classIds));
}

function randomSplit(fc, seed){
  var withRnd = fc.randomColumn('rnd', seed);
  return {
    train: withRnd.filter(ee.Filter.lt('rnd', 0.7)),
    val:   withRnd.filter(ee.Filter.gte('rnd', 0.7))
  };
}

function spatialBlockSplit(fc, seed, cellSizeMeters){
  var proj = ee.Projection('EPSG:32649').atScale(cellSizeMeters);
  var coords = ee.Image.pixelCoordinates(proj);
  var gx = coords.select('x').divide(cellSizeMeters).floor();
  var gy = coords.select('y').divide(cellSizeMeters).floor();
  var gridIdImg = gx.multiply(1000000).add(gy).rename('gridId');

  var withGrid = fc.map(function(f){
    var s = gridIdImg.sample({
      region: f.geometry(),
      scale: cellSizeMeters,
      numPixels: 1,
      geometries: false
    }).first();
    var gid = ee.Number(ee.Feature(s).get('gridId'));
    return f.set('gridId', gid);
  });

  var grids = ee.List(withGrid.aggregate_array('gridId')).distinct();
  var gridsRnd = ee.FeatureCollection(grids.map(function(g){
    g = ee.Number(g);
    var r = ee.Number(g.multiply(1103515245).add(seed).mod(2147483647)).divide(2147483647);
    return ee.Feature(null, {gridId: g, r: r});
  }));

  var valGrids = ee.List(gridsRnd.filter(ee.Filter.gte('r', 0.7)).aggregate_array('gridId'));

  return {
    train: withGrid.filter(ee.Filter.inList('gridId', valGrids).not()),
    val:   withGrid.filter(ee.Filter.inList('gridId', valGrids))
  };
}

function pearsonCorr(fc, b1, b2){
  var d = fc.reduceColumns(ee.Reducer.pearsonsCorrelation(), [b1, b2]);
  var r = d.get('correlation');
  return ee.Number(ee.Algorithms.If(r, r, 0));
}

function greedyCorrFilter(fc, candidates, corrThresh, maxKeep){
  candidates = ee.List(candidates);
  corrThresh = ee.Number(corrThresh);
  maxKeep = ee.Number(maxKeep);
  var init = ee.Dictionary({keep: ee.List([])});

  var out = ee.Dictionary(candidates.iterate(function(b, state){
    state = ee.Dictionary(state);
    var keep = ee.List(state.get('keep'));
    b = ee.String(b);
    var alreadyFull = keep.size().gte(maxKeep);

    return ee.Dictionary(ee.Algorithms.If(alreadyFull, state, (function(){
      var isFirst = keep.size().eq(0);
      var maxAbs = ee.Number(ee.Algorithms.If(isFirst, 0, (function(){
        var corrs = keep.map(function(k){
          k = ee.String(k);
          return pearsonCorr(fc, k, b).abs();
        });
        return ee.Number(ee.List(corrs).reduce(ee.Reducer.max()));
      })()));
      var ok = maxAbs.lt(corrThresh);
      var keep2 = ee.List(ee.Algorithms.If(ok, keep.add(b), keep));
      return ee.Dictionary({keep: keep2});
    })()));
  }, init));

  return ee.List(out.get('keep'));
}

function confusionMatrixToLong(cmObj, classIdsList){
  var cmArr = ee.Array(cmObj.array());
  return ee.FeatureCollection(classIdsList.map(function(tcid){
    tcid = ee.Number(tcid);
    var ti = classIdsList.indexOf(tcid);
    return classIdsList.map(function(pcid){
      pcid = ee.Number(pcid);
      var pi = classIdsList.indexOf(pcid);
      return ee.Feature(null, {
        true_class: tcid,
        pred_class: pcid,
        count: cmArr.get([ti, pi])
      });
    });
  }).flatten());
}

function perClassMetrics(cmObj, classIdsList){
  // GEE returns arrays for consumers/producers accuracy.
  var uaList = ee.Array(cmObj.consumersAccuracy()).toList().flatten(); // user's accuracy / precision
  var paList = ee.Array(cmObj.producersAccuracy()).toList().flatten(); // producer's accuracy / recall

  return ee.FeatureCollection(classIdsList.map(function(cid){
    cid = ee.Number(cid);
    var idx = classIdsList.indexOf(cid);
    var precision = ee.Number(uaList.get(idx));
    var recall = ee.Number(paList.get(idx));
    var f1 = ee.Algorithms.If(
      precision.add(recall).gt(0),
      precision.multiply(recall).multiply(2).divide(precision.add(recall)),
      0
    );
    return ee.Feature(null, {
      class_id: cid,
      users_accuracy_precision: precision,
      producers_accuracy_recall: recall,
      f1: ee.Number(f1)
    });
  }));
}

// Final Table 4: class-wise accuracy with class names and percent columns.
function perClassMetricsTable4(cmObj, classIdsList, classNamesList){
  var uaList = ee.Array(cmObj.consumersAccuracy()).toList().flatten(); // User's accuracy / precision
  var paList = ee.Array(cmObj.producersAccuracy()).toList().flatten(); // Producer's accuracy / recall

  return ee.FeatureCollection(classIdsList.map(function(cid){
    cid = ee.Number(cid);
    var idx = classIdsList.indexOf(cid);

    var ua = ee.Number(uaList.get(idx));
    var pa = ee.Number(paList.get(idx));

    var f1 = ee.Number(ee.Algorithms.If(
      ua.add(pa).gt(0),
      ua.multiply(pa).multiply(2).divide(ua.add(pa)),
      0
    ));

    return ee.Feature(null, {
      class_id: cid,
      class_name: classNamesList.get(idx),
      users_accuracy: ua,
      producers_accuracy: pa,
      f1_score: f1,
      users_accuracy_percent: ua.multiply(100),
      producers_accuracy_percent: pa.multiply(100),
      f1_percent: f1.multiply(100)
    });
  }));
}

// Row-normalized confusion matrix in long format for Figure 4.
function confusionMatrixRowNormToLong(cmObj, classIdsList, classNamesList){
  var cmArr = ee.Array(cmObj.array());

  return ee.FeatureCollection(
    classIdsList.map(function(tcid){
      tcid = ee.Number(tcid);
      var ti = classIdsList.indexOf(tcid);

      var row = cmArr.slice(0, ti, ee.Number(ti).add(1));
      var rowSum = ee.Number(row.reduce('sum', [1]).get([0]));

      return classIdsList.map(function(pcid){
        pcid = ee.Number(pcid);
        var pi = classIdsList.indexOf(pcid);

        var count = ee.Number(cmArr.get([ti, pi]));
        var proportion = ee.Number(ee.Algorithms.If(
          rowSum.gt(0),
          count.divide(rowSum),
          0
        ));

        return ee.Feature(null, {
          true_class_id: tcid,
          true_class_name: classNamesList.get(ti),
          pred_class_id: pcid,
          pred_class_name: classNamesList.get(pi),
          count: count,
          row_total: rowSum,
          row_normalized_proportion: proportion,
          row_normalized_percent: proportion.multiply(100)
        });
      });
    }).flatten()
  );
}


function overallMetrics(cmObj, nTrain, nVal, nFeatures, modelLabel){
  return ee.FeatureCollection([ee.Feature(null, {
    model: modelLabel,
    overall_accuracy: cmObj.accuracy(),
    kappa: cmObj.kappa(),
    n_train: nTrain,
    n_val: nVal,
    n_features: nFeatures,
    corr_thresh: CORR_THRESH,
    max_features: MAX_FEATURES,
    rf_trees: RF_TREES_FINAL,
    rf_variables_per_split: RF_VARS_PER_SPLIT,
    rf_bag_fraction: RF_BAG_FRACTION,
    rf_min_leaf: RF_MIN_LEAF,
    seed: SEED
  })]);
}

function trainRF(sampleFc, classProp, inputBands, nTrees, seed){
  return ee.Classifier.smileRandomForest({
    numberOfTrees: nTrees,
    variablesPerSplit: RF_VARS_PER_SPLIT,
    seed: seed,
    bagFraction: RF_BAG_FRACTION,
    minLeafPopulation: RF_MIN_LEAF
  }).train({
    features: sampleFc,
    classProperty: classProp,
    inputProperties: inputBands
  });
}

// =============================================================================
// 4) PREPARE TRAINING AND VALIDATION POINTS
// =============================================================================
var training = prepareReferencePoints(trainingTable);
print('Reference first:', training.first());
print('Reference count:', training.size());
print('Reference class histogram:', training.aggregate_histogram('class_id'));
print('Reference split histogram:', training.aggregate_histogram('split'));

var splitHist = ee.Dictionary(training.aggregate_histogram('split'));
var hasTrain = ee.Number(splitHist.get('train', 0)).gt(0);
var hasVal   = ee.Number(splitHist.get('val', 0)).gt(0);
var hasBoth  = hasTrain.and(hasVal);
var useExistingSplit = ee.Algorithms.If(SPLIT_MODE === 'use_split_field', hasBoth, false);

var splitObj = ee.Dictionary(ee.Algorithms.If(
  useExistingSplit,
  ee.Dictionary({
    train: training.filter(ee.Filter.eq('split', 'train')),
    val:   training.filter(ee.Filter.eq('split', 'val'))
  }),
  ee.Algorithms.If(
    SPLIT_MODE === 'spatial_blocks',
    ee.Dictionary(spatialBlockSplit(training, SEED, BLOCK_SIZE_M)),
    ee.Dictionary(randomSplit(training, SEED))
  )
));

var trainPts = ee.FeatureCollection(splitObj.get('train'));
var valPts   = ee.FeatureCollection(splitObj.get('val'));

print('Train point count:', trainPts.size());
print('Validation point count:', valPts.size());
print('Train class histogram:', trainPts.aggregate_histogram('class_id'));
print('Validation class histogram:', valPts.aggregate_histogram('class_id'));

Map.addLayer(trainPts, {color: 'red'}, 'Train points', false);
Map.addLayer(valPts, {color: 'blue'}, 'Validation points', false);

// =============================================================================
// 5) SENTINEL-2 PREPROCESSING + FEATURES
// =============================================================================
function maskS2sr(img){
  var qa = img.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var qaMask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  var scl = img.select('SCL');
  var sclMask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))         // cloud medium probability
    .and(scl.neq(9))         // cloud high probability
    .and(scl.neq(10))        // thin cirrus
    .and(scl.neq(11));       // snow/ice

  return img.updateMask(qaMask).updateMask(sclMask).divide(10000);
}

function addS2Indices(img){
  var blue  = img.select('B2');
  var green = img.select('B3');
  var red   = img.select('B4');
  var re1   = img.select('B5');
  var re2   = img.select('B6');
  var nir   = img.select('B8');
  var sw1   = img.select('B11');
  var sw2   = img.select('B12');

  var ndvi  = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI');
  var rdvi  = nir.subtract(red).divide(nir.add(red).sqrt()).rename('RDVI');
  var nli   = nir.pow(2).subtract(red).divide(nir.pow(2).add(red).max(1e-6)).rename('NLI');
  var sr    = nir.divide(red.max(1e-6)).rename('SR');
  var msr   = sr.subtract(1).divide(sr.add(1).sqrt()).rename('MSR');
  var cvi   = nir.multiply(red).divide(green.pow(2).max(1e-6)).rename('CVI');
  var ndi   = green.subtract(red).divide(green.add(red)).rename('NDI');

  var ndvi_re = nir.subtract(re1).divide(nir.add(re1)).rename('NDVI_RE');
  var psri    = red.subtract(blue).divide(re2.max(1e-6)).rename('PSRI');
  var ci_re   = nir.divide(re1.max(1e-6)).subtract(1).rename('CI_RE');
  var mtci    = re2.subtract(re1).divide(re1.subtract(red).max(1e-6)).rename('MTCI');

  var ndmi  = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDMI');
  var ndwi  = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var mndwi = green.subtract(sw1).divide(green.add(sw1)).rename('MNDWI');
  var nbr   = nir.subtract(sw2).divide(nir.add(sw2)).rename('NBR');
  var nbr2  = sw1.subtract(sw2).divide(sw1.add(sw2)).rename('NBR2');

  return img.addBands([
    ndvi, gndvi, rdvi, nli, sr, msr, cvi, ndi,
    ndvi_re, psri, ci_re, mtci,
    ndmi, ndwi, mndwi, nbr, nbr2
  ]);
}

function s2Composite(start, end, tag){
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(start, end)
    .map(maskS2sr)
    .map(addS2Indices);

  var bands = [
    'B2','B3','B4','B5','B6','B7','B8','B11','B12',
    'NDVI','GNDVI','RDVI','NLI','SR','MSR','CVI','NDI',
    'NDVI_RE','PSRI','CI_RE','MTCI',
    'NDMI','NDWI','MNDWI','NBR','NBR2'
  ];

  var img = ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.select(bands).median(),
    safeZeroImage(bands)
  ));

  return renameWithPrefix(img, 'S2_' + tag);
}

// =============================================================================
// 6) LANDSAT 8/9 PREPROCESSING + FEATURES
// =============================================================================
function maskL89sr(img){
  var qa = img.select('QA_PIXEL');
  var fill    = qa.bitwiseAnd(1 << 0).neq(0);
  var dilated = qa.bitwiseAnd(1 << 1).neq(0);
  var cloud   = qa.bitwiseAnd(1 << 3).neq(0);
  var shadow  = qa.bitwiseAnd(1 << 4).neq(0);
  var snow    = qa.bitwiseAnd(1 << 5).neq(0);
  var cirrus  = qa.bitwiseAnd(1 << 7).neq(0);
  var mask = fill.or(dilated).or(cloud).or(shadow).or(snow).or(cirrus).not();

  var refl = img.select(
    ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
    ['Blue','Green','Red','NIR','SWIR1','SWIR2']
  ).multiply(0.0000275).add(-0.2);

  return refl.updateMask(mask);
}

function addL89Indices(img){
  var blue = img.select('Blue');
  var red = img.select('Red');
  var nir = img.select('NIR');
  var sw1 = img.select('SWIR1');
  var sw2 = img.select('SWIR2');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndmi = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDMI');
  var nbr  = nir.subtract(sw2).divide(nir.add(sw2)).rename('NBR');
  var nbr2 = sw1.subtract(sw2).divide(sw1.add(sw2)).rename('NBR2');
  var msi  = sw1.divide(nir.max(1e-6)).rename('MSI');
  var bsi  = sw1.add(red).subtract(nir.add(blue))
    .divide(sw1.add(red).add(nir).add(blue).max(1e-6)).rename('BSI');

  return img.addBands([ndvi, ndmi, nbr, nbr2, msi, bsi]);
}

function l89Composite(start, end, tag){
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi).filterDate(start, end);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(aoi).filterDate(start, end);
  var col = l8.merge(l9).map(maskL89sr).map(addL89Indices);

  var baseBands = ['Blue','Green','Red','NIR','SWIR1','SWIR2','NDVI','NDMI','NBR','NBR2','MSI','BSI'];
  var med = ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.select(baseBands).median(),
    safeZeroImage(baseBands)
  ));

  var swir2p75 = ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.select('SWIR2').reduce(ee.Reducer.percentile([75])).rename('SWIR2_p75'),
    ee.Image.constant(0).rename('SWIR2_p75')
  ));

  return renameWithPrefix(med.addBands(swir2p75), 'L89_' + tag);
}

// =============================================================================
// 7) SENTINEL-1 METRICS + TEXTURE
// =============================================================================
function getS1Collection(start, end){
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV','VH']);
}

function s1Stack(start, end){
  var s1 = getS1Collection(start, end);
  var desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
  var asc  = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));

  var vv_d_med = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VV').median(), ee.Image.constant(0))).rename('VVdesc_p50');
  var vh_d_med = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VH').median(), ee.Image.constant(0))).rename('VHdesc_p50');
  var vv_d_p25 = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VV').reduce(ee.Reducer.percentile([25])), ee.Image.constant(0))).rename('VVdesc_p25');
  var vv_d_p75 = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VV').reduce(ee.Reducer.percentile([75])), ee.Image.constant(0))).rename('VVdesc_p75');
  var vh_d_p25 = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VH').reduce(ee.Reducer.percentile([25])), ee.Image.constant(0))).rename('VHdesc_p25');
  var vh_d_p75 = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VH').reduce(ee.Reducer.percentile([75])), ee.Image.constant(0))).rename('VHdesc_p75');
  var vv_d_std = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VV').reduce(ee.Reducer.stdDev()), ee.Image.constant(0))).rename('VVdesc_std');
  var vh_d_std = ee.Image(ee.Algorithms.If(desc.size().gt(0), desc.select('VH').reduce(ee.Reducer.stdDev()), ee.Image.constant(0))).rename('VHdesc_std');
  var diff_d = vv_d_med.subtract(vh_d_med).rename('VVminusVH_desc_p50');

  var vv_a_med = ee.Image(ee.Algorithms.If(asc.size().gt(0), asc.select('VV').median(), ee.Image.constant(0))).rename('VVasc_p50');
  var vh_a_med = ee.Image(ee.Algorithms.If(asc.size().gt(0), asc.select('VH').median(), ee.Image.constant(0))).rename('VHasc_p50');
  var diff_a = vv_a_med.subtract(vh_a_med).rename('VVminusVH_asc_p50');

  // Texture base: clamp/scale VH dB before GLCM.
  var vhTexBase = vh_d_med.clamp(-25, 0).add(25).multiply(10).toByte().rename('VH');
  var glcm = vhTexBase.glcmTexture({size: 5});
  var tex = glcm.select(
    ['VH_contrast','VH_asm','VH_ent'],
    ['VH_glcm_contrast_5x5','VH_glcm_asm_5x5','VH_glcm_ent_5x5']
  );

  var out = ee.Image.cat([
    vv_d_med, vh_d_med,
    vv_d_p25, vv_d_p75,
    vh_d_p25, vh_d_p75,
    vv_d_std, vh_d_std,
    diff_d,
    vv_a_med, vh_a_med, diff_a,
    tex
  ]);

  return renameWithPrefix(out, 'S1');
}

// =============================================================================
// 8) DEM FEATURES
// =============================================================================
function demStack(){
  // Keep DEM as 4 predictors to preserve the FULL97 feature design used in the manuscript:
  // DEM_elevation, DEM_slope, DEM_aspect, DEM_hillshade.
  var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation');
  var terrain = ee.Terrain.products(dem);
  var hillshade = ee.Terrain.hillshade(dem).rename('hillshade');

  var out = dem.addBands([
    terrain.select('slope').rename('slope'),
    terrain.select('aspect').rename('aspect'),
    hillshade
  ]);

  return renameWithPrefix(out, 'DEM');
}

// =============================================================================
// 9) BUILD FULL FEATURE SPACE
// =============================================================================
var s2Dry = s2Composite(dryStart, dryEnd, 'dry');
var s2Wet = s2Composite(wetStart, wetEnd, 'wet');
var l89Dry = l89Composite(dryStart, dryEnd, 'dry');
var l89Wet = l89Composite(wetStart, wetEnd, 'wet');
var s1 = s1Stack(yearStart, yearEnd);
var dem = demStack();

var featureStackRaw = ee.Image.cat([s2Dry, s2Wet, l89Dry, l89Wet, s1, dem]).clip(aoi).float();
var featureStackAll = ee.Image(ee.Algorithms.If(
  FILL_MISSING_PIXELS,
  featureStackRaw.unmask(FILL_VALUE),
  featureStackRaw
)).clip(aoi).float();

var allBands = featureStackAll.bandNames();
print('All candidate bands:', allBands);
print('Number of candidate features:', allBands.size());

Map.addLayer(s2Dry.select(['S2_dry_B8','S2_dry_B4','S2_dry_B3']), {min: 0.03, max: 0.40}, 'S2 dry false color', false);
Map.addLayer(dem.select('DEM_hillshade'), {min: 0, max: 255}, 'DEM hillshade', false);

// =============================================================================
// 10) SAMPLE TRAIN/VALIDATION ON FULL STACK
// =============================================================================
var trainSampAll = featureStackAll.sampleRegions({
  collection: trainPts,
  properties: ['class_id', 'row_id'],
  scale: SCALE,
  tileScale: 4,
  geometries: true
});

var valSampAll = featureStackAll.sampleRegions({
  collection: valPts,
  properties: ['class_id', 'row_id'],
  scale: SCALE,
  tileScale: 4,
  geometries: true
});

print('Sampled train size FULL:', trainSampAll.size());
print('Sampled validation size FULL:', valSampAll.size());
print('Sampled train class histogram FULL:', trainSampAll.aggregate_histogram('class_id'));
print('Sampled validation class histogram FULL:', valSampAll.aggregate_histogram('class_id'));

// =============================================================================
// 11) FEATURE SELECTION: RF ranking -> correlation filter -> Top 25
// =============================================================================
var trainForCorr = trainSampAll.randomColumn('corr_rnd', SEED).sort('corr_rnd').limit(CORR_SAMPLE_SIZE);

var rfRank = ee.Classifier.smileRandomForest({
  numberOfTrees: RFRANK_TREES,
  variablesPerSplit: RF_VARS_PER_SPLIT,
  seed: SEED
}).train({
  features: trainSampAll,
  classProperty: 'class_id',
  inputProperties: allBands
});

var impDict = ee.Dictionary(rfRank.explain().get('importance'));
var impFc = ee.FeatureCollection(impDict.keys().map(function(k){
  k = ee.String(k);
  return ee.Feature(null, {band: k, importance: ee.Number(impDict.get(k))});
})).sort('importance', false);

var rankedBands = ee.List(impFc.aggregate_array('band'));
print('Ranked bands top 20:', rankedBands.slice(0, 20));

var bandsSelected = ee.List(ee.Algorithms.If(
  USE_CORR_FILTER,
  greedyCorrFilter(trainForCorr, rankedBands, CORR_THRESH, MAX_FEATURES),
  rankedBands.slice(0, MAX_FEATURES)
));

print('Bands after correlation filter:', bandsSelected);
print('Selected feature count:', bandsSelected.size());

var featureStack = featureStackAll.select(bandsSelected).float();

// =============================================================================
// 12) BENCHMARK FULL VS SELECTED
// =============================================================================
var rfFull = trainRF(trainSampAll, 'class_id', allBands, RF_TREES_FINAL, SEED);
var validatedFull = valSampAll.classify(rfFull);
var cmFull = validatedFull.errorMatrix('class_id', 'classification', classIds);

print('FULL RF confusion matrix:', cmFull);
print('FULL RF OA:', cmFull.accuracy());
print('FULL RF Kappa:', cmFull.kappa());

var trainSamp = featureStack.sampleRegions({
  collection: trainPts,
  properties: ['class_id', 'row_id'],
  scale: SCALE,
  tileScale: 4,
  geometries: true
});

var valSamp = featureStack.sampleRegions({
  collection: valPts,
  properties: ['class_id', 'row_id'],
  scale: SCALE,
  tileScale: 4,
  geometries: true
});

print('Sampled train size SELECTED:', trainSamp.size());
print('Sampled validation size SELECTED:', valSamp.size());
print('Sampled train class histogram SELECTED:', trainSamp.aggregate_histogram('class_id'));
print('Sampled validation class histogram SELECTED:', valSamp.aggregate_histogram('class_id'));

var rf = trainRF(trainSamp, 'class_id', featureStack.bandNames(), RF_TREES_FINAL, SEED);
var validated = valSamp.classify(rf);
var cm = validated.errorMatrix('class_id', 'classification', classIds);

print('SELECTED RF confusion matrix:', cm);
print('SELECTED RF OA:', cm.accuracy());
print('SELECTED RF Kappa:', cm.kappa());
print('SELECTED RF Users accuracy:', cm.consumersAccuracy());
print('SELECTED RF Producers accuracy:', cm.producersAccuracy());

// =============================================================================
// 13) FINAL MAP PRODUCTS
// =============================================================================
var classified = featureStack.classify(rf).rename('class_id').clip(aoi);
Map.addLayer(classified, {min: 1, max: 10, palette: palette}, 'DakLak 2024 RF selected');

var coffeeMask = classified.remap([1,2,3], [1,1,1], 0).rename('coffee_mask');
Map.addLayer(coffeeMask.selfMask(), {palette: ['#7f0000']}, 'Coffee mask', false);

// =============================================================================
// 14) TABLES FOR PAPER + SHAP
// =============================================================================
var f1PerClass = perClassMetrics(cm, classIds);
var table4ClassWise = perClassMetricsTable4(cm, classIds, classNames);
var overallSelected = overallMetrics(cm, trainSamp.size(), valSamp.size(), featureStack.bandNames().size(), 'RF_selected_top25');
var cmLongSelected = confusionMatrixToLong(cm, classIds);
var cmRowNormSelected = confusionMatrixRowNormToLong(cm, classIds, classNames);

var overallFull = overallMetrics(cmFull, trainSampAll.size(), valSampAll.size(), allBands.size(), 'RF_full_features');
var cmLongFull = confusionMatrixToLong(cmFull, classIds);
var cmRowNormFull = confusionMatrixRowNormToLong(cmFull, classIds, classNames);
var f1PerClassFull = perClassMetrics(cmFull, classIds);
var table4ClassWiseFull = perClassMetricsTable4(cmFull, classIds, classNames);

var bandListFc = ee.FeatureCollection(bandsSelected.map(function(b){ return ee.Feature(null, {band: ee.String(b)}); }));
var allBandListFc = ee.FeatureCollection(allBands.map(function(b){ return ee.Feature(null, {band: ee.String(b)}); }));

var explainFinal = rf.explain();
var importanceDict = ee.Dictionary(explainFinal.get('importance'));
var viFc = ee.FeatureCollection(importanceDict.keys().map(function(b){
  b = ee.String(b);
  return ee.Feature(null, {variable: b, importance: ee.Number(importanceDict.get(b))});
})).sort('importance', false);
print('Table 4 class-wise accuracy SELECTED:', table4ClassWise);
print('Table 4 class-wise accuracy FULL:', table4ClassWiseFull);
print('Top 20 final-model importance:', viFc.limit(20));

// =============================================================================
// 15) AREA STATISTICS
// =============================================================================
var PIXEL_AREA_HA = ee.Image.pixelArea().divide(10000).rename('area_ha');

var totalAreaHa = ee.Number(PIXEL_AREA_HA.updateMask(classified.mask()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: aoi,
  scale: 10,
  maxPixels: 1e13,
  tileScale: 4
}).get('area_ha'));

var classAreaFc = ee.FeatureCollection(classTable.map(function(d){
  var mask = classified.eq(d.id);
  var areaImg = PIXEL_AREA_HA.updateMask(mask).rename('area_ha');
  var areaHa = ee.Number(areaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e13,
    tileScale: 4
  }).get('area_ha'));

  return ee.Feature(null, {
    class_id: d.id,
    class_name: d.name,
    area_ha: areaHa,
    area_pct: areaHa.divide(totalAreaHa).multiply(100)
  });
}));

var coffeeArea = ee.Number(classAreaFc.filter(ee.Filter.inList('class_id', [1,2,3])).aggregate_sum('area_ha'));
print('Total mapped area ha:', totalAreaHa);
print('Per-class area:', classAreaFc);
print('Total coffee area ha:', coffeeArea);
print('Coffee pct of AOI:', coffeeArea.divide(totalAreaHa).multiply(100));

// =============================================================================
// 16) DISTRICT-LEVEL COFFEE AREA
// =============================================================================
var districtFc = ee.FeatureCollection(ee.Algorithms.If(
  USE_CUSTOM_DISTRICTS,
  ee.FeatureCollection(DISTRICT_ASSET),
  ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam'))
    .filter(ee.Filter.eq('ADM1_NAME', 'Dak Lak'))
));

print('District boundary sample:', districtFc.first());
print('District count:', districtFc.size());

var sunCoffee = classified.eq(1).rename('sun_coffee');
var intercropCoffee = classified.eq(2).rename('intercrop_coffee');
var youngCoffee = classified.eq(3).rename('newly_planted_coffee');

var areaStackDistrict = ee.Image.cat([
  PIXEL_AREA_HA.updateMask(coffeeMask).rename('mapped_coffee_area_ha'),
  PIXEL_AREA_HA.updateMask(sunCoffee).rename('mapped_sun_coffee_area_ha'),
  PIXEL_AREA_HA.updateMask(intercropCoffee).rename('mapped_intercrop_coffee_area_ha'),
  PIXEL_AREA_HA.updateMask(youngCoffee).rename('mapped_newly_planted_coffee_area_ha')
]);

var areaByDistrict = areaStackDistrict.reduceRegions({
  collection: districtFc,
  reducer: ee.Reducer.sum(),
  scale: 10,
  crs: 'EPSG:32649',
  tileScale: 4
}).map(function(f){
  var districtName = ee.String(f.get(DISTRICT_NAME_PROP));
  var coffee = ee.Number(f.get('mapped_coffee_area_ha'));
  var sun = ee.Number(f.get('mapped_sun_coffee_area_ha'));
  var intercrop = ee.Number(f.get('mapped_intercrop_coffee_area_ha'));
  var young = ee.Number(f.get('mapped_newly_planted_coffee_area_ha'));
  return ee.Feature(null, {
    district_name_raw: districtName,
    mapped_coffee_area_ha: coffee,
    mapped_sun_coffee_area_ha: sun,
    mapped_intercrop_coffee_area_ha: intercrop,
    mapped_newly_planted_coffee_area_ha: young,
    mapped_coffee_area_check_ha: sun.add(intercrop).add(young)
  });
});
print('Coffee area by district:', areaByDistrict);

// =============================================================================
// 17) OPTIONAL PROBABILITY + UNCERTAINTY LAYERS
// =============================================================================
var probBands = null;
var maxProb = null;
var entropyNorm = null;
var coffeeConfFc = null;

if (EXPORT_PROBABILITY) {
  var rfProb = ee.Classifier.smileRandomForest({
    numberOfTrees: RF_TREES_FINAL,
    variablesPerSplit: RF_VARS_PER_SPLIT,
    minLeafPopulation: RF_MIN_LEAF,
    bagFraction: RF_BAG_FRACTION,
    seed: SEED
  }).setOutputMode('MULTIPROBABILITY')
    .train({
      features: trainSamp,
      classProperty: 'class_id',
      inputProperties: featureStack.bandNames()
    });

  var probArr = featureStack.classify(rfProb);
  probBands = probArr.arrayFlatten([[
    'p_cls1','p_cls2','p_cls3','p_cls4','p_cls5',
    'p_cls6','p_cls7','p_cls8','p_cls9','p_cls10'
  ]]);

  maxProb = probBands.reduce(ee.Reducer.max()).rename('max_prob');
  var safeP = probBands.max(1e-10);
  var plogp = safeP.multiply(safeP.log());
  var entropy = plogp.reduce(ee.Reducer.sum()).multiply(-1).rename('entropy');
  entropyNorm = entropy.divide(Math.log(10)).rename('entropy_norm');

  Map.addLayer(maxProb, {min: 0.3, max: 1.0, palette: ['#d7191c','#ffffbf','#1a9641']}, 'Max class probability', false);
  Map.addLayer(entropyNorm, {min: 0, max: 1, palette: ['#1a9641','#ffffbf','#d7191c']}, 'Entropy normalized', false);

  var highConfCoffee = coffeeMask.multiply(entropyNorm.lt(0.5));
  var lowConfCoffee = coffeeMask.multiply(entropyNorm.gte(0.5));

  var areaHighConf = ee.Number(highConfCoffee.multiply(PIXEL_AREA_HA).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e13, tileScale: 4
  }).get('coffee_mask'));

  var areaLowConf = ee.Number(lowConfCoffee.multiply(PIXEL_AREA_HA).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: 10, maxPixels: 1e13, tileScale: 4
  }).get('coffee_mask'));

  coffeeConfFc = ee.FeatureCollection([
    ee.Feature(null, {confidence: 'high_entropy_lt_0_5', area_ha: areaHighConf}),
    ee.Feature(null, {confidence: 'low_entropy_gte_0_5', area_ha: areaLowConf})
  ]);
  print('Coffee area by confidence:', coffeeConfFc);
}

// =============================================================================
// 18) EXPORTS
// =============================================================================
if (EXPORT_MAPS) {
  Export.image.toDrive({
    image: classified.toInt(),
    description: 'DakLak_2024_10class_RF_corrTop25',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'DakLak_2024_10class_RF_corrTop25',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: coffeeMask.toByte(),
    description: 'DakLak_2024_CoffeeMask_RF_corrTop25',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'DakLak_2024_CoffeeMask_RF_corrTop25',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });
}

if (EXPORT_CLASSIFIED_ASSET) {
  // Required by Step 02 area statistics. If the asset already exists, delete it or change CLASSIFIED_ASSET_ID.
  Export.image.toAsset({
    image: classified.toInt(),
    description: 'Asset_DakLak_2024_10class_RF_corrTop25',
    assetId: CLASSIFIED_ASSET_ID,
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });
}

if (EXPORT_PREDICTOR_STACK) {
  Export.image.toDrive({
    image: featureStack,
    description: 'PredictorStack_SelectedTop25_2024',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'PredictorStack_SelectedTop25_2024',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });
}

if (EXPORT_TABLES) {
  // Full feature space train/validation for Python Table 3
  Export.table.toDrive({collection: trainSampAll, description: 'Table_TrainSamples_FullFeatureSpace_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_TrainSamples_FullFeatureSpace_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: valSampAll, description: 'Table_ValSamples_FullFeatureSpace_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ValSamples_FullFeatureSpace_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: allBandListFc, description: 'Table_CandidateBandList_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_CandidateBandList_2024', fileFormat: 'CSV'});

  // Final selected-feature train/validation for SHAP and Table 4
  Export.table.toDrive({collection: trainSamp, description: 'Table_TrainSamples_RF_Final_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_TrainSamples_RF_Final_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: valSamp, description: 'Table_ValSamples_RF_Final_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ValSamples_RF_Final_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: validated, description: 'Table_ValPredictions_RF_Final_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ValPredictions_RF_Final_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: validatedFull, description: 'Table_ValPredictions_RF_FullFeatures_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ValPredictions_RF_FullFeatures_2024', fileFormat: 'CSV'});

  // Importance and selected bands
  Export.table.toDrive({collection: viFc, description: 'Table_RF_VariableImportance_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_RF_VariableImportance_DakLak2024_corrTop25', fileFormat: 'CSV'});
  Export.table.toDrive({collection: impFc, description: 'Table_RF_RankingImportance_FullFeatureSpace_2024', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_RF_RankingImportance_FullFeatureSpace_2024', fileFormat: 'CSV'});
  Export.table.toDrive({collection: bandListFc, description: 'Table_SelectedBands_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_SelectedBands_corrTop25', fileFormat: 'CSV'});

  // Accuracy outputs selected and full
  Export.table.toDrive({collection: cmLongSelected, description: 'Table_ConfusionMatrix_Long_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ConfusionMatrix_Long_DakLak2024_corrTop25', fileFormat: 'CSV'});
  Export.table.toDrive({collection: cmRowNormSelected, description: 'Table_ConfusionMatrix_RowNorm_Long_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ConfusionMatrix_RowNorm_Long_DakLak2024_corrTop25', fileFormat: 'CSV'});
  Export.table.toDrive({collection: f1PerClass, description: 'Table_F1_PerClass_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_F1_PerClass_DakLak2024_corrTop25', fileFormat: 'CSV'});
  Export.table.toDrive({collection: table4ClassWise, description: 'Table4_ClassWiseAccuracy_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table4_ClassWiseAccuracy_DakLak2024_corrTop25', fileFormat: 'CSV'});
  Export.table.toDrive({collection: overallSelected, description: 'Table_Accuracy_Overall_DakLak2024_corrTop25', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_Accuracy_Overall_DakLak2024_corrTop25', fileFormat: 'CSV'});

  Export.table.toDrive({collection: cmLongFull, description: 'Table_ConfusionMatrix_Long_DakLak2024_fullFeatures', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ConfusionMatrix_Long_DakLak2024_fullFeatures', fileFormat: 'CSV'});
  Export.table.toDrive({collection: cmRowNormFull, description: 'Table_ConfusionMatrix_RowNorm_Long_DakLak2024_fullFeatures', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_ConfusionMatrix_RowNorm_Long_DakLak2024_fullFeatures', fileFormat: 'CSV'});
  Export.table.toDrive({collection: f1PerClassFull, description: 'Table_F1_PerClass_DakLak2024_fullFeatures', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_F1_PerClass_DakLak2024_fullFeatures', fileFormat: 'CSV'});
  Export.table.toDrive({collection: table4ClassWiseFull, description: 'Table4_ClassWiseAccuracy_DakLak2024_fullFeatures', folder: EXPORT_FOLDER, fileNamePrefix: 'Table4_ClassWiseAccuracy_DakLak2024_fullFeatures', fileFormat: 'CSV'});
  Export.table.toDrive({collection: overallFull, description: 'Table_Accuracy_Overall_DakLak2024_fullFeatures', folder: EXPORT_FOLDER, fileNamePrefix: 'Table_Accuracy_Overall_DakLak2024_fullFeatures', fileFormat: 'CSV'});
}

if (EXPORT_AREA_STATS) {
  Export.table.toDrive({
    collection: classAreaFc,
    description: 'Table_AreaStatistics_DakLak2024',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'Table_AreaStatistics_DakLak2024',
    fileFormat: 'CSV',
    selectors: ['class_id', 'class_name', 'area_ha', 'area_pct']
  });
}

if (EXPORT_DISTRICT_AREA) {
  Export.table.toDrive({
    collection: areaByDistrict,
    description: 'GEE_DakLak_Coffee_Area_By_District_2024',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'GEE_DakLak_Coffee_Area_By_District_2024',
    fileFormat: 'CSV'
  });
}

if (EXPORT_PROBABILITY) {
  Export.image.toDrive({
    image: maxProb.toFloat(),
    description: 'DakLak_2024_MaxProbability',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'DakLak_2024_MaxProbability',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: entropyNorm.toFloat(),
    description: 'DakLak_2024_EntropyNormalised',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'DakLak_2024_EntropyNormalised',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: probBands.toFloat(),
    description: 'DakLak_2024_ClassProbabilities',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'DakLak_2024_ClassProbabilities',
    region: aoi,
    crs: 'EPSG:32649',
    scale: 10,
    maxPixels: 1e13
  });

  Export.table.toDrive({
    collection: coffeeConfFc,
    description: 'Table_CoffeeArea_ByConfidence_DakLak2024',
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'Table_CoffeeArea_ByConfidence_DakLak2024',
    fileFormat: 'CSV',
    selectors: ['confidence', 'area_ha']
  });
}

// =============================================================================
// 19) LEGEND
// =============================================================================
var showLegend = true;
if (showLegend) {
  var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
  legend.add(ui.Label('Classification', {fontWeight: 'bold', fontSize: '14px'}));
  classTable.forEach(function(d){
    var box = ui.Label('', {backgroundColor: d.color, padding: '8px', margin: '0 6px 4px 0'});
    var lab = ui.Label(d.id + '. ' + d.name, {margin: '0 0 4px 0'});
    legend.add(ui.Panel([box, lab], ui.Panel.Layout.flow('horizontal')));
  });
  Map.add(legend);
}
