// QUICK TEST: training data v3 quality check
// BEFORE RUNNING: upload ROI_for_GEE_3000_v3.csv as GEE asset (replace users/ntduc11/ROI1)
// Scale=30m + 200 trees for fast iteration. Promote to 10m + 2000 trees for final run.

// ── Settings ───────────────────────────────────────────────────────────────
var TRAINING_ASSET = 'users/ntduc11/ROIv2';
var DAKLAK_ASSET   = 'users/ntduc11/daklak';
var SCALE = 30;    // change to 10 for final run
var RF_TREES = 200; // change to 2000 for final run
var SEED = 2024;

var region = ee.FeatureCollection(DAKLAK_ASSET);
var aoi    = region.geometry();
Map.centerObject(region, 8);

// ── Class table (matches v3 CSV class_id) ─────────────────────────────────
var CLASS_IDS   = [1,2,3,4,5,6,7,8,9,10];
var CLASS_NAMES = ['Sun coffee','Intercrop coffee','Newly planted','Rubber',
                   'Partially vegetative','Rice','Other upland crops','Forest','Water','Built'];
var PALETTE     = ['#8c3b00','#ff8080','#ffb000','#4caf82','#d4c86a',
                   '#0087a8','#f2e6b8','#1f4d2b','#00c8ff','#9a9a9a'];
var classIds    = ee.List(CLASS_IDS);
var classNames  = ee.List(CLASS_NAMES);

// ── Prepare train / val points ─────────────────────────────────────────────
var pts = ee.FeatureCollection(TRAINING_ASSET).map(function(f){
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number.parse(ee.String(f.get('lon'))),
      ee.Number.parse(ee.String(f.get('lat')))
    ]),
    { class_id: ee.Number.parse(ee.String(f.get('class_id'))),
      split:    f.get('split') }
  );
}).filterBounds(aoi).filter(ee.Filter.inList('class_id', classIds));

var trainPts = pts.filter(ee.Filter.eq('split', 'train'));
var valPts   = pts.filter(ee.Filter.eq('split', 'val'));

print('Train count + class hist:', trainPts.size(), trainPts.aggregate_histogram('class_id'));
print('Val   count + class hist:', valPts.size(),   valPts.aggregate_histogram('class_id'));
Map.addLayer(trainPts, {color: 'red'},  'Train pts', false);
Map.addLayer(valPts,   {color: 'blue'}, 'Val pts',   false);

// ── S2 composite helper ────────────────────────────────────────────────────
function s2comp(start, end, tag) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate(start, end)
    .map(function(img){
      var qa  = img.select('QA60');
      var scl = img.select('SCL');
      return img
        .updateMask(qa.bitwiseAnd(1<<10).eq(0).and(qa.bitwiseAnd(1<<11).eq(0)))
        .updateMask(scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)))
        .divide(10000);
    });

  var RAW_BANDS = ['B2','B3','B4','B5','B8','B11','B12'];
  var med  = col.select(RAW_BANDS).median().unmask(0);
  var blue = med.select('B2');
  var red  = med.select('B4');
  var re1  = med.select('B5');
  var nir  = med.select('B8');
  var sw1  = med.select('B11');

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var ndre = nir.subtract(re1).divide(nir.add(re1)).rename('NDRE');  // red-edge, key for coffee canopy
  var ndmi = nir.subtract(sw1).divide(nir.add(sw1)).rename('NDMI');
  var evi  = nir.subtract(red).multiply(2.5)
               .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
               .rename('EVI');
  var mndwi = med.select('B3').subtract(sw1).divide(med.select('B3').add(sw1)).rename('MNDWI');

  var OUT_BANDS = RAW_BANDS.concat(['NDVI','NDRE','NDMI','EVI','MNDWI']);
  return med.addBands([ndvi, ndre, ndmi, evi, mndwi]).rename(
    OUT_BANDS.map(function(n){ return tag+'_'+n; })
  );
}

// ── Feature stack ──────────────────────────────────────────────────────────
var dem     = ee.Image('USGS/SRTMGL1_003').rename('elev');
var terrain = ee.Terrain.products(dem);
var demStack = dem.addBands(terrain.select('slope'));

var stack = ee.Image.cat([
  s2comp('2023-11-01','2024-04-30','dry'),
  s2comp('2024-05-01','2024-10-31','wet'),
  demStack
]).clip(aoi).float();

print('Feature bands:', stack.bandNames());

// ── Sample & train RF ──────────────────────────────────────────────────────
var trainSamp = stack.sampleRegions({collection: trainPts, properties: ['class_id', 'row_id'], scale: SCALE, tileScale: 4});
var valSamp   = stack.sampleRegions({collection: valPts,   properties: ['class_id', 'row_id'], scale: SCALE, tileScale: 4});

var rf = ee.Classifier.smileRandomForest({numberOfTrees: RF_TREES, seed: SEED})
  .train({features: trainSamp, classProperty: 'class_id', inputProperties: stack.bandNames()});

// ── Validate ───────────────────────────────────────────────────────────────
var validated = valSamp.classify(rf);
var cm = validated.errorMatrix('class_id', 'classification', classIds);

print('=== CONFUSION MATRIX ===', cm);
print('Overall accuracy:', cm.accuracy());
print('Kappa:', cm.kappa());

var ua = ee.Array(cm.consumersAccuracy()).toList().flatten();
var pa = ee.Array(cm.producersAccuracy()).toList().flatten();

// Print as flat lists, visible directly in Console, no clicking needed
var pct = function(x){ return ee.Number(x).multiply(100).round(); };
print('UA% [C1 C2 C3 C4 C5 C6 C7 C8 C9 C10]:', ua.map(pct));
print('PA% [C1 C2 C3 C4 C5 C6 C7 C8 C9 C10]:', pa.map(pct));
print('F1% [C1 C2 C3 C4 C5 C6 C7 C8 C9 C10]:',
  ua.zip(pa).map(function(pair){
    var u = ee.Number(ee.List(pair).get(0));
    var p = ee.Number(ee.List(pair).get(1));
    return ee.Number(ee.Algorithms.If(
      u.add(p).gt(0),
      u.multiply(p).multiply(2).divide(u.add(p)),
      0
    )).multiply(100).round();
  })
);

// ── Map display ────────────────────────────────────────────────────────────
var classified = stack.classify(rf).clip(aoi);
Map.addLayer(classified, {min:1, max:10, palette: PALETTE}, 'Test classification v3');

// Legend
var legend = ui.Panel({style:{position:'bottom-left',padding:'6px'}});
legend.add(ui.Label('Classes (v3)', {fontWeight:'bold'}));
CLASS_IDS.forEach(function(id, i){
  legend.add(ui.Panel([
    ui.Label('', {backgroundColor: PALETTE[i], padding:'7px', margin:'0 5px 2px 0'}),
    ui.Label(id+'. '+CLASS_NAMES[i], {margin:'2px 0'})
  ], ui.Panel.Layout.flow('horizontal')));
});
Map.add(legend);
