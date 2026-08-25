// CLEAN PAPER-1 WORKFLOW FILE
// Run after Step 01 has exported CLASSIFIED_ASSET_ID as an EE asset.
// =============================================================================
// Dak Lak Coffee Mapping 2024, CLEAN WORKFLOW STEP 02: area statistics from exported asset
// Run this AFTER Script 01 successfully exports the classified map as an EE Asset.
// This script is intentionally separated from training to avoid GEE memory errors.
// Time running from 2-3 hours
// =============================================================================

var region = ee.FeatureCollection('users/ntduc11/daklak');
var aoi = region.geometry();
Map.centerObject(region, 8);

var EXPORT_FOLDER = 'GEE_Exports_R3000';
var CLASSIFIED_ASSET_ID = 'users/ntduc11/DakLak_2024_10class_RF_corrTop25';

var USE_CUSTOM_DISTRICTS = true;
var DISTRICT_ASSET = 'users/ntduc11/daklak_districts';
var DISTRICT_NAME_PROP = 'District'; // change to 'District' if needed
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
var palette = classTable.map(function(d){ return d.color; });

var classified = ee.Image(CLASSIFIED_ASSET_ID).rename('class_id').clip(aoi);
print('Classified asset:', classified);
Map.addLayer(classified, {min: 1, max: 10, palette: palette}, 'Classified map asset');

var PIXEL_AREA_HA = ee.Image.pixelArea().divide(10000).rename('area_ha');

// -----------------------------------------------------------------------------
// A) Per-class area
// -----------------------------------------------------------------------------
var totalAreaHa = ee.Number(PIXEL_AREA_HA.updateMask(classified.mask()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: aoi,
  scale: 10,
  maxPixels: 1e13,
  tileScale: 16
}).get('area_ha'));

var classAreaFc = ee.FeatureCollection(classTable.map(function(d){
  var mask = classified.eq(d.id);
  var areaImg = PIXEL_AREA_HA.updateMask(mask).rename('area_ha');
  var areaHa = ee.Number(areaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e13,
    tileScale: 16
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
print('Total coffee area ha:', coffeeArea);
print('Per-class area:', classAreaFc);

Export.table.toDrive({
  collection: classAreaFc,
  description: 'Table_AreaStatistics_DakLak2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'Table_AreaStatistics_DakLak2024',
  fileFormat: 'CSV',
  selectors: ['class_id', 'class_name', 'area_ha', 'area_pct']
});

// -----------------------------------------------------------------------------
// B) District-level coffee area
// -----------------------------------------------------------------------------
var districtFc = ee.FeatureCollection(ee.Algorithms.If(
  USE_CUSTOM_DISTRICTS,
  ee.FeatureCollection(DISTRICT_ASSET),
  ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam'))
    .filter(ee.Filter.eq('ADM1_NAME', 'Dak Lak'))
));

print('District boundary sample:', districtFc.first());
print('District count:', districtFc.size());

var coffeeMask = classified.remap([1,2,3], [1,1,1], 0).rename('coffee_mask');
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
  tileScale: 16
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

print('Coffee area by district:', areaByDistrict.limit(5));

Export.table.toDrive({
  collection: areaByDistrict,
  description: 'GEE_DakLak_Coffee_Area_By_District_2024',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'GEE_DakLak_Coffee_Area_By_District_2024',
  fileFormat: 'CSV'
});
