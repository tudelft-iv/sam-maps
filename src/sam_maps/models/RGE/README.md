# Road Graph Extraction (RGE) Module 

The road graph extraction module is responsible from extracting a road graph matching the region of interest. This provides the connectivity of the road map.

This repository provides two different methods for the RGE module: **OSM-Extractor** and **SAM-Road**. Furthermore it is possible to perform the road graph extraction step in a semi-automated fashion. The proposed road graph by SAM-Road, can be adapted. For this we provide a QGIS-plugin. If you select `manual = true` in the configuration file, a `geojson` file is created of the graph. These can be loaded into QGIS and using the plugin we can adapt it. (More on this in [QGIS-plugin](#gqis-plugin)).

## Input
As described in the main `README.md`, we have multiple different options as an input to the road graph extraction:

#### OSM-Extractor:
1. Set `area_selection_method` to `"place"` and set `place` to the city/town/village of interest.
2. Set `area_selection_method` to `"bbox"` and set `bbox` to the region of interest. You can include multiple bounding boxes here with the format: `[min_longitude, min_latitude, max_longitude, max_latitude]`
3. Set `area_selection_method` to `"annot_bbox"` This will create a bounding box around that captures the entire area around the given ground truth map. 

#### SAM-Road
1. Set `"bbox"` to the bounding box(es) capturing the region you are interested in. 
2. Set `"graph.nodes"` and `"graph.edges"` To geojson files with the road graph you would like to use and additionally set `manual` to `true`. 

## Output
The output is the road graph with the following structure:


## Configuration


## GQIS-Plugin
