# Map Matching using Brisbane bus data

Author: Nick Malleson

Uses the [GraphHopper](https://github.com/graphhopper/graphhopper) [MapMatching](https://github.com/graphhopper/map-matching) library to take GPS points and create a route from them, attached to a road network.

Similar to some code that I wrote for the [surf](http://surf.leeds.ac.uk/) project called [BreezeRoutes](https://github.com/nickmalleson/surf/tree/master/projects/BreezeRoutes)  

The class `org.dust.leeds.mapmatching.MapMatchingMain` does most of the work.

This is an IntelliJ IDEA project maven project, so I recommend using that IDE, but presumably others will work too.

## Required Libraries

All the required libraries should get downloaded to the ./lib/ directory when you build the project. It uses Maven for this..

The main third-party libraries are [Graphhopper](https://github.com/graphhopper/graphhopper) (for routing) and [map-matching](https://github.com/graphhopper/map-matching) for building a route (list of OSM street segments) from the GPS data. These have been imported with Maven.


## 1. Download OSM data

You need to download Open Street Map data first.

The `DataReader` is quite flexible and should take: xml (.osm), a compressed xml (.osm.zip or .osm.gz) or a protobuf file (.pbf).

Download the files and store them in the `map-data` directory. [GeoFabrik](http://download.geofabrik.de) has lots of OSM data.

E.g. the following should work Australian data:

```
cd map-data
wget https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf
```

Although that's a lot of data, so instead I went to the `openstreetmap` website and extracted the area around Brisbane using [this url](https://www.openstreetmap.org/export#map=9/-27.4729/152.8185) and the `Overpass API` option ([full link](https://overpass-api.de/api/map?bbox=151.1801,-28.9409,154.4568,-25.9852)). Then used a program to compress the massive xml file that comes from the overapass API:

```
./osmosis --read-xml file=brisbane.osm --write-pbf file=brisbane.osm.pbf
```

## 2. Convert GPS input data to GPX

  1. The csv file provided needs to be edited slightly. Firstly the `gps_fix` column, which contains the GPS points for each entry, needs to be split into two separate columns (called `gpsfix_A` and `gpsfix_B`. This can be done in a text editor. Get rid of all the brackets as well. You can see if it has worked by opening in Excel and making sure the columns are correct.

  2. Then use the `gps_data/convert_csv_to_gps.R` script to convert the CSV file to GPX using R. You might need to change the working directory in the script to match your own setup. The script should create a new gpx file in the `../gpx` directory.
