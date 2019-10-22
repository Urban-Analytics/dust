# Map Matching

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

E.g. the following should work for Massachusetts data:

```
cd map-data
wget http://download.geofabrik.de/north-america/us/massachusetts-latest.osm.pbf
```
