package org.dust.leeds.mapmatching;

import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import com.graphhopper.GraphHopper;
import com.graphhopper.PathWrapper;
import com.graphhopper.matching.*;
import com.graphhopper.matching.gpx.Gpx;
import com.graphhopper.reader.osm.GraphHopperOSM;
import com.graphhopper.routing.AlgorithmOptions;
import com.graphhopper.routing.Dijkstra;
import com.graphhopper.routing.Path;
import com.graphhopper.routing.QueryGraph;
import com.graphhopper.routing.util.*;
import com.graphhopper.storage.GraphHopperStorage;
import com.graphhopper.storage.index.LocationIndexTree;
import com.graphhopper.storage.index.QueryResult;
import com.graphhopper.util.*;
import com.graphhopper.util.gpx.GpxFromInstructions;
import com.graphhopper.util.shapes.GHPoint;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class MapMatchingMain {


    /** Debug mode. Stop afer 20 files */
    private final static boolean DEBUG = true;

    /** Whether to overide matched- or shortest-paths that have already been created */
    private final static boolean OVERWRITE = false;

    /** The location of the input OSM pbf file */
    private final static String OSM_DATA_FILE = "./map-data/brisbane.osm.pbf";

    /** The location to cache the graph after reading the OSM data (Graphhopper creates this on first run) */
    private final static String CACHE_DIR = "./cache/";

    /** A class to store the names of the directories where data files can be read and written */
    private static final class Directories {
        /** The root for all files */
        private static final String ROOT = System.getProperty("user.home")+"/gp/dust/Projects/MapMatching/";
        //String ROOT = "./traces"
        /** Subdirectory contains the original gpx files (to be read) */
        static final String ORIG_GPX = ROOT + "gpx/";
        /** Subdirectory contains the matched (output) gpx files */
        static final String GPX_MATCHED = ROOT + "gpx-matched/";
        /** Subdirectory contains the shortest path (output) gpx files */
        //static final String GPX_SHORTEST = ROOT + "gpx-shortest/";
        /** Subdirectory for GPS data **/
        static final String ORIG_GPS = ROOT + "gps_data/";
    }

    /** Whether or not to write out the GPX traces after matching. These files contain the original gps points and path, as well as the matched route.*/
    private static boolean WRITE_MATCHED_PATH = true;

    private static GraphHopperOSM hopper;
    private static MapMatching mapMatching;
    private static GraphHopperStorage graph;
    private static FlagEncoder encoder;
    //private static MiniGraphUI ui; // For visualising the graph(s)

    /**
     * Do the matching.
     * @param osmDataFile The file that contains the OSM data.
     * @param cacheDir The directory to use for cacheing the graph (generated from the OSM data)
     * @param writeMatchedPath Whether to write GPX files for each matched path
     */
    public MapMatchingMain(String osmDataFile, String cacheDir, boolean writeMatchedPath) throws Exception {
        init(osmDataFile, cacheDir);
        run(writeMatchedPath);
    }

    /**
     * Initialise the map matcher. Load the OSM data, cache the graph, and prepare the required objects.
     */
    private static void init(String osmDataFile, String cacheDir) {
        System.out.println("Initialising Map Matching");
        // import OpenStreetMap data
        hopper = new GraphHopperOSM();
        hopper.setOSMFile(osmDataFile);
        //hopper.setDataReaderFile(osmDataFile);
        hopper.setGraphHopperLocation(cacheDir);
        hopper.setEncodingManager(EncodingManager.start().add(new CarFlagEncoder()).build());
        //hopper.getCHFactoryDecorator().setEnabled(false);
        hopper.importOrLoad();

        // Set algorithm options
        AlgorithmOptions opts = AlgorithmOptions.start()
                //.maxVisitedNodes(maxVisitedNodes)
                .algorithm(Parameters.Algorithms.DIJKSTRA_BI)
                .hints(new HintsMap()
                        .put("vehicle", "car") // XXXX TRY 'bus'
                        .put("ch.disable", new Object()) // Not sure about this, I think it is a workaround for a bug
                )
                .build();

        mapMatching= new MapMatching(hopper, opts);

        /* OLD
        // This Encoder specifies how the network should be navigated. If it is changes (i.e. from foot to car) then the
        // cache needs to be deleted to force importOrLoad() to recalculate the graph.
        //CarFlagEncoder Encoder = new CarFlagEncoder();
        encoder = new FootFlagEncoder();
        hopper.setEncodingManager(new EncodingManager(encoder));
        hopper.getCHFactoryDecorator().setEnabled(false);
        hopper.importOrLoad();

        //ui = new MiniGraphUI(hopper, true);


        // create MapMatching object, can and should be shared across threads
        graph = hopper.getGraphHopperStorage();
        LocationIndexMatch locationIndexMatch = new LocationIndexMatch(graph, (LocationIndexTree) hopper.getLocationIndex());
        mapMatching = new MapMatching(graph, locationIndexMatch, encoder);

        // Configure some parameters to try to stop the algorithm breaking.

        // The following attempts to fix errors like:
        // Could not match file 58866174745e88db19b3eced744141fc.gpx. Message: Cannot find matching path! Wrong vehicle foot or missing OpenStreetMap data? Try to increase max_visited_nodes (500). Current gpx sublist:2, start list:[6547-300786  42.35620562568512,...
        // mapMatching.setMaxVisitedNodes(1000); // Didn't work

        // The following attempts to fix errors like:
        // Could not match file 5e6c6e4d6cc9efe9e314aa77a229f88e.gpx. Message:  Result contains illegal edges. Try to decrease the separated_search_distance (300.0) or use force_repair=true. Errors:[duplicate edge::457954->304150
        //mapMatching.setSeparatedSearchDistance(200);
        mapMatching.setForceRepair(true);
        */
    }

    /**
     * Run the map matcher.
     * @param writeMatchedPath Whether or not to write each mathched path (as GPX).
     * @throws IOException
     */
    private static void run(boolean writeMatchedPath) throws IOException, Exception {
        System.out.println("Running Map Matching");
        // Check which directories will be used to read and write the data to/from.
        File gpxDir = new File(Directories.ORIG_GPX);
        File matchedDir = new File(Directories.GPX_MATCHED);
        //File shortestDir = new File(Directories.GPX_SHORTEST);
        //for (File dir : Arrays.asList(new File[]{gpxDir, matchedDir, shortestDir}) ) {
        for (File dir : Arrays.asList(new File[]{gpxDir, matchedDir}) ) {
            if (!dir.isDirectory()) {
                throw new IOException("Error: '"+dir+"' is not a directory");
            }
        }

        // Read all the gpx files

        System.out.println("Reading directory: "+gpxDir);
        File[] allFiles = gpxDir.listFiles();
        System.out.println("\tThere are "+allFiles.length+" files in the directory.\nReading files.");

        XmlMapper xmlMapper = new XmlMapper(); // Used to parse the GPX file

        int success = 0; // Remember the number of files successfull processed (or not)
        int failed = 0;
        int ignored = 0;
        for (int i = 0; i < allFiles.length; i++) {
            if (DEBUG && i > 20) {
                System.out.println("Debug mode is on. Stopping now.");
                break;
            }
            if (i % 5000 == 0) {
                System.out.println("\t .. read file "+i);
            }
            File file = allFiles[i];
            if ( file.isFile() && file.getName().endsWith(".gpx") ) {

                System.out.println("Reading file ("+i+"): "+file);
                String matchedFilename =  Directories.GPX_MATCHED + file.getName().substring(0,file.getName().length()-4)+ "-matched.gpx";
                //String shortestFilename = Directories.GPX_SHORTEST + file.getName().substring(0,file.getName().length()-4)+ "-shortest.gpx";

                //if ( ( new File(matchedFilename).exists() || new File(shortestFilename).exists()) && !OVERWRITE) {
                if ( new File(matchedFilename).exists() && !OVERWRITE) {
                    System.out.println("\tShortest- or matched-file already exists, ignoring ");
                    ignored++;
                    continue;
                }

                // get the GPX entries from a file

                Gpx gpx = xmlMapper.readValue(file, Gpx.class);
                if (gpx.trk == null) {
                    throw new IllegalArgumentException("No tracks found in GPX document. Are you using waypoints or routes instead?");
                }
                if (gpx.trk.size() > 1) {
                    throw new IllegalArgumentException("GPX documents with multiple tracks not supported yet.");
                }
                List<Observation> inputGPXEntries = gpx.trk.get(0).getEntries();
                //List<Observation> inputGPXEntries = new GPXFile().doImport(file.getAbsolutePath()).getEntries();
                MatchResult mr = mapMatching.doWork(inputGPXEntries );
                System.out.println(file);
                System.out.println("\tmatches:\t" + mr.getEdgeMatches().size() + ", gps entries:" + inputGPXEntries.size());
                System.out.println("\tgpx length:\t" + (float) mr.getGpxEntriesLength() + " vs " + (float) mr.getMatchLength());

                // Do the matching
                /*Path matchedPath;
                try {
                    matchedPath = match(inputGPXEntries);
                }
                catch (java.lang.RuntimeException ex) {
                    System.err.println("Could not match file "+file.getName() + ". Message: "+ex.getMessage());
                    // TODO do something about these errors - maybe move the files to make it easier to analyse them
                    failed++;
                    continue;
                }*/

                if (writeMatchedPath) {
                    Translation tr = new TranslationMap().doImport().getWithFallBack(Helper.getLocale("en"));
                    final boolean withRoute = true; // ??
                    PathWrapper pathWrapper = new PathWrapper();
                    new PathMerger().doWork(pathWrapper, Collections.singletonList(mr.getMergedPath()), hopper.getEncodingManager(), tr);

                    BufferedWriter writer = new BufferedWriter(new FileWriter(matchedFilename));
                        long time = gpx.trk.get(0).getStartTime()
                                .map(Date::getTime)
                                .orElse(System.currentTimeMillis());
                        writer.append(GpxFromInstructions.createGPX(
                                pathWrapper.getInstructions(), gpx.trk.get(0).name != null ? gpx.trk.get(0).name : "", time, hopper.hasElevation(), withRoute, true, false, Constants.VERSION, tr));

                }

                success++;


            } // if isfile
            else {
                System.out.println("\tIgnoring "+file);
                ignored++;
            }
        } // For all files
        if (DEBUG) {
            System.out.println("WARN: Debug is ON, so no output will actually have been created");
        }
        System.out.println("Finished. Processed "+ (failed+success+ignored)+" files."+
                "\n\tSuccess:"+success+
                "\n\tFailed: "+failed+
                "\n\tIgnored: "+ignored
        );


    }


    /**
     * Take the input gpx entries (e.g. from <code>new GPXFile().doImport("a_file.gpx").getEntries()</code>) and match
     * the route to an OSM path.
     * @param inputGPXEntries
     * @return The matched path.
     */
    private static Path match(List<Observation> inputGPXEntries) {
        MatchResult mr = mapMatching.doWork(inputGPXEntries);
        double dist = mr.getMatchLength();
        System.out.println("Finished matching. Length: " + dist);
        mr.getMergedPath();
        //Path path = mapMatching.calcPath(mr);
        Path path = mr.getMergedPath();
        return path;
    }


    public static void main(String[] args ) throws Exception {

        new MapMatchingMain(OSM_DATA_FILE, CACHE_DIR, WRITE_MATCHED_PATH);

    }


}
