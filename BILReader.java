package hyperspecutils;

/*
 * Utility class Java class for processing BIL files from PSI hyperspectral camera and training and using machine-learning pixel classification models.

Author: Alexander Ivakov
 */

import com.opencsv.CSVWriter;
import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Core;
import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.CvType.CV_16U;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC3;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvStatModel;
import org.opencv.ml.CvKNearest;
import javax.swing.JFileChooser;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32S;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 *
 * @author u4110553
 */

public class BILReader {
    
        private int calibframes = 4;
        private int calibimageframes;
        private int calibwidth;
        private int calibdepth;
        private int imageframes;
        private int width;
        private int depth;
        private final boolean normalise;
        private float[] wavelengths;
        private float[] calibwavelengths;
        private ArrayList<Mat> calibration;
        private ArrayList<Mat> turnedhyperimage;
        private ArrayList<Mat> pcaCube;
        private Mat eigenvectors;
        private CvStatModel model;
        private ArrayList<Spectrum> class1spectra;
        private ArrayList<Spectrum> class2spectra;
        private Mat trainingMatrix;
        private Mat responseMatrix;
        private Mat classifiedImage;
        private final String BILFile;
        private int offset;
        
    public BILReader(String BILFile, String calibfile, boolean normalise){
        readHDR(BILFile, false);
        if(calibfile != null){
        readHDR(calibfile, true);
        }
        this.normalise = normalise;
        this.calibration = new ArrayList();
        this.turnedhyperimage = new ArrayList();
        this.BILFile = BILFile;
        if(!Arrays.equals(wavelengths, calibwavelengths)){
            System.out.println("Image and calibration files are incompatible");
        }
    }

    /**
     * @return the calibframes
     */
    public int getCalibframes() {
        return calibframes;
    }

    /**
     * @return the imageframes
     */
    public int getImageframes() {
        return imageframes;
    }

    /**
     * @return the width
     */
    public int getWidth() {
        return width;
    }

    /**
     * @return the depth
     */
    public int getDepth() {
        return depth;
    }
    
    /**
     * @return the offset
     */
    public int getOffset() {
        return offset;
    }

    /**
     * @return the normalise
     */
    public boolean isNormalise() {
        return normalise;
    }
    
    public ArrayList<Mat> getHyperimage(){
        return turnedhyperimage;
    }
    
    public void readBIL(String BILFile, String calibFile){
        
    ArrayList<Mat> hyperimage = new ArrayList();

    DataInputStream image_data_in = null;
            try {
                
                if(normalise){
                    File f = new File(calibFile);
                    
                    try (DataInputStream data_in = new DataInputStream(
                            new BufferedInputStream(
                                    new FileInputStream(f)))) {
                        
                        for(int j=0; j<calibframes; j++){
                            
                            short[] pixels = new short[calibwidth*calibdepth];
                            for(int i=0; i<calibwidth*calibdepth; i++){
                                
                                try {
                                    short c = data_in.readShort();//read 2 bytes
                                    
                                    short d = Short.reverseBytes(c);
                                    //int e = d & 0xffff;
                                    
                                    pixels[i] = (short)(d/16);
                                    
                                    //System.out.println(e);
                                }
                                catch (java.io.EOFException eof) {
                                    break;
                                }
                                
                            }
                            
                            //Mat img = new Mat();
                            Mat calib = new Mat(calibdepth, calibwidth, CV_16U);
                            calib.put(0, 0, pixels);
                            
                            Mat calib32 = new Mat(calibdepth, calibwidth, CV_32F);
                            calib.convertTo(calib32, CV_32F);
                            
                            calibration.add(calib32);
                            
                        }       
                        
                        
                    }       catch (IOException ex) {
                        Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }       
                
                //Loading BIL file here
                
                String imagepath = BILFile;
                File imagef = new File(imagepath);
                image_data_in = new DataInputStream(
                        new BufferedInputStream(new FileInputStream(imagef)));
                
                for(int o=0; o<offset/2; o++){
                    try {
                            short temp = image_data_in.readShort();//read 2 bytes
                        }
                        catch (java.io.EOFException eof) {
                            break;
                        } catch (IOException ex) {
                            Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                        }
                }
                
                for(int k=0; k<imageframes; k++){
                    
                    short[] image_pixels = new short[depth*width];
                    for(int i=0; i<width*depth; i++){
                        
                        try {
                            short c = image_data_in.readShort();//read 2 bytes
                            
                            short d = Short.reverseBytes(c);
                            //int e = d & 0xffff;
                            
                            image_pixels[i] = (short)(d/16);
                            
                        }
                        catch (java.io.EOFException eof) {
                            break;
                        } catch (IOException ex) {
                            Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                        }
                        
                    }
                    
                    //Mat img = new Mat();
                    Mat img = new Mat(depth, width, CV_16U);
                    img.put(0, 0, image_pixels);
                    
                    Mat img32 = new Mat(depth, width, CV_32F);
                    img.convertTo(img32, CV_32F);
                    
                    Mat norm = new Mat(depth, width, CV_32F);
                    if(!normalise){
                    Core.divide(img32, new Scalar(65535F), norm);
                    } else {
                    
                    Core.divide(img32, calibration.get(0), norm);
                    }

                    hyperimage.add(norm);
                    
                }
                
            } catch (FileNotFoundException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                /*try {
                    //image_data_in.close();
                } catch (IOException ex) {
                    Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                }*/
            }
    
    for(int m=0; m<depth; m++){
        
        Mat current = new Mat(imageframes, width, CV_32F); 
        
        for(int n=0; n<imageframes; n++){
            
            float[] widthrow = new float[width];
            
            hyperimage.get(n).get(m, 0, widthrow);
           
            current.put(n, 0, widthrow);
            
        }
        
        turnedhyperimage.add(current);
    }
    
}
    
    public Mat getRGB(){
        
        ArrayList<Mat> rgblist = new ArrayList();
        Mat blue = new Mat();
        int blueindex = 105;
        int greenindex = 140; 
        int redindex = 250;
        
        for(int scan=0; scan<wavelengths.length; scan++){
            
            if((int)wavelengths[scan] == 484){
                blueindex = scan;
            }
            if((int)wavelengths[scan] == 512){
                greenindex = scan;
            }
            if((int)wavelengths[scan] == 638){
                redindex = scan;
                break;
            }
        }
        System.out.println(blueindex);
        System.out.println(greenindex);
        System.out.println(redindex);
        
        turnedhyperimage.get(blueindex).convertTo(blue, CV_8U, 255);
        rgblist.add(blue);
        Mat green = new Mat();
        turnedhyperimage.get(greenindex).convertTo(green, CV_8U, 255);
        rgblist.add(green);
        Mat red = new Mat();
        turnedhyperimage.get(redindex).convertTo(red, CV_8U, 255);
        rgblist.add(red);

        Mat dst = new Mat();
        Core.merge(rgblist, dst);
        Mat dst16norm = new Mat();
        Core.normalize(dst, dst16norm, 0, 1000, NORM_MINMAX);
        Mat dst8 = new Mat();
        dst16norm.convertTo(dst8, CV_8UC3);
        
        return dst8;
    }
    
    public void saveRGB(String path){
        
        ArrayList<Mat> rgblist = new ArrayList();
        Mat blue = new Mat();
        turnedhyperimage.get(105).convertTo(blue, CV_8U, 255);
        rgblist.add(blue);
        Mat green = new Mat();
        turnedhyperimage.get(140).convertTo(green, CV_8U, 255);
        rgblist.add(green);
        Mat red = new Mat();
        turnedhyperimage.get(250).convertTo(red, CV_8U, 255);
        rgblist.add(red);

        Mat dst = new Mat();
        Core.merge(rgblist, dst);
        Mat dst8 = new Mat();
        dst.convertTo(dst8, CV_8UC3);

        Highgui.imwrite(path, dst8);
    }
    
    public void writeAllFrames(ArrayList<Mat> hyperimage, String path){    
    
    for(int p=0; p<depth; p++){
    
    Mat img3 = new Mat();
    hyperimage.get(p).convertTo(img3, CV_16U, 65535);
    Highgui.imwrite(path + "/"  + p + ".tif", img3);
    }
    
    }
    
    public Mat getFrame(int number){
        return turnedhyperimage.get(number);
    }
    
    public static void displayImage(Mat img){

            try {
                BufferedImage image = null;
                MatOfByte matOfByte = new MatOfByte();
                Highgui.imencode(".jpg", img, matOfByte);
                byte[] byteArray = matOfByte.toArray();
                InputStream in = new ByteArrayInputStream(byteArray);
                image = ImageIO.read(in);
                ImageIcon icon=new ImageIcon(image);
                JFrame frame = new JFrame("Image");
                frame.getContentPane().setLayout(new FlowLayout());
                frame.getContentPane().add(new JLabel(icon));
                frame.pack();    
                frame.setVisible(true);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            } catch (IOException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            }
    }
    
    public ArrayList<Mat> doPCA(int nPCs){
        
        pcaCube = new ArrayList();
        
        System.out.println("Hyperimage size:" + turnedhyperimage.size());
        System.out.println("Imageframes:" + imageframes);
        System.out.println("Depth" + depth);
        
        Mat pcamatrix = new Mat(depth, width*imageframes, CV_32F);

            for(int q=0; q<depth; q++){
                float[] wavepixels = new float[width*imageframes];
                turnedhyperimage.get(q).get(0,0, wavepixels);
                pcamatrix.put(q, 0, wavepixels);
            }

            Mat t_pcamatrix = pcamatrix.t();

            Mat pcaprojection = new Mat(width*imageframes, depth, CV_32F);

            Mat mean = new Mat();
            eigenvectors = new Mat();

            Core.PCACompute(t_pcamatrix, mean, eigenvectors, nPCs);

            Core.PCAProject(t_pcamatrix, mean, eigenvectors, pcaprojection);

            Mat t_pcaprojection = pcaprojection.t();

            for(int p=0; p<nPCs; p++){

                Mat pcaimage = new Mat(imageframes, width, CV_32F);
                float[] pcapixels = new float[width*imageframes];
                t_pcaprojection.get(p,0, pcapixels);
                pcaimage.put(0, 0, pcapixels);
                Mat normpcaimage = new Mat(imageframes, width, CV_16U);
                Core.normalize(pcaimage, normpcaimage, 0, 65535, NORM_MINMAX);

                pcaCube.add(normpcaimage);

            }

            return pcaCube;
    }
    
    public Mat getEigenvectors(){
        return eigenvectors;
    }
    
    public Spectrum getSpectrum(Point p){
        
        float[] values = new float[depth];
        
        for(int i=0; i<depth; i++){
            float[] currentpixel = new float[1];
            turnedhyperimage.get(i).get((int)p.y, (int)p.x, currentpixel);
            values[i] = currentpixel[0];
        }
        
        return(new Spectrum(wavelengths, values));
        
        
    }
    
    private void readHDR(String calibfile, boolean calib){
        
        ArrayList wavelengthsarraylist = new ArrayList();
        Charset charset = Charset.forName("US-ASCII");
        String file = calibfile.substring(0, calibfile.length()-3) + "hdr";
        Path filepath = Paths.get(file);
        try (BufferedReader reader = Files.newBufferedReader(filepath, charset)) {
        String line = null;
        
        line=reader.readLine();
        if(line.contains("ENVI")){
            readENVI(calibfile, calib);
        }
        else if(line.contains("BYTEORDER")){
            readPSI(calibfile, calib);
        }
        
        } catch (IOException x) {
            System.err.format("IOException: %s%n", x);
        }
        
    }
    
    private void readPSI(String calibfile, boolean calib){
        
        ArrayList wavelengthsarraylist = new ArrayList();
        Charset charset = Charset.forName("US-ASCII");
        String file = calibfile.substring(0, calibfile.length()-3) + "hdr";
        Path filepath = Paths.get(file);
        try (BufferedReader reader = Files.newBufferedReader(filepath, charset)) {
        String line = null;
        
        while (!(line=reader.readLine()).contains("NROWS")) {
        //do nothing
        }
        
        String[] rows = line.split(" ");
        if(calib){
            calibimageframes = Integer.parseInt(rows[1]);
        } else {
        imageframes = Integer.parseInt(rows[1]);
        }
        
        while (!(line=reader.readLine()).contains("NCOLS")) {
        //do nothing
        }
        
        String[] cols = line.split(" ");
        if(calib){
        calibwidth = Integer.parseInt(cols[1]);
        } else {
            width = Integer.parseInt(cols[1]);
        }
        while (!(line=reader.readLine()).contains("NBANDS")) {
        //do nothing
        }
        
        String[] bands = line.split(" ");
       
        
        if(calib){
            calibdepth = Integer.parseInt(bands[1]);
        } else {
        depth = Integer.parseInt(bands[1]);
        }
        
        while ((!reader.readLine().equalsIgnoreCase("WAVELENGTHS"))) {
            
        }
        
       while(!(line=reader.readLine()).equalsIgnoreCase("WAVELENGTHS_END")){
           
           wavelengthsarraylist.add(Float.parseFloat(line));
           //System.out.println(line);
           
        } 
        } catch (IOException x) {
            System.err.format("IOException: %s%n", x);
        }
        
        float[] wavelengtharray =  new float[wavelengthsarraylist.size()];
        
        for(int i=0; i<wavelengthsarraylist.size(); i++){
            wavelengtharray[i] = (float) wavelengthsarraylist.get(i);
        }
        
        if(calib){
            calibwavelengths = wavelengtharray;
        } else {
        wavelengths = wavelengtharray;
        }
    }

    private void readENVI(String calibfile, boolean calib){
        
        ArrayList wavelengthsarraylist = new ArrayList();
        Charset charset = Charset.forName("US-ASCII");
        String file = calibfile.substring(0, calibfile.length()-3) + "hdr";
        Path filepath = Paths.get(file);
        try (BufferedReader reader = Files.newBufferedReader(filepath, charset)) {
        String line = null;
        
        while (!(line=reader.readLine()).contains("samples")) {
        //do nothing
        }
        
        String[] cols = line.split(" = ");
        if(calib){
        calibwidth = Integer.parseInt(cols[1]);
        } else {
            width = Integer.parseInt(cols[1]);
        }
        
        while (!(line=reader.readLine()).contains("lines")) {
        //do nothing
        }
        
        String[] rows = line.split(" = ");
        if(calib){
            calibimageframes = Integer.parseInt(rows[1]);
        } else {
        imageframes = Integer.parseInt(rows[1]);
        }
        
        while (!(line=reader.readLine()).contains("bands")) {
        //do nothing
        }
        
        String[] bands = line.split(" = ");
               
        if(calib){
            calibdepth = Integer.parseInt(bands[1]);
        } else {
        depth = Integer.parseInt(bands[1]);
        }
        
        while (!(line=reader.readLine()).contains("header offset")) {
        //do nothing
        }
        
        String offsetstring  = line.split(" = ")[1];
        offset = Integer.parseInt(offsetstring);
        
        while (!(line=reader.readLine()).contains("wavelength  = ")) {
            
        }
        
        String waves = line.split(" = ")[1];
        
        waves = waves.substring(0, waves.length()-1);
        waves = waves.substring(1, waves.length()-1);
        String[] wavesarray = waves.split(", ");
       
       for(String wave : wavesarray){    
           wavelengthsarraylist.add(Float.parseFloat(wave));
        } 
        
        } catch (IOException x) {
            System.err.format("IOException: %s%n", x);
        }
        
        float[] wavelengtharray =  new float[wavelengthsarraylist.size()];
        
        for(int i=0; i<wavelengthsarraylist.size(); i++){
            wavelengtharray[i] = (float) wavelengthsarraylist.get(i);
        }
        
        if(calib){
            calibwavelengths = wavelengtharray;
        } else {
        wavelengths = wavelengtharray;
        
    }
    }

    
    /**
     * @return the wavelengths
     */
    public float[] getWavelengths() {
        return wavelengths;
    }
    
    private void getSpectraFromGUI(ArrayList<Point> class1, ArrayList<Point> class2){
        int totalnumber = class1.size() + class2.size();
       
        
        class1spectra = new ArrayList();
        class2spectra = new ArrayList();
        
        for(int i =0; i<class1.size(); i++){
            class1spectra.add(getSpectrum(class1.get(i)));
        }
        
        for(int j = 0; j<class2.size(); j++){
            class2spectra.add(getSpectrum(class2.get(j)));
        }
        
    }
    
    private void createTrainingMatrix(){
        int totalnumber = class1spectra.size() + class2spectra.size();
        
        trainingMatrix = new Mat(totalnumber, depth, CV_32F);
        responseMatrix = new Mat(totalnumber, 1, CV_32S);
                
        for(int i =0; i<class1spectra.size(); i++){
            trainingMatrix.put(i,0,class1spectra.get(i).getValues());
            responseMatrix.put(i,0,1);
        }
        
        for(int j =0; j<class2spectra.size(); j++){
            trainingMatrix.put(j+class1spectra.size(),0,class2spectra.get(j).getValues());
            responseMatrix.put(j+class1spectra.size(),0,0);
        }
        
        Mat normTraining = new Mat();
        Core.normalize(trainingMatrix, normTraining, 0, 255, NORM_MINMAX, CV_8U);
        Highgui.imwrite("C:/HyperSpec/trainingmatrix.tif", normTraining);
        
    }
    
      
    public boolean trainModel(ArrayList<Point> class1, ArrayList<Point> class2){
        
        getSpectraFromGUI(class1, class2);
        createTrainingMatrix();
        
        int totalnumber = class1spectra.size() + class2spectra.size();     
        
        /*model = Boost.create();
        ((Boost)model).setBoostType(1);
        ((Boost)model).setMaxDepth(50);
        ((Boost)model).setWeakCount(200);
        ((Boost)model).setWeightTrimRate(0.5);
        */
        
        model = new CvKNearest();
        
        
        boolean trained = ((CvKNearest)model).train(trainingMatrix, responseMatrix);
        
        if(trained){
            return true;
        } else {
            return false;
        }
        
    }
    
    public boolean trainModel(Mat trainingMatrix, Mat responseMatrix){
        
        model = new CvKNearest();
        
        boolean trained = ((CvKNearest)model).train(trainingMatrix, responseMatrix);
        
        if(trained){
            return true;
        } else {
            return false;
        }
        
    }
    
    /*public Mat predictImage(){                //fast version for smaller images
        
        classifiedImage = new Mat(imageframes, width, CV_32F);
        
         Mat pcamatrix = new Mat(depth, width*imageframes, CV_32F);

            for(int q=0; q<depth; q++){
                float[] wavepixels = new float[width*imageframes];
                turnedhyperimage.get(q).get(0,0, wavepixels);
                pcamatrix.put(q, 0, wavepixels);
            }

            Mat t_pcamatrix = pcamatrix.t();

            Mat results = new Mat();
            ((CvKNearest)model).find_nearest(t_pcamatrix, 5, results, new Mat(), new Mat());
            
            float[] resultarray = new float[imageframes*width];
            results.get(0,0, resultarray);
            
            classifiedImage = new Mat(imageframes, width, CV_32F);
            classifiedImage.put(0, 0, resultarray);    
                
        Mat classifiedImage16 =  new Mat();
        classifiedImage.convertTo(classifiedImage16, CV_16U);
        //System.out.println(classifiedImage.get(100, 100)[0]);
        
        Highgui.imwrite("C://HyperSpec//Classified.tif", classifiedImage16);
        
        return classifiedImage16;
        
    }*/
    
    /*public Mat predictImage(){          //slow version for large images that overflow signed int type 
        
        classifiedImage = new Mat(imageframes, width, CV_32F);
        
        if((long)width*(long)imageframes*(long)depth*4 <Integer.MAX_VALUE){
            
            Mat pcamatrix = new Mat(depth, width*imageframes, CV_32F);

            for(int q=0; q<depth; q++){
                float[] wavepixels = new float[width*imageframes];
                turnedhyperimage.get(q).get(0,0, wavepixels);
                pcamatrix.put(q, 0, wavepixels);
            }

            Mat t_pcamatrix = pcamatrix.t();

            Mat results = new Mat();
            ((CvKNearest)model).find_nearest(t_pcamatrix, 5, results, new Mat(), new Mat());
            
            float[] resultarray = new float[imageframes*width];
            results.get(0,0, resultarray);
            
            classifiedImage = new Mat(imageframes, width, CV_32F);
            classifiedImage.put(0, 0, resultarray); 
            } else {

                for(int r=0; r<width; r++){
                        for(int s=0; s<imageframes; s++){
                            Mat pcamatrix = new Mat(depth, 1, CV_32F);
                            for(int q=0; q<depth; q++){
                                float[] buf = new float[1];
                                turnedhyperimage.get(q).get(s, r, buf);
                                pcamatrix.put(q, 0, buf);
                            }
                           Mat t_pcamatrix = pcamatrix.t();
                           Mat results = new Mat();
                            ((CvKNearest)model).find_nearest(t_pcamatrix, 5, results, new Mat(), new Mat());
                            classifiedImage.put(s, r, (float)results.get(0, 0)[0]);  
                            
                        }
                }
        }

        Mat classifiedImage16 =  new Mat();
        classifiedImage.convertTo(classifiedImage16, CV_16U);
        //System.out.println(classifiedImage.get(100, 100)[0]);
        
        Highgui.imwrite("C://HyperSpec//Classified.tif", classifiedImage16);
        
        return classifiedImage16;
            
        
    }*/
    
    public Mat predictImage(){          //multithreaded prediction function 
        
        ForkPredict threadPredictor = new ForkPredict(turnedhyperimage, model); //instantiate a ThreadPredictor class to do the predictions in parallel
              
        Mat classifiedImage = threadPredictor.invoke();                         //invoke the class 
        
        return classifiedImage;
    }
    
    public CvStatModel getModel(){
        return model;
    }
    
    public void saveModel(){
        JFileChooser fc = new JFileChooser();
        int returnVal = fc.showSaveDialog(null);
        String file = fc.getSelectedFile().getAbsolutePath();
        String savepath = file;
        //((CvKNearest)model).save(savepath);
        //((CvKNearest)model).save("C:/HyperSpec/model.xml", "name");
        writeModelToXML(trainingMatrix, responseMatrix, wavelengths, savepath);
        //readWriteXML.writeClassesToXML(responseMatrix, savepath);
        
    }
    
    public void loadModelFromXML(String path) {							//function to parse a XML file containing test cases
      
      Mat trainingMatrix;
      Mat responseMatrix;
      float[] wavelengths;
      NodeList nList;
      
    try {

	File fXmlFile = new File(path);											//define a File path
	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();				//define a DocumentBuilder DOM parser
	Document doc = dBuilder.parse(fXmlFile);								//parse the XML file

	doc.getDocumentElement().normalize();									//normalise document structure
        
	nList = doc.getElementsByTagName("sample");								//get all case elements
        
        int nWavelengths = ((Element)nList.item(0)).getElementsByTagName("spectrum").item(0).getChildNodes().getLength();
        
        int nSamples = nList.getLength();
        
        wavelengths = new float[nWavelengths];
        
        NodeList currentspectrum = ((Element)nList.item(0)).getElementsByTagName("spectrum");
        
        for(int j=0; j<nWavelengths; j++){
                wavelengths[j] = Float.parseFloat(((Element)currentspectrum.item(0)).getElementsByTagName("wavelength").item(j).getAttributes().item(0).getTextContent());
            }
        
        trainingMatrix = new Mat(nSamples, nWavelengths, CV_32F);
        responseMatrix = new Mat(nSamples, 1, CV_32S);
        
        for(int i=0; i<nSamples; i++){
            Node currentSpectrumElement = ((Element)nList.item(i)).getElementsByTagName("spectrum").item(0);
            float[] currentFloatSpectrum = new float[nWavelengths];
                    
            for(int j=0; j<nWavelengths; j++){
                currentFloatSpectrum[j] = Float.parseFloat(currentSpectrumElement.getChildNodes().item(j).getTextContent());
            }
            trainingMatrix.put(i,0, currentFloatSpectrum);
            Node currentResponseElement = ((Element)nList.item(i)).getElementsByTagName("response").item(0);
            int[] c = new int[1];
            c[0] = Integer.parseInt(((Element)currentResponseElement).getTextContent());
            responseMatrix.put(i, 0, c);
        }
        
    } catch (ParserConfigurationException | SAXException | IOException e) {	//catch IO exceptions
	e.printStackTrace();  
        throw new RuntimeException(e);
    }
    trainModel(trainingMatrix, responseMatrix);
  }
  
  public static void writeModelToXML(Mat trainingMatrix, Mat responseMatrix, float[] wavelengths, String path) {							//function to parse a XML file containing test cases
      
    try {

	DocumentBuilderFactory dbFactory =
         DocumentBuilderFactory.newInstance();
         DocumentBuilder dBuilder = 
            dbFactory.newDocumentBuilder();
         Document doc = dBuilder.newDocument();
         // root element
         Element rootElement = doc.createElement("samples");
         doc.appendChild(rootElement);							//get all case elements
         
         for(int i=0; i<trainingMatrix.rows(); i++){
         Element sample = doc.createElement("sample");
         Element spectrum = doc.createElement("spectrum");
                for(int j=0; j<trainingMatrix.cols(); j++){                
                 Element wavelength = doc.createElement("wavelength");
                Attr attr = doc.createAttribute("wavelength");
		attr.setValue(Float.toString(wavelengths[j]));
		wavelength.setAttributeNode(attr);
                
                float[] a = new float[1];
                trainingMatrix.get(i, j, a);
                //wavelength.appendChild(doc.createElement("value"));
                wavelength.appendChild(doc.createTextNode(Float.toString(a[0])));
		spectrum.appendChild(wavelength);
                }
                sample.appendChild(spectrum);
                Element response = doc.createElement("response");
                int[] b = new int[1];
                responseMatrix.get(i, 0, b);
                
                response.appendChild(doc.createTextNode(Integer.toString(b[0])));
                sample.appendChild(response);
                rootElement.appendChild(sample);
         }
         
         TransformerFactory transformerFactory = TransformerFactory.newInstance();
		Transformer transformer = transformerFactory.newTransformer();
		DOMSource source = new DOMSource(doc);
                
		StreamResult result = new StreamResult(new File(path + ".xml"));

		// Output to console for testing
		// StreamResult result = new StreamResult(System.out);

		transformer.transform(source, result);
         
    } catch (ParserConfigurationException e) {	//catch IO exceptions
	e.printStackTrace();  
        throw new RuntimeException(e);
    }       catch (TransformerConfigurationException ex) { 
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } catch (TransformerException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } 
    
  }
  
  public LDAModel loadLDAFromXML(String path) {							//function to parse a XML file containing test cases
      
      float[] coefficients;
      float[] xmeans;
      NodeList nList;
      
    try {

	File fXmlFile = new File(path);											//define a File path
	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();				//define a DocumentBuilder DOM parser
	Document doc = dBuilder.parse(fXmlFile);								//parse the XML file

	doc.getDocumentElement().normalize();									//normalise document structure
        
	nList = doc.getElementsByTagName("LDAModel");								//get all case elements
        
        NodeList currentspectrum = ((Element)nList.item(0)).getElementsByTagName("coefficients");
        NodeList currentxmeans = ((Element)nList.item(0)).getElementsByTagName("xmeans");
        
        int nWavelengths = ((Element)currentspectrum.item(0)).getElementsByTagName("wavelength").getLength();
        
        coefficients = new float[nWavelengths];
        xmeans = new float[nWavelengths];
        
        for(int j=0; j<nWavelengths; j++){
            coefficients[j] = Float.parseFloat(((Element)currentspectrum.item(0)).getElementsByTagName("wavelength").item(j).getTextContent());
            xmeans[j] = Float.parseFloat(((Element)currentxmeans.item(0)).getElementsByTagName("wavelength").item(j).getTextContent());
            }
        
        float[] wavelengthRange = Arrays.copyOfRange(wavelengths, wavelengths.length-nWavelengths, wavelengths.length);
        return new LDAModel(wavelengthRange, coefficients, xmeans);
         
    } catch (ParserConfigurationException | SAXException | IOException e) {	//catch IO exceptions
	e.printStackTrace();  
        throw new RuntimeException(e);
    }
    
  }
  
  private void writeSpectraToCSV(Mat trainingMatrix, ArrayList<String> labels, float[] wavelengths, String path){
      
      
            try {
                CSVWriter writer = new CSVWriter(new FileWriter(path + ".csv"));
                
                 for(int j=0; j<trainingMatrix.rows(); j++){ 
                     
                    float[] a = new float[trainingMatrix.cols()];
                    trainingMatrix.get(j, 0, a);
                    
                    
                    String[] floatStringArray = new String[a.length+1];
                    
                    for(int f=0;f<a.length;f++){
                        floatStringArray[f] = Float.toString(a[f]);
                    }
                                        
                    floatStringArray[a.length] = labels.get(j);
                    writer.writeNext(floatStringArray);
                 }
                 
                 writer.close();
                
            } catch (IOException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            }
      
  }
  
  private void writeSpectraToXML(Mat trainingMatrix, ArrayList<String> labels, float[] wavelengths, String path) {							//function to parse a XML file containing test cases
      
      
    try {

	DocumentBuilderFactory dbFactory =
         DocumentBuilderFactory.newInstance();
         DocumentBuilder dBuilder = 
            dbFactory.newDocumentBuilder();
         Document doc = dBuilder.newDocument();
         // root element
         Element rootElement = doc.createElement("samples");
         doc.appendChild(rootElement);							//get all case elements
         
         for(int i=0; i<trainingMatrix.rows(); i++){
         Element sample = doc.createElement("sample");
         Element spectrum = doc.createElement("spectrum");
                for(int j=0; j<trainingMatrix.cols(); j++){                
                 Element wavelength = doc.createElement("wavelength");
                Attr attr = doc.createAttribute("wavelength");
		attr.setValue(Float.toString(wavelengths[j]));
		wavelength.setAttributeNode(attr);
                
                float[] a = new float[1];
                trainingMatrix.get(i, j, a);
                //wavelength.appendChild(doc.createElement("value"));
                wavelength.appendChild(doc.createTextNode(Float.toString(a[0])));
		spectrum.appendChild(wavelength);
                }
                sample.appendChild(spectrum);
                Element response = doc.createElement("response");
                
                response.appendChild(doc.createTextNode(labels.get(i)));
                sample.appendChild(response);
                rootElement.appendChild(sample);
         }
         
         TransformerFactory transformerFactory = TransformerFactory.newInstance();
		Transformer transformer = transformerFactory.newTransformer();
		DOMSource source = new DOMSource(doc);
               
		StreamResult result = new StreamResult(new File(path + ".xml"));

		// Output to console for testing
		// StreamResult result = new StreamResult(System.out);

		transformer.transform(source, result);
         
    } catch (ParserConfigurationException e) {	//catch IO exceptions
	e.printStackTrace();  
        throw new RuntimeException(e);
    }       catch (TransformerConfigurationException ex) { 
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } catch (TransformerException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } 
    
  }
  
  public void exportSpectraFromClassifiedGrid(Mat predictedimage){
      
      Mat image8 = new Mat();
      predictedimage.convertTo(image8, CV_8U, 255.0);
    
        int rows = 4;
        int cols = 5;
        
        int topleft_x = (int)(0.11F*(float)width);
        int topleft_y = (int)(0.10F*(float)imageframes);;
        
        int roi_width = (int)(0.13F*(float)width);
        int roi_height = (int)(0.1625F*(float)imageframes);
        int hgap = (int)(0.026F*(float)width);
        int vgap = (int)(0.0525*(float)imageframes);
               
        ArrayList<String> pots = new ArrayList();
        
        pots.add("D5");
        pots.add("D4");
        pots.add("D3");
        pots.add("D2");
        pots.add("D1");
        pots.add("C5");
        pots.add("C4");
        pots.add("C3");
        pots.add("C2");
        pots.add("C1");
        pots.add("B5");
        pots.add("B4");
        pots.add("B3");
        pots.add("B2");
        pots.add("B1");
        pots.add("A5");
        pots.add("A4");
        pots.add("A3");
        pots.add("A2");
        pots.add("A1");
        
        ArrayList<Rect> Rois = new ArrayList();
        
        for(int r=0; r<rows; r++){
            
            for(int c=0; c<cols; c++){
                
                int height1 = roi_height;
                int width1 = roi_width;
                int xext, yext;
                
                xext = (topleft_x+c*(roi_width+hgap)+roi_width);
                yext = (topleft_y+(r*(roi_height+vgap))+roi_height);
                
                //System.out.println("Xext: " + xext);
                //System.out.println("Yext: " + yext);
                

                if(yext>predictedimage.rows()){
                    height1 = roi_height - (yext-predictedimage.rows());
                }
                
                if(xext>predictedimage.cols()){
                    width1 = roi_width - (xext-predictedimage.cols());
                }
                
                if(height1 <= 0 || width1 <= 0){
                    Rois.add(new Rect());
                } else{
                Rois.add(new Rect((topleft_x+c*(roi_width+hgap)), (topleft_y+(r*(roi_height+vgap))),  width1, height1));
                }
            }
        }
        
        ArrayList<Mat> allPotSpectra = new ArrayList();
                
        for(int i =0; i<Rois.size(); i++){
             
            Mat current = new Mat(predictedimage, Rois.get(i));
            Mat currentspectra = new Mat(Core.countNonZero(current), wavelengths.length, CV_32F);
            //Highgui.imwrite("C:/HyperSpec/"+ pots.get(i)+ ".tif", current);
            int count = 0;
            
            int roi_X = Rois.get(i).x;
            int roi_Y = Rois.get(i).y;
            
            for (int k = roi_Y; k < roi_Y+roi_height; ++k)
            {
                for (int j = roi_X; j < roi_X+roi_width; ++j)
                {
                    short[] label = new short[1]; 
                    predictedimage.get(k, j, label);
                    //System.out.println("Label: " + label[0]);
                    if (label[0] == 0)
                    {
                        continue;   // No contour
                    }
                    else
                    {
                                        
                    Spectrum spectrum = getSpectrum(new Point(j, k));
                    currentspectra.put(count++, 0, spectrum.getValues());
                    }
                }
            }
            
            allPotSpectra.add(currentspectra);
            Core.rectangle(image8, new Point(Rois.get(i).x, Rois.get(i).y), new Point(Rois.get(i).x + Rois.get(i).width, Rois.get(i).y + Rois.get(i).height), new Scalar(255));
        }
    
    for(int m=0; m<allPotSpectra.size(); m++){    
        Mat normTraining = new Mat();
        Core.normalize(allPotSpectra.get(m), normTraining, 0, 255, NORM_MINMAX, CV_8U);
        
        Highgui.imwrite(BILFile + pots.get(m) +".tif", normTraining);
        
        String[] file = BILFile.split("\\\\");
        String filestring = file[file.length-1] + "_" + pots.get(m);
        ArrayList<String> responseArray = new ArrayList();
        
        for(int p=0; p<normTraining.rows(); p++){
            responseArray.add(filestring);
        }
        
        writeSpectraToCSV(allPotSpectra.get(m), responseArray, wavelengths, BILFile + "_"+ pots.get(m));
                
    }
    System.out.println(BILFile + "_" + "Predicted");
    
    Highgui.imwrite(BILFile + "_" + "Predicted.jpg", image8);
    
  }
 
  public void clusterSpectraOnClassifiedGrid(Mat predictedimage){
      
      Mat image8 = new Mat();
      predictedimage.convertTo(image8, CV_8U, 255.0);
    
        int rows = 4;
        int cols = 5;
        
        int topleft_x = (int)(0.11F*(float)width);
        int topleft_y = (int)(0.10F*(float)imageframes);;
        
        int roi_width = (int)(0.13F*(float)width);
        int roi_height = (int)(0.1625F*(float)imageframes);
        int hgap = (int)(0.026F*(float)width);
        int vgap = (int)(0.0525*(float)imageframes);
               
        ArrayList<String> pots = new ArrayList();
        
        pots.add("D5");
        pots.add("D4");
        pots.add("D3");
        pots.add("D2");
        pots.add("D1");
        pots.add("C5");
        pots.add("C4");
        pots.add("C3");
        pots.add("C2");
        pots.add("C1");
        pots.add("B5");
        pots.add("B4");
        pots.add("B3");
        pots.add("B2");
        pots.add("B1");
        pots.add("A5");
        pots.add("A4");
        pots.add("A3");
        pots.add("A2");
        pots.add("A1");
        
        ArrayList<Rect> Rois = new ArrayList();
        
        for(int r=0; r<rows; r++){
            
            for(int c=0; c<cols; c++){
                
                int height1 = roi_height;
                int width1 = roi_width;
                int xext, yext;
                
                xext = (topleft_x+c*(roi_width+hgap)+roi_width);
                yext = (topleft_y+(r*(roi_height+vgap))+roi_height);
                
                //System.out.println("Xext: " + xext);
                //System.out.println("Yext: " + yext);
                

                if(yext>predictedimage.rows()){
                    height1 = roi_height - (yext-predictedimage.rows());
                }
                
                if(xext>predictedimage.cols()){
                    width1 = roi_width - (xext-predictedimage.cols());
                }
                
                if(height1 <= 0 || width1 <= 0){
                    Rois.add(new Rect());
                } else{
                Rois.add(new Rect((topleft_x+c*(roi_width+hgap)), (topleft_y+(r*(roi_height+vgap))),  width1, height1));
                }
            }
        }
        
        ArrayList<Mat> allPotSpectra = new ArrayList();
                
        for(int i =0; i<Rois.size(); i++){
             
            Mat current = new Mat(predictedimage, Rois.get(i));
            Mat currentspectra = new Mat(Core.countNonZero(current), wavelengths.length, CV_32F);
            //Highgui.imwrite("C:/HyperSpec/"+ pots.get(i)+ ".tif", current);
            int count = 0;
            
            int roi_X = Rois.get(i).x;
            int roi_Y = Rois.get(i).y;
            
            for (int k = roi_Y; k < roi_Y+roi_height; ++k)
            {
                for (int j = roi_X; j < roi_X+roi_width; ++j)
                {
                    short[] label = new short[1]; 
                    predictedimage.get(k, j, label);
                    //System.out.println("Label: " + label[0]);
                    if (label[0] == 0)
                    {
                        continue;   // No contour
                    }
                    else
                    {
                                        
                    Spectrum spectrum = getSpectrum(new Point(j, k));
                    currentspectra.put(count++, 0, spectrum.getValues());
                    }
                }
            }
            
            allPotSpectra.add(currentspectra);
            Core.rectangle(image8, new Point(Rois.get(i).x, Rois.get(i).y), new Point(Rois.get(i).x + Rois.get(i).width, Rois.get(i).y + Rois.get(i).height), new Scalar(255));
        }
    
        ArrayList<Mat> allLabels = new ArrayList();
        
    for(int m=0; m<allPotSpectra.size(); m++){    
        Mat normTraining = new Mat();
        Core.normalize(allPotSpectra.get(m), normTraining, 0, 255, NORM_MINMAX, CV_8U);
        
        Highgui.imwrite(BILFile + pots.get(m) +".tif", normTraining);
        
        String[] file = BILFile.split("\\\\");
        String filestring = file[file.length-1] + "_" + pots.get(m);
        ArrayList<String> responseArray = new ArrayList();
        
        for(int p=0; p<normTraining.rows(); p++){
            responseArray.add(filestring);
        }
        
        Mat labels = new Mat();
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT,100,1);
        Mat centers = new Mat();
        
        //System.out.println("m=" + m);
        
        
        if(allPotSpectra.get(m).rows() > 2){
        Core.kmeans(allPotSpectra.get(m), 3, labels, criteria, 1, Core.KMEANS_PP_CENTERS, centers);        
        }
        
        allLabels.add(labels);

    }
    
    Mat clusteredImage = new Mat(predictedimage.rows(), predictedimage.cols(), CV_32S, new Scalar(0));
    
    for(int i =0; i<Rois.size(); i++){
             
            Mat current = new Mat(predictedimage, Rois.get(i));
            Mat currentLabels = allLabels.get(i);
            
            int count = 0;
            
            int roi_X = Rois.get(i).x;
            int roi_Y = Rois.get(i).y;
            
            for (int k = roi_Y; k < roi_Y+roi_height; ++k)
            {
                for (int j = roi_X; j < roi_X+roi_width; ++j)
                {
                    short[] label = new short[1]; 
                    predictedimage.get(k, j, label);
                    //System.out.println("Label: " + label[0]);
                    if (label[0] == 0)
                    {
                        continue;   // No contour
                    }
                    else
                    {
                    
                    int[] cluster = new int[1];
                    currentLabels.get(count++, 0, cluster);
                    clusteredImage.put(k, j, cluster[0]+1);
                    
                    }
                }
            }
            
        }
    
    Mat clusteredImage8 = new Mat();
    clusteredImage.convertTo(clusteredImage8, CV_8U);
    Mat clusteredImage8Norm = new Mat(imageframes, width, CV_8U);
    
    Core.normalize(clusteredImage8, clusteredImage8Norm, 0, 255, NORM_MINMAX, CV_8U, image8);
    
    Highgui.imwrite(BILFile + "_" + "Clustered.tif", clusteredImage8Norm);
    
  }
  
  public Mat visualizeLDA(LDAModel model){
      
      int start = (wavelengths.length-model.getDepth());
      
      ArrayList<Mat> truncatedHyperImage = new ArrayList<Mat>(turnedhyperimage.subList(start, wavelengths.length));
      
      Mat average = new Mat(imageframes, width, CV_32F, new Scalar(0.0));
            
      float[] modelcoefs = model.getCoefficients();
      float[] modelxmeans = model.getXmeans();
      
      for(int i=0; i<truncatedHyperImage.size(); i++){
          
          Mat currentCoefImage = new Mat();
          Core.multiply(truncatedHyperImage.get(i), new Scalar(modelcoefs[i]), currentCoefImage);
          Core.add(currentCoefImage, new Scalar(modelxmeans[i]), currentCoefImage);
          Core.add(currentCoefImage, average, average);
      }
      return average;
  }
  
  private Mat averageImage(ArrayList<Mat> hyperimage){
      
      Mat average = new Mat(imageframes, width, CV_32F);
      
      for(int i=0; i<hyperimage.size(); i++){
          
          Core.add(average, hyperimage.get(i), average);
          
      }
      
      Core.divide(average, new Scalar(hyperimage.size()), average);
      
      return(average);
  }
  
  public void subsampleWavelengths(int interval, boolean bin){
      
      ArrayList<Mat> newHyperImage = new ArrayList(depth/interval);
      float[] newWavelengths = new float[depth/interval];
      
      if(!bin){
          
          int ctr = 0;
          for(int i=0; i<depth; i=i+interval){
              newHyperImage.add(turnedhyperimage.get(i));
              newWavelengths[ctr++] = wavelengths[i];
          }
          
      } else {
          int ctr = 0;
          for(int i=0; i<depth; i=i+interval){
              
              ArrayList<Mat> currentBatch = new ArrayList(interval);
              for(int j=0; j<interval; j++){
                  currentBatch.add(turnedhyperimage.get(+j));
              }
              
              newHyperImage.add(averageImage(currentBatch));
              newWavelengths[ctr++] = wavelengths[i];
          }
      }
      
      turnedhyperimage = newHyperImage;
      depth = depth/interval;
      wavelengths = newWavelengths;
      
  }
  
  public void applyMask(Mat mask){
      
      ArrayList<Mat> maskedHyperImage = new ArrayList();
      
      Mat zeroOneMask = new Mat();
      mask.convertTo(zeroOneMask, CV_32F, 1/255.0);
      
      
      for(int i=0; i<depth; i++){
          Mat currentMasked = new Mat();
          Core.multiply(turnedhyperimage.get(i), zeroOneMask, currentMasked);
          maskedHyperImage.add(currentMasked);
          
      }
      
      turnedhyperimage = maskedHyperImage;
      
  }
          
    
    
}
