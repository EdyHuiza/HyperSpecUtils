/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperspecutils;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
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
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC3;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import static org.opencv.imgproc.Imgproc.BORDER_TRANSPARENT;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import org.opencv.ml.CvStatModel;
import ru.sscc.spline.Spline;
import ru.sscc.spline.polynomial.POddSplineCreator;
import ru.sscc.util.CalculatingException;

/**
 *
 * @author u4110553
 */
public class PhenoLiteReader {
    
        private int imageframes;
        private int width;
        private int depth;
        private float[] wavelengths;
        private ArrayList<Mat> turnedhyperimage;
        private ArrayList<Mat> pcaCube;
        private Mat eigenvectors;
        private CvStatModel model;
        private ArrayList<Spectrum> class1spectra;
        private ArrayList<Spectrum> class2spectra;
        private Mat trainingMatrix;
        private Mat responseMatrix;
        private Mat classifiedImage;
        private String RAWFile;
        private int header = 16;
        private int bits;
        private int gap =8;
        private ArrayList<Float> odometer;
        private ArrayList<LocalDateTime> timestamps;
        private ArrayList<Integer> encoder;
        
    public PhenoLiteReader(String RAWFile){
        
        this.turnedhyperimage = new ArrayList();
        this.RAWFile = RAWFile;
        
    }
    
    public void readRAW(String RAWFile){
        
    ArrayList<Mat> hyperimage = new ArrayList();

    DataInputStream image_data_in = null;
            try {
                
                //Loading RAW file here
                
                String imagepath = RAWFile;
                File imagef = new File(imagepath);
                image_data_in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(imagef)));
                
                
                    try {
                            short temp = image_data_in.readShort();
                            short d = Short.reverseBytes(temp);
                            width = (int)d;
                            
                            temp = image_data_in.readShort();
                            temp = image_data_in.readShort();
                            d = Short.reverseBytes(temp);
                            depth = (int)d;
                            
                            temp = image_data_in.readShort();
                            temp = image_data_in.readShort();
                            d = Short.reverseBytes(temp);
                            bits = (int)d;
                            
                            temp = image_data_in.readShort();
                            temp = image_data_in.readShort();
                            d = Short.reverseBytes(temp);
                            imageframes = (int)d;
                            
                            temp = image_data_in.readShort();
                            temp = image_data_in.readShort();
                            temp = image_data_in.readShort();
                        }
                        catch (java.io.EOFException eof) {
                            
                        } catch (IOException ex) {
                            Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    
                    System.out.println(width);
                    System.out.println(depth);
                    System.out.println(imageframes);
                    //imageframes=1000;
                    
                
                short c;
                short d;
                for(int k=0; k<imageframes; k++){
                    
                    short[] image_pixels = new short[depth*width];
                    for(int i=0; i<width*depth; i++){
                        
                        try {
                            c = image_data_in.readShort();//read 2 bytes
                            
                            d = Short.reverseBytes(c);
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
                    
                    /*Mat img32 = new Mat(depth, width, CV_32F);
                    img.convertTo(img32, CV_32F);
                    
                    Mat norm = new Mat(depth, width, CV_32F);
                    Core.divide(img32, new Scalar(65535F), norm);
                    */
                    hyperimage.add(img);
                    
                    try {
                    short temp;
                        for(int t = 0; t<gap/2; t++){
                            temp = image_data_in.readShort();
                        }
                    
                    } catch (java.io.EOFException eof) {
                            break;
                        } catch (IOException ex) {
                            Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                        }
                }
                
            } catch (FileNotFoundException ex) {
                Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
            } finally {
                try {
                    image_data_in.close();
                } catch (IOException ex) {
                    Logger.getLogger(BILReader.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
    
    for(int m=0; m<depth; m++){
        
        Mat current = new Mat(imageframes, width, CV_16U); 
        
        for(int n=0; n<imageframes; n++){
            
            short[] widthrow = new short[width];
            
            hyperimage.get(n).get(m, 0, widthrow);
           
            current.put(n, 0, widthrow);
            
        }
        
        turnedhyperimage.add(current);
        
    }
    
}

        
public Mat getRGB(){
        
        ArrayList<Mat> rgblist = new ArrayList();
        Mat blue = new Mat();
        int blueindex = 58;
        int greenindex = 73; 
        int redindex = 143;
        
        turnedhyperimage.get(blueindex).convertTo(blue, CV_8U);
        rgblist.add(blue);
        Mat green = new Mat();
        turnedhyperimage.get(greenindex).convertTo(green, CV_8U);
        rgblist.add(green);
        Mat red = new Mat();
        turnedhyperimage.get(redindex).convertTo(red, CV_8U);
        rgblist.add(red);

        Mat dst = new Mat();
        Core.merge(rgblist, dst);
        Mat dst16norm = new Mat();
        Core.normalize(dst, dst16norm, 0, 255, NORM_MINMAX);
        Mat dst8 = new Mat();
        dst16norm.convertTo(dst8, CV_8UC3);
        
        return dst8;
    }

    public void writeAllFrames(ArrayList<Mat> hyperimage, String path){    
    
    for(int p=0; p<depth; p++){
    
    Mat img3 = new Mat();
    hyperimage.get(p).convertTo(img3, CV_16U);
    Highgui.imwrite(path + "/"  + p + ".tif", img3);
    }
    
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

    public ArrayList<Mat> getHyperimage(){
        return turnedhyperimage;
    }
    
    public void readGPS(String gpspath){
        
        String line = "";
        String cvsSplitBy = ",";
        odometer = new ArrayList();
        timestamps = new ArrayList();
        encoder = new ArrayList();
        
        try (BufferedReader br = new BufferedReader(new FileReader(gpspath))) {
            int ctr =0;
            while ((line = br.readLine()) != null) {
                if(ctr++ == 0){
                    continue;
                }
                // use comma as separator
                String[] gps = line.split(cvsSplitBy);

                odometer.add(Float.parseFloat(gps[13]));
                encoder.add(Integer.parseInt(gps[12]));
                //System.out.println(Float.parseFloat(gps[13]));
                DateTimeFormatter formatter =
                      DateTimeFormatter.ISO_DATE_TIME;
                LocalDateTime date = LocalDateTime.parse(gps[1], formatter);
                timestamps.add(date);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    
    public Mat remapImage(Mat img){
        
        double[] times = new double[timestamps.size()]; 
        
        for(int i=0; i<timestamps.size(); i++){
            times[i] = (double)timestamps.get(0).until(timestamps.get(i), ChronoUnit.MILLIS);
            
        }
        
        float[] odos = new float[odometer.size()]; 
        
        for(int i=0; i<timestamps.size(); i++){
            odos[i] = odometer.get(i);
            
        }
        
        double[] clicks = new double[encoder.size()]; 
        
        for(int i=0; i<timestamps.size(); i++){
            clicks[i] = (double)encoder.get(i);
            
        }
        
        double[] newtimes = new double[imageframes];
        double interval = (times[times.length-1] - times[0])/imageframes;
        
        newtimes[0] = times[0];
        newtimes[newtimes.length-1] = times[times.length-1];
        
        for(int j=1; j<imageframes; j++){
            newtimes[j] = (newtimes[j-1] + interval);
        }
        
        double[] newclicks = new double[imageframes];
        
        try{
        Spline spl = POddSplineCreator.createSpline(2, times, clicks);
        
        for(int k=0; k<imageframes; k++){
            newclicks[k] = spl.value(newtimes[k]);
            
        }
        
        for(int l=0; l<imageframes; l++){
            newclicks[l] = (newclicks[l]/newclicks[newclicks.length-1])*imageframes;
            System.out.println("Newtimes: " + newtimes[l] + " Newclicks: " +newclicks[l]);
        }

        } catch (CalculatingException e){
            System.out.println(e.toString());
        }
        
        Mat map_x = new Mat(img.size(), CV_32FC1);
        Mat map_y = new Mat(img.size(), CV_32FC1);
        
        for(int j=0; j<img.rows(); j++){
            
            for(int i=0; i<img.cols(); i++){
                
                map_x.put(j, i, i);
                map_y.put(j, i, Math.round(newclicks[j]));

            }
        }
        
        Mat dst = new Mat(img.size(), img.type());
        Imgproc.remap(img, dst, map_x, map_y, INTER_LINEAR, BORDER_TRANSPARENT, new Scalar(0.0));
        
        return(dst);
    }
    
}
