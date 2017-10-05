/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperimageclassifier;

import hyperspecutils.BILReader;
import hyperspecutils.LDAModel;
import javax.swing.JFileChooser;
import org.opencv.core.Core;
import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.CvType.CV_16U;
import static org.opencv.core.CvType.CV_8U;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author u4110553
 */
public class HyperImageVisualiseLDA {

    /**
     * @param args the command line arguments
     */
    
    public static void main(String[] args) {
        
        final String calibpath = args[1];
        final String imgpath = args[0];
        final String modelpath = args[2];
  
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
              
        BILReader currentimage = new BILReader(imgpath, calibpath, true);
        currentimage.readBIL(imgpath, calibpath);
        
        currentimage.loadModelFromXML("C:\\HyperSpec\\model1_fullRes.xml"); //pre-classify the image with a KNN model first, then only plot the LDA scores for the plant pixels
        
        Mat predicted = currentimage.predictImage();
        Mat predicted8 = new Mat();
        
        Mat predictederoded = new Mat();
        int erosion_size = 1;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(2*erosion_size, 2*erosion_size));
        Imgproc.erode(predicted, predictederoded, element);
        predictederoded.convertTo(predicted8, CV_8U);
        
        LDAModel ldamodel = currentimage.loadLDAFromXML(modelpath);
        Mat ldaimage = currentimage.visualizeLDA(ldamodel);
        Mat normldaimage = new Mat(currentimage.getImageframes(), currentimage.getWidth(), CV_16U);
        
        Core.normalize(ldaimage, normldaimage, 0, 65535, NORM_MINMAX, -1, predicted8);
        
        Mat normldaimage16 = new Mat();
        normldaimage.convertTo(normldaimage16, CV_16U);
        
        String[] modelsplit = modelpath.split("\\\\");
        String modelname = modelsplit[modelsplit.length-1];
        
        String[] imagesplit = imgpath.split("\\\\");
        String imagename = imagesplit[imagesplit.length-1];
        
        
        Highgui.imwrite("C:\\HyperSpec\\LDAImage_" + imagename + "_"+ modelname + ".tif", normldaimage16);
    }
    
}
