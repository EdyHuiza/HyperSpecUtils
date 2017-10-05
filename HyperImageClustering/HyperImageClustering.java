/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperimageclassifier;

import hyperspecutils.BILReader;
import javax.swing.JFileChooser;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_8U;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author u4110553
 */
public class HyperImageClustering {

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
        Mat rgb = currentimage.getRGB();
        
        Mat resizeimage = new Mat();
        Size sz = new Size(1000,800);
        Imgproc.resize(rgb, resizeimage, sz );
        
        /*JFileChooser fc = new JFileChooser();
        int returnVal = fc.showSaveDialog(null);
        String file = fc.getSelectedFile().getAbsolutePath();
        String modelpath = file;*/
        
        currentimage.loadModelFromXML(modelpath);
        
        Mat predicted = currentimage.predictImage();
        Mat predicted8 = new Mat();
        
        Mat predictederoded = new Mat();
        int erosion_size = 1;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(2*erosion_size, 2*erosion_size));
        Imgproc.erode(predicted, predictederoded, element);
        predictederoded.convertTo(predicted8, CV_8U, 255.0);
        //PredictedImage.main(predicted8, currentimage);
        
        currentimage.clusterSpectraOnClassifiedGrid(predictederoded);
        
    }
    
}
