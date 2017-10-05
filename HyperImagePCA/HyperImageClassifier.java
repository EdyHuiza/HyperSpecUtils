/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperimageclassifier;

import hyperspecutils.BILReader;
import java.util.ArrayList;
import javax.swing.JFileChooser;
import org.opencv.core.Core;
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
public class HyperImageClassifier {

    /**
     * @param args the command line arguments
     */
    
    public static void main(String[] args) {
        
        final String calibpath;
        final String imgpath;
        final int nPCs = 10;
        
        System.setProperty("java.library.path", "C:\\opencv\\opencv\\build\\java\\x64");
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        JFileChooser fc = new JFileChooser();
        fc.setDialogTitle("Load image file");
        int returnVal = fc.showOpenDialog(null);
        String file = fc.getSelectedFile().getAbsolutePath();
        imgpath = file;
        System.out.println(file);
        
        JFileChooser fc_calib = new JFileChooser();
        fc_calib.setDialogTitle("Load calibration file");
        int returnVal_calib = fc_calib.showOpenDialog(null);
        String file_calib = fc_calib.getSelectedFile().getAbsolutePath();
        calibpath = file_calib;
        
        BILReader currentimage = new BILReader(imgpath, calibpath, true);
        currentimage.readBIL(imgpath, calibpath);
        currentimage.subsampleWavelengths(4, false);
        
        currentimage.loadModelFromXML("C:\\HyperSpec\\masha_setaria.xml");
        
        Mat predicted = currentimage.predictImage();
        //Mat predicted = new Mat(new Size(currentimage.getImageframes(), currentimage.getWidth()), CV_16U, new Scalar(65535.0));
        Mat predicted8 = new Mat();
        
        Mat predictederoded = new Mat();
        int erosion_size = 1;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new  Size(2*erosion_size, 2*erosion_size));
        Imgproc.erode(predicted, predictederoded, element);
        predictederoded.convertTo(predicted8, CV_8U, 255.0);
        Highgui.imwrite("C:\\HyperSpecTemp\\mask.png", predicted8);
        
        currentimage.applyMask(predicted8);
        
        currentimage.writeAllFrames(currentimage.getHyperimage(), "C:\\HyperSpecTemp\\");
        
        ArrayList<Mat> pcacube = currentimage.doPCA(nPCs);
        
        System.out.println("PCA layers: " + pcacube.size());
        
        for(int i=0; i<pcacube.size(); i++){
            
            Mat pcaimage16 = new Mat();
            pcacube.get(i).convertTo(pcaimage16, CV_16U);
        
            Highgui.imwrite(imgpath + "_PC" + i+1 + ".tif", pcaimage16);
        }
        
    }
    
}
