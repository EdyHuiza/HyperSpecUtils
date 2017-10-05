/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


package hyperspecjava;




import hyperspecutils.BILReader;
import java.io.IOException;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import org.opencv.core.*;

import org.opencv.imgproc.Imgproc;

/**
 *
 * @author u4110553
 */
public class HyperSpecJava {
    
    /**
     * @param args the command line arguments
     */
    
    
    
    public static void main(String[] args)  throws IOException{
        
        //final String calibpath = args[1];
        //final String imgpath = args[0];
        
        //System.setProperty("java.library.path", "C:\\opencv\\opencv\\build\\java\\x64");
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        JFileChooser fc = new JFileChooser();
        fc.setDialogTitle("Load image file");
        int returnVal = fc.showOpenDialog(null);
        String file = fc.getSelectedFile().getAbsolutePath();
        String imgpath = file;
        System.out.println(file);
        
        JFileChooser fc_calib = new JFileChooser();
        fc_calib.setDialogTitle("Load calibration file");
        int returnVal_calib = fc_calib.showOpenDialog(null);
        String file_calib = fc_calib.getSelectedFile().getAbsolutePath();
        String calibpath = file_calib;
        
              
        BILReader currentimage = new BILReader(imgpath, calibpath, true);
        currentimage.readBIL(imgpath, calibpath);
        currentimage.subsampleWavelengths(4, false);
        
        System.out.println("Width = " + currentimage.getWidth());
        System.out.println("Imageframes = " + currentimage.getImageframes());
        System.out.println("Depth = " + currentimage.getDepth());
        System.out.println("Offset = " + currentimage.getOffset());
        System.out.println("Data type = " + currentimage.getHyperimage().get(0).depth());
        //currentimage.writeAllFrames(currentimage.getHyperimage(), "C:\\Hyperspec");
        Mat rgb = currentimage.getRGB();
        
        Mat resizeimage = new Mat();
        Size sz = new Size(1000,800);
        Imgproc.resize(rgb, resizeimage, sz );
       
        JFrame frame = new JFrame("Image Trainer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
         
        //Create and set up the content pane.
        JComponent newContentPane = new ImageTrainer(resizeimage, currentimage);
        newContentPane.setOpaque(true); //content panes must be opaque
        frame.setContentPane(newContentPane);
        frame.getContentPane().setSize(1500,1000);
        //Display the window.
        frame.pack();
        frame.setVisible(true);
        
        
    }
    
        
}
