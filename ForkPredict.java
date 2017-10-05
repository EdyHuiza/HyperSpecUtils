/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperspecutils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.RecursiveTask;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_16U;
import static org.opencv.core.CvType.CV_32F;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvKNearest;
import org.opencv.ml.CvStatModel;

/**
 *
 * @author u4110553
 */
public class ForkPredict extends RecursiveTask<Mat> {
    
    private final int nCPUs=4;
    ArrayList<Mat> hyperimage;
    CvStatModel model;
    Mat classifiedImage; 
    int depth, imageframes, width, start, end, range;
    
    public ForkPredict(ArrayList<Mat> hyperimage, CvStatModel model, int start, int end){       //constructor to split the hyperimage into parts and run them on separate threads
        
        this.hyperimage = hyperimage;
        this.model =  model;
        this.depth = hyperimage.size();
        this.width = hyperimage.get(0).cols();
        this.imageframes = hyperimage.get(0).rows();
        this.start=start;
        this.end=end;
        this.range = end-start;
        this.classifiedImage = new Mat(range, width, CV_32F);
    }
    
    public ForkPredict(ArrayList<Mat> hyperimage, CvStatModel model){           //constructor to pass a whole image to ForPredict
        
        //Runtime runtime = Runtime.getRuntime();
        //nCPUs = runtime.availableProcessors();
        this.hyperimage = hyperimage;
        this.model =  model;
        this.depth = hyperimage.size();
        this.width = hyperimage.get(0).cols();
        this.imageframes = hyperimage.get(0).rows();
        this.start=0;
        this.end=imageframes;
        this.range = end-start;
        this.classifiedImage = new Mat(range, width, CV_32F);
    }
    
    protected void computeDirectly() {
        
        Mat pcamatrix = new Mat(depth, width*range, CV_32F);
        
            for(int q=0; q<depth; q++){
                float[] wavepixels = new float[width*range];
                hyperimage.get(q).get(start,0, wavepixels);
                pcamatrix.put(q, 0, wavepixels);
            }

            Mat t_pcamatrix = pcamatrix.t();

            Mat results = new Mat();
            
            ((CvKNearest)model).find_nearest(t_pcamatrix, 5, results, new Mat(), new Mat());
            
            float[] resultarray = new float[imageframes*width];
            results.get(0,0, resultarray);
            
            classifiedImage.put(0, 0, resultarray); 
    }
    
    public Mat compute(){
        
        if((long)width*(long)range*(long)depth*4 <Integer.MAX_VALUE){
        
            computeDirectly();
            
        } else {
            
            ArrayList<ForkPredict> forks = new ArrayList();
            int prev=0;
            for(int i=0; i<nCPUs; i++){

                int next = prev + (int)(imageframes/nCPUs);
                if(i == nCPUs-1){
                    next = imageframes;
                }
                
                forks.add(new ForkPredict(hyperimage, model, prev, next));
                prev = next;
            }
            
            for(ForkPredict fork: forks){
                fork.fork();
            }
            
            List<Mat> results = new ArrayList();
            
            for(ForkPredict fork: forks){
                results.add((Mat) fork.join());
            }
            
            Core.vconcat(results, classifiedImage);
                       
        }
        
        Mat classifiedImage16 =  new Mat();
        classifiedImage.convertTo(classifiedImage16, CV_16U);
        //System.out.println(classifiedImage.get(100, 100)[0]);
        
        Highgui.imwrite("C://HyperSpec//Classified.tif", classifiedImage16);
        return classifiedImage16;
    }
    
}
