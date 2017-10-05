/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperimageclassifier;

import java.util.Arrays;

/**
 *
 * @author u4110553
 */
public class Spectrum {
    
    private int depth;
    private float[] values;
    private float[] wavelengths;
    
    public Spectrum(float[] wavelengths, float[] values){
        this.wavelengths =  wavelengths;
        this.values = values;
        this.depth = wavelengths.length;
    }

    /**
     * @return the depth
     */
    public int getDepth() {
        return depth;
    }

    /**
     * @return the values
     */
    public float[] getValues() {
        return values;
    }

    /**
     * @return the wavelengths
     */
    public float[] getWavelengths() {
        return wavelengths;
    }
    
    
    
    
    
}
