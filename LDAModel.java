/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hyperspecutils;

/**
 *
 * @author u4110553
 */
public class LDAModel {
    
    private int depth;
    private float[] coefficients;
    private float[] wavelengths;
    private float[] xmeans;
    
    public LDAModel(float[] wavelengths, float[] coefficients, float[] xmeans){
        this.wavelengths =  wavelengths;
        this.coefficients = coefficients;
        this.depth = wavelengths.length;
        this.xmeans = xmeans;
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
    public float[] getCoefficients() {
        return coefficients;
    }

    /**
     * @return the wavelengths
     */
    public float[] getWavelengths() {
        return wavelengths;
    }
    
    public float[] getXmeans() {
        return xmeans;
    }
    
}
