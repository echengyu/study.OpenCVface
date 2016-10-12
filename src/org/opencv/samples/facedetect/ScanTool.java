package org.opencv.samples.facedetect;

import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.android.JavaCameraView;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

public class ScanTool extends JavaCameraView implements PictureCallback {
	
    private static final String TAG = "Sample::ScanTool";
    private String mPictureFileName;
    private boolean isFlashLightON = false;
    
    // ROI TODO
    private Rect mROIrect;
    private Point resolutionPoint;
    private int cutY0 = 0;
    private int cutY1 = 0;

    public ScanTool(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public List<String> getEffectList() {
        return mCamera.getParameters().getSupportedColorEffects();
    }

    public boolean isEffectSupported() {
        return (mCamera.getParameters().getColorEffect() != null);
    }

    public String getEffect() {
        return mCamera.getParameters().getColorEffect();
    }

    public void setEffect(String effect) {
        Camera.Parameters params = mCamera.getParameters();
        params.setColorEffect(effect);
        mCamera.setParameters(params);
    }
    
    public List<Size> getResolutionList() {
        return mCamera.getParameters().getSupportedPreviewSizes();
    }

    public void setResolution(Size resolution) {
        disconnectCamera();
        mMaxHeight = resolution.height;
        mMaxWidth = resolution.width;
        connectCamera(getWidth(), getHeight());
    }

    public Size getResolution() {
        return mCamera.getParameters().getPreviewSize();
    }

    public void takePicture(final String fileName) {
        Log.i(TAG, "Taking picture");
        this.mPictureFileName = fileName;
        // Postview and jpeg are sent in the same buffers if the queue is not empty when performing a capture.
        // Clear up buffers to avoid mCamera.takePicture to be stuck because of a memory issue
        mCamera.setPreviewCallback(null);

        // PictureCallback is implemented by the current class
        mCamera.takePicture(null, null, this);
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        Log.i(TAG, "Saving a bitmap to file");
        // The camera preview was automatically stopped. Start it again.
        mCamera.startPreview();
        mCamera.setPreviewCallback(this);

        // Write the image in a file (in jpeg format)
        try {
            FileOutputStream fos = new FileOutputStream(mPictureFileName);

            fos.write(data);
            fos.close();

        } catch (java.io.IOException e) {
            Log.e("PictureDemo", "Exception in photoCallback", e);
        }
    }
    
    // 閃光燈
    public void setCameraFlashLight(boolean isFlashLightON) {
	    Camera  camera = mCamera;
	    if (camera != null) {
	    Parameters params = camera.getParameters();
	
	    if (params != null) {
	        if (isFlashLightON) {
	            isFlashLightON = false;
	            params.setFlashMode(Parameters.FLASH_MODE_OFF);
	            camera.setParameters(params);
	            camera.startPreview();
	        } else {
	            isFlashLightON = true;
	            params.setFlashMode(Parameters.FLASH_MODE_TORCH);
	            camera.setParameters(params);
	            camera.startPreview();
	            }
	        }
	    }
    }
    
    // 閃光燈
    public void setCameraFlashLight() {
	    Camera  camera = mCamera;
	    if (camera != null) {
	    Parameters params = camera.getParameters();
	
	    if (params != null) {
	        if (isFlashLightON) {
	            isFlashLightON = false;
	            params.setFlashMode(Parameters.FLASH_MODE_OFF);
	            camera.setParameters(params);
	            camera.startPreview();
	        } else {
	            isFlashLightON = true;
	            params.setFlashMode(Parameters.FLASH_MODE_TORCH);
	            camera.setParameters(params);
	            camera.startPreview();
	            }
	        }
	    }
    }
    
    // 閃光燈
    public boolean getCameraFlashLight() {
        return isFlashLightON;
    }
    
    // ROI TODO
    public Mat setROImat(Mat mRgba, Mat mGray, Point resolution, Point offest, float zoomX, float zoomY) {
    	
    	zoomX = (zoomX > 1.0f) ? 1.0f : zoomX;
    	zoomY = (zoomY > 1.0f) ? 1.0f : zoomY; 	
    	
    	int cutXp0 = (int) Math.round((resolution.x / 2) * (1 - zoomX) + offest.x);
    	int cutXp1 = (int) Math.round((resolution.x / 2) * (1 + zoomX) + offest.x);
    	int cutYp0 = (int) Math.round((resolution.y / 2) * (1 - zoomY) + offest.y);
    	int cutYp1 = (int) Math.round((resolution.y / 2) * (1 + zoomY) + offest.y);
    	
    	Rect mROIrect = new Rect(cutXp0, cutYp0, cutXp1 - cutXp0, cutYp1 - cutYp0);
    	
//    	Log.e(TAG, String.valueOf(mROIrect.x+", "+mROIrect.y));
    	
		Mat mTmp = new Mat();
		mGray.submat(mROIrect).copyTo(mTmp);		
		Imgproc.Canny(mTmp, mTmp, 128, 255, 3, false);
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.drawContours(mGray, contours, -1, new Scalar(255, 255, 0, 255), 1, 8, new Mat(), 1, new Point());
		
		mTmp.copyTo(mGray.submat(mROIrect));
		
		return mGray;
    }
}
