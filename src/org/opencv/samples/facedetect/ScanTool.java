package org.opencv.samples.facedetect;

import java.io.FileOutputStream;
import java.util.List;

import org.opencv.android.JavaCameraView;
import org.opencv.core.Mat;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

public class ScanTool extends JavaCameraView implements PictureCallback {

    private static final String TAG = "Sample::ScanTool";
    private String mPictureFileName;

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
    
//    // Lower and Upper bounds for range checking in HSV color space
//    private Scalar mLowerBound = new Scalar(0);
//    private Scalar mUpperBound = new Scalar(0);
//    // Minimum contour area in percent for contours filtering
//    private static double mMinContourArea = 0.1;
//    // Color radius for range checking in HSV color space
//    private Scalar mColorRadius = new Scalar(25,50,50,0);
//    private Mat mSpectrum = new Mat();
//    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();

//    // Cache
//    Mat mPyrDownMat = new Mat();
//    Mat mHsvMat = new Mat();
//    Mat mMask = new Mat();
//    Mat mDilatedMask = new Mat();
//    Mat mHierarchy = new Mat();
//
//    public void setColorRadius(Scalar radius) {
//        mColorRadius = radius;
//    }
//
//    public void setHsvColor(Scalar hsvColor) {
//        double minH = (hsvColor.val[0] >= mColorRadius.val[0]) ? hsvColor.val[0]-mColorRadius.val[0] : 0;
//        double maxH = (hsvColor.val[0]+mColorRadius.val[0] <= 255) ? hsvColor.val[0]+mColorRadius.val[0] : 255;
//
//        mLowerBound.val[0] = minH;
//        mUpperBound.val[0] = maxH;
//
//        mLowerBound.val[1] = hsvColor.val[1] - mColorRadius.val[1];
//        mUpperBound.val[1] = hsvColor.val[1] + mColorRadius.val[1];
//
//        mLowerBound.val[2] = hsvColor.val[2] - mColorRadius.val[2];
//        mUpperBound.val[2] = hsvColor.val[2] + mColorRadius.val[2];
//
//        mLowerBound.val[3] = 0;
//        mUpperBound.val[3] = 255;
//
//        Mat spectrumHsv = new Mat(1, (int)(maxH-minH), CvType.CV_8UC3);
//
//        for (int j = 0; j < maxH-minH; j++) {
//            byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
//            spectrumHsv.put(0, j, tmp);
//        }
//
//        Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
//    }
//
//    public Mat getSpectrum() {
//        return mSpectrum;
//    }
//
//    public void setMinContourArea(double area) {
//        mMinContourArea = area;
//    }
//
//    public void process(Mat rgbaImage) {
//        Imgproc.pyrDown(rgbaImage, mPyrDownMat);
//        Imgproc.pyrDown(mPyrDownMat, mPyrDownMat);
//
//        Imgproc.cvtColor(mPyrDownMat, mHsvMat, Imgproc.COLOR_RGB2HSV_FULL);
//
//        Core.inRange(mHsvMat, mLowerBound, mUpperBound, mMask);
//        Imgproc.dilate(mMask, mDilatedMask, new Mat());
//
//        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
//
//        Imgproc.findContours(mDilatedMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//
//        // Find max contour area
//        double maxArea = 0;
//        Iterator<MatOfPoint> each = contours.iterator();
//        while (each.hasNext()) {
//            MatOfPoint wrapper = each.next();
//            double area = Imgproc.contourArea(wrapper);
//            if (area > maxArea)
//                maxArea = area;
//        }
//
//        // Filter contours by area and resize to fit the original image size
//        mContours.clear();
//        each = contours.iterator();
//        while (each.hasNext()) {
//            MatOfPoint contour = each.next();
//            if (Imgproc.contourArea(contour) > mMinContourArea*maxArea) {
//                Core.multiply(contour, new Scalar(4,4), contour);
//                mContours.add(contour);
//            }
//        }
//    }
//
//    public List<MatOfPoint> getContours() {
//        return mContours;
//    }
}
