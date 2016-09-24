package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

public class FdActivity extends Activity implements CvCameraViewListener2 {
	
    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    
    private boolean findContoursFUN = false;
    private boolean findContoursFUNtmp = false;
    
    private Tutorial3View mTutorial3View;
    private List<Size> mResolutionList;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;
    private MenuItem               mItemFindContours;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	private Mat hierarchy;
	private MatOfPoint2f approxCurve;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
//                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        InputStream is = getResources().openRawResource(R.raw.haarcascade_fullbody);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
//                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        mCascadeFile = new File(cascadeDir, "haarcascade_fullbody.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        
        mOpenCvCameraView.setMaxFrameSize(640, 480);
        
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

//        Rect[] facesArray = faces.toArray();
//        for (int i = 0; i < facesArray.length; i++)
//            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        
        if(findContoursFUN == true){ 
        	mGray = inputFrame.gray();
        	Point resolutionPoint = new Point(inputFrame.rgba().width(), inputFrame.rgba().height());
        	
        	// 二值化
    		Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2GRAY, 0);

    		// 高斯濾波器
    		Imgproc.GaussianBlur(mRgba, mRgba, new org.opencv.core.Size(3, 3), 6);

    		// 邊緣偵測
    		Imgproc.Canny(mRgba, mRgba, 360, 180);

    		// 蝕刻
    		Imgproc.erode(mRgba, mRgba, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(1, 1)));

    		// 膨脹
    		Imgproc.dilate(mRgba, mRgba, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(4, 4)));

    		contours = new ArrayList<MatOfPoint>();
    		hierarchy = new Mat();
        	
    		// 找影像輪廓		
    		Imgproc.findContours(mRgba, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
    		hierarchy.release();
    		
    		if(contours.size() != 0 &&contours.size() < 500){
    			// 劃出輪廓線
    			Imgproc.drawContours(inputFrame.rgba(), contours, -1, new Scalar(255, 255, 0, 255), 1);       	        
    	        
    	        //For each contour found
    	        approxCurve = new MatOfPoint2f();
    	        for (int i=0; i<contours.size(); i++)
    	        {
    	            //Convert contours(i) from MatOfPoint to MatOfPoint2f
    	            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );	            
    	            
    	            //Processing on mMOP2f1 which is in type MatOfPoint2f
    	            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
    	            
    	            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

    	            //Convert back to MatOfPoint
    	            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

    	            // Get bounding rect of contour
    	            Rect rect = Imgproc.boundingRect(points);
    	            
    	            if(i==0){
    	            	
    	            	// 質心
    	            	// http://monkeycoding.com/?p=617
    	            	Moments mu = Imgproc.moments(contours.get(i), false);
    	            	Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
//    	            	Core.circle(mRgba, momentsPoint, 10, new Scalar(255, 255, 0, 255), -1);
    			        Core.rectangle(mRgba, new Point(momentsPoint.x-10, momentsPoint.y-10), 
    			        		new Point(momentsPoint.x+10, momentsPoint.y+10), new Scalar(0, 255, 255, 255), 2); 
    	            	
    	            	// 面積
    			        // http://monkeycoding.com/?p=617
    		            double contourArea = Imgproc.contourArea(contour2f, false);
    		            Core.putText(mRgba, String.valueOf(contourArea), 
    	            			new Point(10, resolutionPoint.y - 45), 3, 1, new Scalar(0, 255, 128, 255), 2);
    		            
    		            // 周長
    		            // http://monkeycoding.com/?p=617
    		            double arcLength = Imgproc.arcLength(contour2f, true);
    		            Core.putText(mRgba, String.valueOf(arcLength), 
    	            			new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(0, 255, 128, 255), 2);
    		            
//    		            // 凸殼
//    		            // http://monkeycoding.com/?p=612
//    		            MatOfInt mOi= new MatOfInt();
//    		            Imgproc.convexHull(contours.get(i), mOi);                    
//                        Point convexHullPoint = contours.get(i).toList().get(mOi.toList().get(i));
//                        Core.circle(mRgba, convexHullPoint, 10, new Scalar(255, 0, 0, 255), -1);

    	            }else{
    	            	Moments mu = Imgproc.moments(contours.get(i), false);
    	            	Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
//    	            	Core.circle(mRgba, momentsPoint, 10, new Scalar(255, 255, 0, 255), -1);
    			        Core.rectangle(mRgba, new Point(momentsPoint.x-10, momentsPoint.y-10), 
    			        		new Point(momentsPoint.x+10, momentsPoint.y+10), new Scalar(0, 255, 0, 255), 2); 
    	            }
    	            
//    	            Point centerPoint = new Point(rect.x+(rect.width)/2, rect.y+(rect.height)/2);
//    		        Core.rectangle(mRgba, new Point(centerPoint.x-10, centerPoint.y-10), 
//    		        		new Point(centerPoint.x+10, centerPoint.y+10), new Scalar(0, 255, 0, 255), 2); 
    	            
    	            // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
//    	            Core.rectangle(mRgba, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 255, 0, 255), 2); 
    	        }
    		}
    		Core.putText(mRgba, String.valueOf(contours.size()), new Point(10, resolutionPoint.y - 75), 3, 1, new Scalar(255, 0, 0, 255), 2);
        }else{
        	Core.trace(inputFrame.rgba());
        }
        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
        
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
//        mItemFace50 = menu.add("Face size 50%");
//        mItemFace40 = menu.add("Face size 40%");
//        mItemFace30 = menu.add("Face size 30%");
//        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        mItemFindContours = menu.add("FindContours");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }else if (item == mItemFindContours) {
        	findContoursFUN = true;
        	if(findContoursFUN != findContoursFUNtmp){
        		findContoursFUNtmp = findContoursFUN;
        		Toast.makeText(this, "true", Toast.LENGTH_SHORT).show();
        	}else{
        		findContoursFUN = false;
        		findContoursFUNtmp = findContoursFUN;
        		Toast.makeText(this, "false", Toast.LENGTH_SHORT).show();
        	}     	
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
