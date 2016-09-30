package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

public class FdActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
	
	private static final String TAG = "OCVSample::Activity";
	private Mat mRgba;
	private Mat mGray;
	private ScanTool mOpenCvCameraView;
	private Mat mRgbaOrg;
	private Mat mGrayOrg;
	
	// 人臉
	private Mat faceMat;
	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;
	private DetectionBasedTracker mNativeDetector;
	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSizeMin = 0;
	private int mAbsoluteFaceSizeMax = 0;
	private MenuItem mItemFace;
	private boolean faceFUN = false;
	private boolean faceFUNtmp = false;

	// 輪廓
	private Mat	hierarchy;
	private Mat findContoursMat;
	private MatOfPoint2f approxCurve;
	private Point resolutionPoint;
	private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	private MenuItem mItemFindContours;
	private boolean findContoursFUN = false;
	private boolean findContoursFUNtmp = false;
	
	// 色彩
	private Mat colorMat;
	private Mat mSpectrum;
    private ColorBlobDetector mDetector;   
    private Size SPECTRUM_SIZE;
    private Scalar CONTOUR_COLOR;
	private Scalar mBlobColorRgba;
	private Scalar mBlobColorHsv;
	private MenuItem mItemColor;
    private boolean mIsColorSelected = false;
    private boolean colorFUN = false;
	private boolean colorFUNtmp = false;
	
	// 解析度
	private boolean onCameraViewStarted = true;
	private List<android.hardware.Camera.Size> mResolutionList;
	private android.hardware.Camera.Size resolution = null;
	private SubMenu mResolutionMenu;
	private MenuItem[] mResolutionMenuItems;

	private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				// Load native library after(!) OpenCV initialization
				System.loadLibrary("detection_based_tracker");

				try {
					// load cascade file from application resources
//                    InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
//					InputStream is = getResources().openRawResource(R.raw.haarcascade_fullbody);
					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
//                    mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                    mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
//					mCascadeFile = new File(cascadeDir, "haarcascade_fullbody.xml");
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
				mOpenCvCameraView.setOnTouchListener(FdActivity.this);
			}
			break;
			default: {
				super.onManagerConnected(status);
			}
			break;
			}
		}
	};

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		setContentView(R.layout.face_detect_surface_view);
		mOpenCvCameraView = (ScanTool) findViewById(R.id.fd_activity_surface_view);		
		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		
		// 色彩
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255,0,0,255);
		
		// 螢幕解析度設定
		if(onCameraViewStarted == true){
    		onCameraViewStarted = false;
	        mResolutionList = mOpenCvCameraView.getResolutionList();
	        for(int i=0; i<mResolutionList.size(); i++){
	        	Log.e("mResolutionList", mResolutionList.get(i).height+", "+mResolutionList.get(i).width);
	        	if(mResolutionList.get(i).width == 640){
	        		resolution = mResolutionList.get(i);
	        		mOpenCvCameraView.setResolution(resolution);
	        		resolution = mOpenCvCameraView.getResolution();
	        		String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
	        		Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
	        	}
	        }
        }
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {		
		mRgbaOrg = new Mat();
		mGrayOrg = new Mat();
		mRgbaOrg = inputFrame.rgba();
		mGrayOrg = inputFrame.gray();
		mRgbaOrg.copyTo(mRgba);
		mGrayOrg.copyTo(mGray);
		resolutionPoint = new Point(mRgbaOrg.width(), mRgbaOrg.height());
				
		// 臉部
		if(faceFUN){
			setFaceFUN(mGrayOrg, mRgba);
		}
		
		// 輪廓
		if(findContoursFUN){
			setFindContoursFUN(mRgbaOrg, mRgba);
		}
		
		// 顏色
		if(colorFUN){
			setColorFUN(mRgbaOrg, mRgba);
		}
       
		return mRgba;
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		
		// 臉部
		mItemFace = menu.add("Face");
		
		// 輪廓
		mItemFindContours = menu.add("FindContours");
		
		// 顏色
//		mItemColor = menu.add("Color");
		
		// 螢幕解析度
        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];
        ListIterator<android.hardware.Camera.Size> resolutionItr = mResolutionList.listIterator();
        int idx = 0;
        while(resolutionItr.hasNext()) {
            android.hardware.Camera.Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		
		if (item.getGroupId() == 2){
			int id = item.getItemId();
            android.hardware.Camera.Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution);
            resolution = mOpenCvCameraView.getResolution();
            String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
            Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
		}
		
		// 臉部
		if (item == mItemFace) {
			faceFUN = true;
			if(faceFUN != faceFUNtmp) {
				faceFUNtmp = faceFUN;
				Toast.makeText(this, "Face: true", Toast.LENGTH_SHORT).show();
			} else {
				faceFUN = false;
				faceFUNtmp = faceFUN;
				Toast.makeText(this, "Face: false", Toast.LENGTH_SHORT).show();
			}
		}
		
		// 輪廓
		if (item == mItemFindContours) {
			findContoursFUN = true;
			if(findContoursFUN != findContoursFUNtmp) {
				findContoursFUNtmp = findContoursFUN;
				Toast.makeText(this, "FindContours: true", Toast.LENGTH_SHORT).show();
			} else {
				findContoursFUN = false;
				findContoursFUNtmp = findContoursFUN;
				Toast.makeText(this, "FindContours: false", Toast.LENGTH_SHORT).show();
			}
		}
		
		// 顏色
		if (item == mItemColor) {
			colorFUN = true;
			if(colorFUN != colorFUNtmp) {
				colorFUNtmp = colorFUN;
				Toast.makeText(this, "colorFUN: true", Toast.LENGTH_SHORT).show();
			} else {
				colorFUN = false;
				colorFUNtmp = findContoursFUN;
				Toast.makeText(this, "colorFUN: false", Toast.LENGTH_SHORT).show();
			}
		}
		
		return true;
	}
	
	// 臉部
	private Mat setFaceFUN(Mat mGrayOrg, Mat mRgba){
		faceMat = new Mat();
		mGrayOrg.copyTo(faceMat);
		
		if (mAbsoluteFaceSizeMin == 0) {
			int height = faceMat.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSizeMin = Math.round(height * mRelativeFaceSize);
				mAbsoluteFaceSizeMax = Math.round(height * mRelativeFaceSize * 2);
				if(mAbsoluteFaceSizeMax > height){
					mAbsoluteFaceSizeMax = height;
				}
			}
			mNativeDetector.setMinFaceSize(mAbsoluteFaceSizeMin);
		}
		MatOfRect faces = new MatOfRect();
		
//		if (mNativeDetector != null)
//            mNativeDetector.detect(mGray, faces);
		
		mJavaDetector.detectMultiScale(faceMat, faces, 1.1, 6, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                new Size(mAbsoluteFaceSizeMin, mAbsoluteFaceSizeMin), new Size(mAbsoluteFaceSizeMax, mAbsoluteFaceSizeMax));
		
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++){
			Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(255, 0, 255, 255), 3);
			
//			Point facePutText = new Point(
//					Math.abs(facesArray[i].tl().x + facesArray[i].br().x)/2, 
//					Math.abs(facesArray[i].tl().y + facesArray[i].br().y)/2);
//			Core.putText(mRgba, ""+i, facePutText, 2, 1, new Scalar(0, 255, 128, 255), 1);
		}		
		return mRgba;
	}
	
	// 輪廓
	private Mat setFindContoursFUN(Mat mRgbaOrg, Mat mRgba){
		findContoursMat = new Mat();
		mRgbaOrg.copyTo(findContoursMat);
		
		// 二值化
		Imgproc.cvtColor(findContoursMat, findContoursMat, Imgproc.COLOR_RGBA2GRAY, 0);
//		Imgproc.threshold(findContoursMat, findContoursMat, 50, 255, Imgproc.THRESH_BINARY);

		// 高斯濾波器
		Imgproc.GaussianBlur(findContoursMat, findContoursMat, new org.opencv.core.Size(3, 3), 6);

		// 邊緣偵測
		Imgproc.Canny(findContoursMat, findContoursMat, 360, 180);

		// 蝕刻
		Imgproc.erode(findContoursMat, findContoursMat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(1, 1)));

		// 膨脹
		Imgproc.dilate(findContoursMat, findContoursMat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(4, 4)));
				
		contours = new ArrayList<MatOfPoint>();
		hierarchy = new Mat();

		// 找影像輪廓
		Imgproc.findContours(findContoursMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
		hierarchy.release();

		if(contours.size() != 0 && contours.size() < 500) {
			// 劃出輪廓線
			Imgproc.drawContours(mRgba, contours, -1, new Scalar(255, 255, 0, 255), 1);

			//For each contour found
			approxCurve = new MatOfPoint2f();
			for (int i=0; i<contours.size(); i++) {
				//Convert contours(i) from MatOfPoint to MatOfPoint2f
				MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );

				//Processing on mMOP2f1 which is in type MatOfPoint2f
				double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;

				Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

				//Convert back to MatOfPoint
				MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

				// Get bounding rect of contour
				Rect rect = Imgproc.boundingRect(points);
				
				RotatedRect box = Imgproc.minAreaRect(contour2f);
				
				Point[] pointsList = new Point[4];;

				box.points(pointsList);
				
				
			    for(int r = 0; r<4; ++r){
//			        Core.line(mRgba, pointsList[r], pointsList[(r+1)%4], new Scalar(127, 255, 255, 255), 2);
			    }
			
				if(i==0) {

//					// 質心
//					// http://monkeycoding.com/?p=617
//					Moments mu = Imgproc.moments(contours.get(i), false);
//					Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
//	            	Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
//					Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
//					Core.rectangle(mRgba, new Point(momentsPoint.x-10, momentsPoint.y-10),
//					               new Point(momentsPoint.x+10, momentsPoint.y+10), new Scalar(0, 255, 255, 255), 2);

//					// 面積
//					// http://monkeycoding.com/?p=617
//					double contourArea = Imgproc.contourArea(contour2f, false);
//					Core.putText(mRgba, String.valueOf(contourArea),
//					             new Point(10, resolutionPoint.y - 45), 3, 1, new Scalar(0, 255, 128, 255), 2);

//					// 周長
//					// http://monkeycoding.com/?p=617
//					double arcLength = Imgproc.arcLength(contour2f, true);
//					Core.putText(mRgba, String.valueOf(arcLength),
//					             new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(0, 255, 128, 255), 2);

//		            // 凸殼
//		            // http://monkeycoding.com/?p=612
//		            MatOfInt mOi= new MatOfInt();
//		            Imgproc.convexHull(contours.get(i), mOi);
//                    Point convexHullPoint = contours.get(i).toList().get(mOi.toList().get(i));
//                    Core.circle(mRgba, convexHullPoint, 10, new Scalar(255, 0, 0, 255), -1);

				} else {
//					Moments mu = Imgproc.moments(contours.get(i), false);
//					Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
//	            	Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
//					Core.rectangle(mRgba, new Point(momentsPoint.x-10, momentsPoint.y-10),
//					               new Point(momentsPoint.x+10, momentsPoint.y+10), new Scalar(0, 255, 0, 255), 2);
				}

				// draw enclosing rectangle (all same color, but you could use variable i to make them unique)
	            Core.rectangle(mRgba, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 255, 0, 255), 2);
			}

			// 找影像輪廓數量顯示
// 			Core.putText(mRgba, String.valueOf(contours.size()), new Point(10, resolutionPoint.y - 75), 3, 1, new Scalar(255, 0, 0, 255), 2);
 			Core.putText(mRgba, String.valueOf(contours.size()), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);
 			
		} else {

			// 找影像輪廓數量顯示
//			Core.putText(mRgba, String.valueOf(0), new Point(10, resolutionPoint.y - 75), 3, 1, new Scalar(255, 0, 0, 255), 2);
			Core.putText(mRgba, String.valueOf(0), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);

//			// 面積
//			Core.putText(mRgba, String.valueOf(0),
//			             new Point(10, resolutionPoint.y - 45), 3, 1, new Scalar(0, 255, 128, 255), 2);
//
//			// 周長
//			Core.putText(mRgba, String.valueOf(0),
//			             new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(0, 255, 128, 255), 2);
		}
		return mRgba;
	}
	
	// 顏色
	private Mat setColorFUN(Mat mRgbaOrg, Mat mRgba){
		colorMat = new Mat();
		mRgbaOrg.copyTo(colorMat);
		
        if (mIsColorSelected) {
            mDetector.process(mRgba);
            List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());
            Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR);

            Mat colorLabel = mRgba.submat(4, 68, 4, 68);
            colorLabel.setTo(mBlobColorRgba);

            Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
            mSpectrum.copyTo(spectrumLabel);
        }
        
		return mRgba;
	}
	
	@Override
	public boolean onTouch(View v, MotionEvent event) {
				
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;

        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");
        
        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        Rect touchedRect = new Rect();

        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;
        
        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);
        
        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;
        
        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);
        
        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] +
                ", " + mBlobColorRgba.val[2] + ", " + mBlobColorRgba.val[3] + ")");
        
        mDetector.setHsvColor(mBlobColorHsv);

        Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);

        mIsColorSelected = true;

        touchedRegionRgba.release();
        touchedRegionHsv.release();       

		return false; // don't need subsequent touch events
	}
	
    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
