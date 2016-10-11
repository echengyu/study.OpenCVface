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

	// 人臉
	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;
	private DetectionBasedTracker mNativeDetector;
	private float mRelativeFaceSizeMin = 0.095f;
	private float mRelativeFaceSizeMax = 0.9f;
	private int mAbsoluteFaceSizeMin = 0;
	private int mAbsoluteFaceSizeMax = 0;
	private MenuItem mItemFace;
	private boolean faceFUN = false;

	// 輪廓
	private MatOfPoint2f approxCurve;
	private Point resolutionPoint;
	private MenuItem mItemFindContours;
	private final int HYSTERESIS_THRESHOLD1 = 128;
	private final int HYSTERESIS_THRESHOLD2 = 255;
	private boolean findContoursFUN = false;

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

	// 閃光燈
	private MenuItem mItemCameraFlashLight;
	
	// ROI TODO
	private MenuItem mItemROI;
	private boolean ROIFUN = false;

	// 解析度
	private boolean onCameraViewStarted = true;
	private List<android.hardware.Camera.Size> mResolutionList;
	private android.hardware.Camera.Size resolution = null;
	private SubMenu mResolutionMenu;
	private MenuItem[] mResolutionMenuItems;

	Point mIsColorSelectedPoint;

	int tmpEdit = 2;

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
					InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
					mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
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
		if(onCameraViewStarted == true) {
			onCameraViewStarted = false;
			mResolutionList = mOpenCvCameraView.getResolutionList();
			for(int i=0; i<mResolutionList.size(); i++) {
				if(mResolutionList.get(i).width == 640) {
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
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();

		// 取得解析度
		resolutionPoint = new Point(mRgba.width(), mRgba.height());

		// 輪廓辨識
		if(findContoursFUN) {
			setFindContoursFUN(mGray, mRgba);
		}

		// 臉部辨識
		if(faceFUN && mJavaDetector != null) {
			setFaceFUN(mGray, mRgba);
		}

		// 顏色輪廓辨識
		if(colorFUN && mIsColorSelected) {
			setColorFUN(mRgba, mRgba);
		}
		
		// ROI 
		if(ROIFUN){
			// TODO
		}

		return mRgba;
	}

	// 臉部辨識
	private Mat setFaceFUN(Mat mMatOrg, Mat mRgba) {
		if (mAbsoluteFaceSizeMin == 0) {
			int height = mRgba.rows();
			if (Math.round(height * mRelativeFaceSizeMin) > 0) {
				mAbsoluteFaceSizeMin = Math.round(height * mRelativeFaceSizeMin);
				mAbsoluteFaceSizeMax = Math.round(height * mRelativeFaceSizeMax);
				if(mAbsoluteFaceSizeMax > height) {
					mAbsoluteFaceSizeMax = height;
				}
			}
			mNativeDetector.setMinFaceSize(mAbsoluteFaceSizeMin);
		}

		MatOfRect faces = new MatOfRect();	
		Mat mTmp = new Mat();
		
		if(ROIFUN){
			float cutOffestY = 0.7f;
			cutOffestY = (cutOffestY > 1.0f) ? 1.0f : cutOffestY;
			int cutY0 = (int) Math.round((resolutionPoint.y / 2) * (1 - cutOffestY));
			int cutY1 = (int) Math.round((resolutionPoint.y / 2) * (1 + cutOffestY));
			Rect mRect = new Rect(0, cutY0, (int) resolutionPoint.x, cutY1 - cutY0);
			mMatOrg.submat(mRect).copyTo(mTmp);
			
			mJavaDetector.detectMultiScale(mTmp, faces, 1.1, 6, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSizeMin, mAbsoluteFaceSizeMin), new Size(mAbsoluteFaceSizeMax, mAbsoluteFaceSizeMax));

			Rect[] facesArray = faces.toArray();
			
			for (int i = 0; i < facesArray.length; i++){
				Point pi1 = new Point(facesArray[i].tl().x, facesArray[i].tl().y + cutY0);
				Point pi2 = new Point(facesArray[i].br().x, facesArray[i].br().y + cutY0);
				Core.rectangle(mRgba, pi1, pi2, new Scalar(255, 0, 255, 255), 3);
			}
			Core.line(mRgba, new Point(0, cutY0), new Point(resolutionPoint.x, cutY0), new Scalar(255, 51, 153, 255), 2);
			Core.line(mRgba, new Point(0, cutY1), new Point(resolutionPoint.x, cutY1), new Scalar(255, 51, 153, 255), 2);
			
		}else{
			mMatOrg.copyTo(mTmp);
			
			mJavaDetector.detectMultiScale(mTmp, faces, 1.1, 6, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSizeMin, mAbsoluteFaceSizeMin), new Size(mAbsoluteFaceSizeMax, mAbsoluteFaceSizeMax));

			Rect[] facesArray = faces.toArray();
			
			for (int i = 0; i < facesArray.length; i++){
				Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(255, 0, 255, 255), 3);
			}
		}
		return mRgba;
	}

	// 輪廓辨識
	private Mat setFindContoursFUN(Mat mMatOrg, Mat mRgba) {
		Mat mTmp = new Mat();

		float cutOffestY = 0.7f;
		cutOffestY = (cutOffestY > 1.0f) ? 1.0f : cutOffestY;
		int cutY0 = (int) Math.round((resolutionPoint.y / 2) * (1 - cutOffestY));
		int cutY1 = (int) Math.round((resolutionPoint.y / 2) * (1 + cutOffestY));
		Rect mRect = new Rect(0, cutY0, (int) resolutionPoint.x, cutY1 - cutY0);
		
		if(ROIFUN){
			mMatOrg.submat(mRect).copyTo(mTmp);
		}else{
			mMatOrg.copyTo(mTmp);
		}
		
		// 二值化
		Imgproc.threshold(mTmp, mTmp, 100, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C | Imgproc.THRESH_BINARY);

		// 影像金字塔(縮小)
		Imgproc.pyrDown(mTmp, mTmp, new Size(mTmp.cols()/2, mTmp.rows()/2));

		// 蝕刻
		Imgproc.erode(mTmp, mTmp, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7)));

		// 邊緣偵測
		Imgproc.Canny(mTmp, mTmp, HYSTERESIS_THRESHOLD1, HYSTERESIS_THRESHOLD2, 3, false);

		// 影像金字塔(放大)
		Imgproc.pyrUp(mTmp, mTmp, new Size(mTmp.cols()*2, mTmp.rows()*2));

		// 找影像輪廓
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(mTmp, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

		if(contours.size() != 0 && contours.size() < 500) {
			// 劃出輪廓線
			if(ROIFUN){
				Imgproc.drawContours(mRgba, contours, -1, new Scalar(255, 255, 0, 255), 1, 8, new Mat(), 1, new Point(0, cutY0));
				
			}else{
				Imgproc.drawContours(mRgba, contours, -1, new Scalar(255, 255, 0, 255), 1);
			}
			
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

//				// 輪廓取矩形可自動調整大小
//				RotatedRect rect2 = Imgproc.minAreaRect(contour2f);
//				Point[] vertices = new Point[4];
//				rect2.points(vertices);
//				for (int j = 0; j < 4; j++) {
//					Core.line(mRgba, vertices[j], vertices[(j + 1) % 4], new Scalar(0, 255, 0, 255), 2);
//				}

				// draw enclosing rectangle (all same color, but you could use variable i to make them unique)
				if(ROIFUN){
					Core.rectangle(mRgba, new Point(rect.x,rect.y + cutY0), new Point(rect.x+rect.width,rect.y+rect.height + cutY0), new Scalar(0, 255, 0, 255), 2);
					Core.line(mRgba, new Point(0, cutY0), new Point(resolutionPoint.x, cutY0), new Scalar(255, 51, 153, 255), 2);
					Core.line(mRgba, new Point(0, cutY1), new Point(resolutionPoint.x, cutY1), new Scalar(255, 51, 153, 255), 2);
				}else{
					Core.rectangle(mRgba, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 255, 0, 255), 2);
				}

				/***** 測試用 *****
				if(i==0) {

					// 質心
					// http://monkeycoding.com/?p=617
					Moments mu = Imgproc.moments(contours.get(i), false);
					Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
					Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
					Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
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

				    // 凸殼
				    // http://monkeycoding.com/?p=612
				    MatOfInt mOi= new MatOfInt();
				    Imgproc.convexHull(contours.get(i), mOi);
				    Point convexHullPoint = contours.get(i).toList().get(mOi.toList().get(i));
				    Core.circle(mRgba, convexHullPoint, 10, new Scalar(255, 0, 0, 255), -1);

				} else {
					Moments mu = Imgproc.moments(contours.get(i), false);
					Point momentsPoint = new Point((int)(mu.get_m10() / mu.get_m00()), (int)(mu.get_m01() / mu.get_m00()));
					Core.circle(mRgba, momentsPoint, 3, new Scalar(255, 255, 0, 255), -1);
					Core.rectangle(mRgba, new Point(momentsPoint.x-10, momentsPoint.y-10),
					               new Point(momentsPoint.x+10, momentsPoint.y+10), new Scalar(0, 255, 0, 255), 2);
				}
				**********/
			}

			// 找尋到影像輪廓數量顯示
			Core.putText(mRgba, String.valueOf(contours.size()), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);
		} else {

			// 找影像輪廓數量顯示
			Core.putText(mRgba, String.valueOf(0), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);

			/***** 測試用 *****
			// 面積
			Core.putText(mRgba, String.valueOf(0),
			             new Point(10, resolutionPoint.y - 45), 3, 1, new Scalar(0, 255, 128, 255), 2);

			// 周長
			Core.putText(mRgba, String.valueOf(0),
			             new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(0, 255, 128, 255), 2);
			**********/
		}
		return mRgba;
	}

	// 顏色
	private Mat setColorFUN(Mat mRgbaOrg, Mat mRgba) {
		colorMat = new Mat();
		mRgbaOrg.copyTo(colorMat);

//        if (mIsColorSelected) {
//            mDetector.process(mRgba);
//            List<MatOfPoint> contours = mDetector.getContours();
//            Log.e(TAG, "Contours count: " + contours.size());
//            Imgproc.drawContours(mRgba, contours, -1, CONTOUR_COLOR);
//
//            Mat colorLabel = mRgba.submat(4, 68, 4, 68);
//            colorLabel.setTo(mBlobColorRgba);
//
//            Mat spectrumLabel = mRgba.submat(4, 4 + mSpectrum.rows(), 70, 70 + mSpectrum.cols());
//            mSpectrum.copyTo(spectrumLabel);
//        }

		return mRgba;
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");

		// 臉部
		mItemFace = menu.add("Face");

		// 輪廓
//		mItemFindContours = menu.add("FindContours");

		// 顏色
//		mItemColor = menu.add("Color");
		
		// ROI
//		mItemROI = menu.add("ROI");

		// 閃光燈
//		mItemCameraFlashLight = menu.add("FlashLight");

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

		if (item.getGroupId() == 2) {
			int id = item.getItemId();
			android.hardware.Camera.Size resolution = mResolutionList.get(id);
			mOpenCvCameraView.setResolution(resolution);
			resolution = mOpenCvCameraView.getResolution();
			String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
			Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
		}

		// 臉部辨識
		if (item == mItemFace) {
			if(!faceFUN) {
				faceFUN = true;
				Toast.makeText(this, "Face: true", Toast.LENGTH_SHORT).show();
			} else {
				faceFUN = false;
				Toast.makeText(this, "Face: false", Toast.LENGTH_SHORT).show();
			}
		}

		// 輪廓辨識
		if (item == mItemFindContours) {
			if(!findContoursFUN) {
				findContoursFUN = true;
				Toast.makeText(this, "FindContours: true", Toast.LENGTH_SHORT).show();
			} else {
				findContoursFUN = false;
				Toast.makeText(this, "FindContours: false", Toast.LENGTH_SHORT).show();
			}
		}

		// 顏色辨識
		if (item == mItemColor) {
			if(!colorFUN) {
				colorFUN = true;
				Toast.makeText(this, "colorFUN: true", Toast.LENGTH_SHORT).show();
			} else {
				colorFUN = false;
				Toast.makeText(this, "colorFUN: false", Toast.LENGTH_SHORT).show();
			}

//			AlertDialog.Builder editDialog = new AlertDialog.Builder(FdActivity.this);
//			editDialog.setCancelable(true);
//
//			final SeekBar seekBar = new SeekBar(FdActivity.this);
//			seekBar.setMax(5);
//			editDialog.setView(seekBar);
//			seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
//
//		        @Override
//		        public void onStopTrackingTouch(SeekBar seekBar) {
//		        	Toast.makeText(FdActivity.this, String.valueOf(tmpEdit), Toast.LENGTH_SHORT).show();
//		        }
//
//		        @Override
//		        public void onStartTrackingTouch(SeekBar seekBar) {
//
//		        }
//
//	            @Override
//	            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
//	            	tmpEdit = progress;
//	            	Core.putText(mRgba, String.valueOf(progress),
//	       	             new Point(10, resolutionPoint.y - 45), 3, 1, new Scalar(0, 255, 128, 255), 2);
//	            }
//		    });
//
//			editDialog.setNegativeButton("Exit", new DialogInterface.OnClickListener() {
//				// do something when the button is clicked
//				public void onClick(DialogInterface arg0, int arg1) {
//					//...
//				}
//			});
//			editDialog.show();

		}

		// ROI
		if (item == mItemROI) {
			if(!ROIFUN) {
				ROIFUN = true;
				Toast.makeText(this, "ROIFUN: true", Toast.LENGTH_SHORT).show();
			} else {
				ROIFUN = false;
				Toast.makeText(this, "ROIFUN: false", Toast.LENGTH_SHORT).show();
			}
		}
		
		// 閃光燈
		if (item == mItemCameraFlashLight) {
			mOpenCvCameraView.setCameraFlashLight();
		}

		return true;
	}

	// 點擊螢幕觸發事件
	@Override
	public boolean onTouch(View v, MotionEvent event) {

		// 取得影像
		int cols = mRgba.cols();
		int rows = mRgba.rows();

		// 偏移量
		int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
		int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

		int x = (int)event.getX() - xOffset;
		int y = (int)event.getY() - yOffset;

		mIsColorSelectedPoint = new Point(x, y);

		Log.e(TAG, "Touch image coordinates: (" + x + ", " + y + ")");

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

		Log.e(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] +
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
