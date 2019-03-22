package onl.fdt.java.android.odappliedwsc;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.SeekBar;
import android.widget.TextView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class JavaActivity extends AppCompatActivity implements CvCameraViewListener2 {
    @Override
    protected void onStart() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    Manifest.permission.CAMERA)) {

                // Show an expanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {

                // No explanation needed, we can request the permission.

                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CAMERA},
                        CAMERA_PERMISSION_CODE);

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        }
        super.onStart();
    }

    public static final int CAMERA_PERMISSION_CODE = 1;

    public static final boolean ENABLE_GPIO_FUNCTION = false;

    static {
        System.loadLibrary("opencv_java3");
        //System.loadLibrary("libfirefly_api");
    }

    // Initialize OpenCV manager.
//    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
//        @Override
//        public void onManagerConnected(int status) {
//            switch (status) {
//                case LoaderCallbackInterface.SUCCESS: {
//                    Log.i(TAG, "OpenCV loaded successfully");
//                    mOpenCvCameraView.enableView();
//                    break;
//                }
//                default: {
//                    super.onManagerConnected(status);
//                    break;
//                }
//            }
//        }
//    };

    @Override
    public void onResume() {
        super.onResume();
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case CAMERA_PERMISSION_CODE: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.

                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_java);
        // Set up camera listener.
        SeekBar seekBar = findViewById(R.id.seekBar);

        SeekBar.OnSeekBarChangeListener seekBarChangeListener = new SeekBar.OnSeekBarChangeListener() {

            TextView tvProgressLabel;

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // updated continuously as the user slides the thumb
                seekBarProgress = progress;
                THRESHOLD = (seekBarProgress + 0.0) / 100.0;
                tvProgressLabel = findViewById(R.id.text1);
                tvProgressLabel.setText("Progress: " + progress);
                Log.i(TAG, "Threshold: " + THRESHOLD);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // called when the user first touches the SeekBar
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // called after the user finishes moving the SeekBar
            }
        };

        seekBar.setOnSeekBarChangeListener(seekBarChangeListener);

        seekBarProgress = seekBar.getProgress();

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        //for gpio issues
        //if (ENABLE_GPIO_FUNCTION) {
        //    this.fireflyApi = new FireflyApi();
        //    this.io1 = FireflyApi.gpioParse("GPIO0_B4");
        //    this.io2 = FireflyApi.gpioParse("GPIO4_D5");
        //    Log.i(TAG, "io1 port: " + this.io1 + " io2 port: " + this.io2);
        //}

    }

    private int seekBarProgress = 50;

    // Load a network.
    public void onCameraViewStarted(int width, int height) {
        final String proto = getPath(
            "v1_graph.pbtxt"
                , this);
        final String weights = getPath(
            "v1_2018.pb"
                , this);
        Log.i(TAG, "proto: " + proto + ", weights: " + weights);
        net = Dnn.readNetFromTensorflow(weights, proto);

        net.setPreferableTarget(Dnn.DNN_TARGET_OPENCL_FP16);
        //net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        Log.i(TAG, "Network loaded successfully");
    }

    double[] getConfidentContribute(int cols, int rows, int left, int top, int right, int bottom, double confidence) {
        int midcols = cols / 2;
        int midrows = rows / 2;
        double[] result = {0.0, 0.0, 0.0, 0.0};
        double subarea = midcols * midrows + 0.0;

        int mincol = (Math.min(midcols, right) - Math.min(left, midcols));
        int maxcol = (Math.max(midcols, right) - Math.max(midcols, left));
        int minrow = (Math.min(midrows, bottom) - Math.min(midrows, top));
        int maxrow = (Math.max(midrows, bottom) - Math.max(midrows, top));

        result[1] = mincol * minrow;
        result[0] = maxcol * minrow;
        result[2] = mincol * maxrow;
        result[3] = maxcol * maxrow;

        for (int i = 0; i < result.length; ++i) {
            result[i] /= subarea;
            result[i] *= confidence;
        }

        return result;
    }

    final int IN_WIDTH = 300;
    final int IN_HEIGHT = 300;

    final float WH_RATIO = (float) IN_WIDTH / IN_HEIGHT;
    final double IN_SCALE_FACTOR = 1.0 / 127.5;
    final double MEAN_VAL = 127.5;
    private double THRESHOLD = 0.6;

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat frame;
//        try {
        frame = inputFrame.rgba();
//        return frame;

        Mat frameT = frame.t();
        Core.flip(frame.t(), frameT, 1);
        Imgproc.resize(frameT, frameT, frame.size());
        frame.release();
        frame = frameT;
        // Get a new frame

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        // Forward image through network.
        Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR,
                new Size(IN_WIDTH, IN_HEIGHT),
                new Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), false, false);
        net.setInput(blob);
        Mat detections = net.forward();
        int cols = frame.cols();
        int rows = frame.rows();
        detections = detections.reshape(1, (int) detections.total() / 7);
        double[] areaConfidence = {0.0, 0.0, 0.0, 0.0};
        boolean hasPersion = false;
        for (int i = 0; i < detections.rows(); ++i) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > THRESHOLD) {
                int classId = (int) detections.get(i, 1)[0];
                if (classId == 1) {
                    hasPersion = true;
                    int left = (int) (detections.get(i, 3)[0] * cols);
                    int top = (int) (detections.get(i, 4)[0] * rows);
                    int right = (int) (detections.get(i, 5)[0] * cols);
                    int bottom = (int) (detections.get(i, 6)[0] * rows);

                    //Log.i(TAG, "l,t,r,b = " + left + " " + top + " " + right + " " + bottom);
                    double[] currentContribution = getConfidentContribute(cols, rows, left, top, right, bottom, confidence);
                    for (int j = 0; j < areaConfidence.length; ++j) {
                        areaConfidence[j] += currentContribution[j];
                    }

                    // Draw rectangle around detected object.
                    Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                            new Scalar(0, 255, 0));
                    String label = "Person: " + String.valueOf(confidence);
                    int[] baseLine = new int[1];
                    Size labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
                    // Draw background for label.
                    Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
                            new Point(left + labelSize.width, top + baseLine[0]),
                            new Scalar(255, 255, 255), Core.FILLED);
                    // Write class name and confidence.
                    Imgproc.putText(frame, label, new Point(left, top),
                            Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));
                }

            }
        }
        int maxAreaId = 0;

        for (int i = 1; i < areaConfidence.length; ++i) {
            if (areaConfidence[i] > areaConfidence[maxAreaId]) {
                maxAreaId = i;
            }
        }
        Log.i(TAG, "maxId: " + maxAreaId + " 00: " + areaConfidence[0] + " 01: " + areaConfidence[1] + " 10: " + areaConfidence[2] + " 11:" + areaConfidence[3]);

        if (
//                    lastRequestTimeMill + REQUEST_TIME_GAP < System.currentTimeMillis() &&
                lastMaxAreaId != maxAreaId) {
            //if (ENABLE_GPIO_FUNCTION) {
            //    this.writeGPIOofMaxAreaId(maxAreaId);
            //}

            lastRequestTimeMill = System.currentTimeMillis();
            lastMaxAreaId = maxAreaId;
        }
        //if (ENABLE_GPIO_FUNCTION) {
        //    Log.i(TAG, "io1: " + fireflyApi.gpioRead(this.io1) + " io2: " + fireflyApi.gpioRead(this.io2));
        //}


        return frame;
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

//        return null;
    }

    long lastRequestTimeMill = 0L;
//    public static final long REQUEST_TIME_GAP = 5000L;

    int lastMaxAreaId = 0;
    boolean lastHasPerson = false;

    //private int io1;
    //private int io2;

    //private boolean writeGPIOofMaxAreaId(int maxAreaId) {
    //    return fireflyApi.gpioCtrl(this.io2, "out", maxAreaId > 1 ? 1 : 0) && fireflyApi.gpioCtrl(this.io1, "out", maxAreaId % 2);
    //}

    //FireflyApi fireflyApi;

    public void onCameraViewStopped() {
    }

    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    public static final String TAG = "fdtnet";
    private Net net;
    private CameraBridgeViewBase mOpenCvCameraView;
}
