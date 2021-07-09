/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.gson.Gson;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final int RESULT_LOAD_IMAGE = 656;
    private static final int STORAGE_PERMISSION = 655;
    private final Gson gson = new Gson();
    // FaceNet
//  private static final int TF_OD_API_INPUT_SIZE = 160;
//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
//  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
//  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";

    // MobileFaceNet
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";


    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    //private static final int CROP_SIZE = 320;
    //private static final Size CROP_SIZE = new Size(320, 320);


    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private SimilarityClassifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;
    private boolean addPending = false;
    //private boolean adding = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    //private Matrix cropToPortraitTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    // Face detector
    private FaceDetector faceDetector;

    // here the preview image is drawn in portrait way
    private Bitmap portraitBmp = null;
    // here the face is cropped and drawn
    private Bitmap faceBmp = null;

    private FloatingActionButton fabAdd;
    private FloatingActionButton fabSearch;
    private Context activityContext;

    //private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        activityContext = this;

        fabAdd = findViewById(R.id.fab_add);
        fabAdd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                requestPermissions();
            }
        });

        fabSearch = findViewById(R.id.fab_seach);
        fabSearch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                onAddClick();
            }
        });

//     Real-time contour detection of multiple faces
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                        .build();


        FaceDetector initDetector = FaceDetection.getClient(options);

        faceDetector = initDetector;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == Activity.RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            String[] filePathColumn = {MediaStore.Images.Media.DATA};
            Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            Uri storageUri = Uri.parse(picturePath);
            BitmapFactory.Options bmOptions = new BitmapFactory.Options();
            Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
//      detector.recognizeImage(bitmap,true);
//asdf 1
            try {
                InputImage image = InputImage.fromFilePath(this, selectedImage);
                ++timestamp;
                final long currTimestamp = timestamp;

                if(faceDetector != null){
                    addPending = true;
                    faceDetector
                            .process(image)
                            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                                @Override
                                public void onSuccess(List<Face> faces) {
                                    if (faces.size() == 0) {
                                        updateResults(currTimestamp, new LinkedList<>());
                                        return;
                                    }
                                    runInBackground(
                                            new Runnable() {
                                                @Override
                                                public void run() {
//                                  onFacesDetected(currTimestamp, faces, addPending);
                                                    //asdf 3
                                                    detector.enableStatLogging(true);
                                                    List<SimilarityClassifier.Recognition> mappedFaces = loopThroughFaces(currTimestamp, faces,selectedImage);
                                                    for (SimilarityClassifier.Recognition mappedFace:mappedFaces) {
                                                        showAddFaceDialog(mappedFace, true);
                                                    }
                                                    addPending = false;
                                                }
                                            });
                                }
                            });
                }
            } catch (IOException err) {
                Log.e("faceDetector",gson.toJson(err));
            }
        }
    }

    private void onAddClick() {
        addPending = true;
        //Toast.makeText(this, "click", Toast.LENGTH_LONG ).show();
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);


        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            //cropSize = TF_OD_API_INPUT_SIZE;



            SharedPreferences sharedPref = getApplicationContext().getSharedPreferences(
                    getString(R.string.tfe_od_local), Context.MODE_PRIVATE);
            String prevLocalImage = sharedPref.getString(getString(R.string.tfe_od_local), "kosong");
            Log.e("asdf","prevLocalImage: " + prevLocalImage);
            if(!prevLocalImage.equals("kosong")) {
                SimilarityClassifier.Recognition prevRec = gson.fromJson(prevLocalImage, SimilarityClassifier.Recognition.class);
                Log.e("asdf","prevRec: " + gson.toJson(prevRec));

                if(detector != null){
                    Log.e("asdf","REREGISTER PREV REC: ");
                    detector.register("User",prevRec);
                }
            }
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


        int targetW, targetH;
        if (sensorOrientation == 90 || sensorOrientation == 270) {
            targetH = previewWidth;
            targetW = previewHeight;
        }
        else {
            targetW = previewWidth;
            targetH = previewHeight;
        }
        int cropW = (int) (targetW / 2.0);
        int cropH = (int) (targetH / 2.0);

        croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

        portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
        faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropW, cropH,
                        sensorOrientation, MAINTAIN_ASPECT);

//    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);


        Matrix frameToPortraitTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        targetW, targetH,
                        sensorOrientation, MAINTAIN_ASPECT);



        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }


    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
        if(faceDetector != null){
            faceDetector
                    .process(image)
                    .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                        @Override
                        public void onSuccess(List<Face> faces) {
                            if (faces.size() == 0) {
                                updateResults(currTimestamp, new LinkedList<>());
                                return;
                            }
                            runInBackground(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            onFacesDetected(currTimestamp, faces, addPending);
                                            addPending = false;
                                        }
                                    });
                        }
                    });
        }
    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }


    // Face Processing
    private Matrix createTransform(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation) {

        Matrix matrix = new Matrix();
        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

        if (applyRotation != 0) {

            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;

    }

    private void showAddFaceDialog(SimilarityClassifier.Recognition rec, boolean manualAdd) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = getLayoutInflater();
        View dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null);
        ImageView ivFace = dialogLayout.findViewById(R.id.dlg_image);
        TextView tvTitle = dialogLayout.findViewById(R.id.dlg_title);
        EditText etName = dialogLayout.findViewById(R.id.dlg_input);

        tvTitle.setText("Add Face");
        ivFace.setImageBitmap(rec.getCrop());
        etName.setHint("Input name");

        builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
            @Override
            public void onClick(DialogInterface dlg, int i) {
                //aaaaaa
                if(manualAdd){

                    SharedPreferences sharedPreferences = getApplicationContext().getSharedPreferences(
                            getString(R.string.tfe_od_local), Context.MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPreferences.edit();
                    Log.e("asdf","gsontoJson: " + gson.toJson(rec));
                    editor.putString(getString(R.string.tfe_od_local), gson.toJson(rec));
                    editor.apply();
                }

                String name = etName.getText().toString();
                if (name.isEmpty()) {
                    return;
                }
                Log.e("name",name);
                Log.e("rec",gson.toJson(rec));
                detector.register(name, rec);
                //knownFaces.put(name, rec);
                dlg.dismiss();
            }
        });
        builder.setView(dialogLayout);
        builder.show();
    }

    private void requestPermissions (){
        Log.e("start requesst","");
        String[] PERMISSIONS = {
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
        };
        ActivityCompat.requestPermissions(this,PERMISSIONS,STORAGE_PERMISSION);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        Log.e("PermissionsResult", String.valueOf(requestCode));
        if (requestCode == STORAGE_PERMISSION) {
            for (int i = 0; i < permissions.length; i++) {
                String permission = permissions[i];
                int grantResult = grantResults[i];

                if (permission.equals(Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    if (grantResult == PackageManager.PERMISSION_GRANTED) {
                        addPending = true;
                        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                        startActivityForResult(intent, RESULT_LOAD_IMAGE);
                    } else {
                        requestPermissions();
                    }
                }
            }
        }
    }

    private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

        tracker.trackResults(mappedRecognitions, currTimestamp);
        trackingOverlay.postInvalidate();
        computingDetection = false;
        //adding = false;

        if (mappedRecognitions.size() > 0) {
            LOGGER.i("Adding results");
            SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
            if (rec.getExtra() != null) {//asdf 2 face detected
                showAddFaceDialog(rec, false);
            }
        }

        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        showFrameInfo(previewWidth + "x" + previewHeight);
                        showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                        showInference(lastProcessingTimeMs + "ms");
                    }
                });

    }

    private void onFacesDetected(long currTimestamp, List<Face> faces, boolean add) {
        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
        final Canvas canvas = new Canvas(cropCopyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Style.STROKE);
        paint.setStrokeWidth(2.0f);

        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        switch (MODE) {
            case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
        }

        final List<SimilarityClassifier.Recognition> mappedRecognitions = new LinkedList<SimilarityClassifier.Recognition>();

        //final List<Classifier.Recognition> results = new ArrayList<>();

        // Note this can be done only once
        int sourceW = rgbFrameBitmap.getWidth();
        int sourceH = rgbFrameBitmap.getHeight();
        int targetW = portraitBmp.getWidth();
        int targetH = portraitBmp.getHeight();
        Matrix transform = createTransform(
                sourceW,
                sourceH,
                targetW,
                targetH,
                sensorOrientation);
        final Canvas cv = new Canvas(portraitBmp);

        // draws the original image in portrait mode.
        cv.drawBitmap(rgbFrameBitmap, transform, null);

        final Canvas cvFace = new Canvas(faceBmp);

        boolean saved = false;

        Log.e("faces",String.valueOf(faces.size()));
        Log.e("getStatString",detector.getStatString());

        for (Face face : faces) {

            LOGGER.i("FACE" + face.toString());
            LOGGER.i("Running detection on face " + currTimestamp);
            //results = detector.recognizeImage(croppedBitmap);

            final RectF boundingBox = new RectF(face.getBoundingBox());

            //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
            final boolean goodConfidence = true; //face.get;
            if (boundingBox != null && goodConfidence) { //asdf ori

                // maps crop coordinates to original
                cropToFrameTransform.mapRect(boundingBox);

                // maps original coordinates to portrait coordinates
                RectF faceBB = new RectF(boundingBox);
                transform.mapRect(faceBB);

                // translates portrait to origin and scales to fit input inference size
                //cv.drawRect(faceBB, paint);
                float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
                float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
                Matrix matrix = new Matrix();
                matrix.postTranslate(-faceBB.left, -faceBB.top);
                matrix.postScale(sx, sy);

                cvFace.drawBitmap(portraitBmp, matrix, null);

                //canvas.drawRect(faceBB, paint);
                String label = "";
                float confidence = -1f;
                Integer color = Color.BLUE;
                Object extra = null;
                Bitmap crop = null;

                if (add && portraitBmp != null) {
                    crop = Bitmap.createBitmap(portraitBmp,
                            (int) faceBB.left,
                            (int) faceBB.top,
                            (int) faceBB.width(),
                            (int) faceBB.height());
                }

                final long startTime = SystemClock.uptimeMillis();
                final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                if (resultsAux.size() > 0) {

                    SimilarityClassifier.Recognition result = resultsAux.get(0);

                    extra = result.getExtra();
                    if (extra != null) {
                        Log.e("embeeding", gson.toJson(extra));
                    }

                    float conf = result.getDistance();
                    if (conf < 1.0f) {

                        confidence = conf;
                        label = result.getTitle();
                        if (result.getId().equals("0")) {
                            color = Color.GREEN;
                        }
                        else {
                            color = Color.RED;
                        }
                    }

                }

                if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

                    // camera is frontal so the image is flipped horizontally
                    // flips horizontally
                    Matrix flip = new Matrix();
                    if (sensorOrientation == 90 || sensorOrientation == 270) {
                        flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
                    }
                    else {
                        flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
                    }
                    //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
                    flip.mapRect(boundingBox);

                }

                final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                        "0", label, confidence, boundingBox);

                result.setColor(color);
                result.setLocation(boundingBox);
                result.setExtra(extra);
                result.setCrop(crop);
                mappedRecognitions.add(result);

            }


        }

        //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

        updateResults(currTimestamp, mappedRecognitions);
    }

    private List<SimilarityClassifier.Recognition> loopThroughFaces(long currTimestamp, List<Face> faces, Uri selectedImage){
        final List<SimilarityClassifier.Recognition> mappedRecognitions = new LinkedList<SimilarityClassifier.Recognition>();
        ImageDecoder.Source source = ImageDecoder.createSource(this.getContentResolver(), selectedImage);
        Bitmap imgBitmap = null;

        try {
            imgBitmap = ImageDecoder.decodeBitmap(source);
            Bitmap mutableBitmap = imgBitmap.copy(Bitmap.Config.ARGB_8888, true);

            // Note this can be done only once
            int sourceW = rgbFrameBitmap.getWidth();
            int sourceH = rgbFrameBitmap.getHeight();
            int targetW = portraitBmp.getWidth();
            int targetH = portraitBmp.getHeight();
            Matrix transform = createTransform(
                    sourceW,
                    sourceH,
                    targetW,
                    targetH,
                    sensorOrientation);
            final Canvas cv = new Canvas(portraitBmp);

            // draws the original image in portrait mode.
            cv.drawBitmap(rgbFrameBitmap, transform, null);

            final Canvas cvFace = new Canvas(mutableBitmap);

            boolean saved = false;

            for (Face face : faces) {

                LOGGER.i("FACE" + face.toString());
                LOGGER.i("Running detection on face " + currTimestamp);
                //results = detector.recognizeImage(croppedBitmap);

                final RectF boundingBox = new RectF(face.getBoundingBox());

                //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
                final boolean goodConfidence = true; //face.get;
                if (boundingBox != null && goodConfidence) {

                    // maps crop coordinates to original
                    cropToFrameTransform.mapRect(boundingBox);

                    // maps original coordinates to portrait coordinates
                    RectF faceBB = new RectF(boundingBox);
                    transform.mapRect(faceBB);

                    // translates portrait to origin and scales to fit input inference size
                    //cv.drawRect(faceBB, paint);
                    float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
                    float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
                    Matrix matrix = new Matrix();
                    matrix.postTranslate(-faceBB.left, -faceBB.top);
                    matrix.postScale(sx, sy);

                    cvFace.drawBitmap(portraitBmp, matrix, null);

                    String label = "";
                    float confidence = -1f;
                    int color = Color.BLUE;
                    Object extra = null;
                    Bitmap crop = Bitmap.createBitmap(mutableBitmap);

                    final long startTime = SystemClock.uptimeMillis();
                    final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, true);
                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                    if (resultsAux.size() > 0) {

                        SimilarityClassifier.Recognition result = resultsAux.get(0);

                        extra = result.getExtra();
                        if (extra != null) {
                            Log.e("embeeding",gson.toJson(extra));
                        }

                        float conf = result.getDistance();
                        if (conf < 1.0f) {

                            confidence = conf;
                            label = result.getTitle();
                            if (result.getId().equals("0")) {
                                color = Color.GREEN;
                            }
                            else {
                                color = Color.RED;
                            }
                        }

                    }

                    if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

                        // camera is frontal so the image is flipped horizontally
                        // flips horizontally
                        Matrix flip = new Matrix();
                        if (sensorOrientation == 90 || sensorOrientation == 270) {
                            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
                        }
                        else {
                            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
                        }
                        //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
                        flip.mapRect(boundingBox);

                    }

                    final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                            "0", label, confidence, boundingBox);

                    result.setColor(color);
                    result.setLocation(boundingBox);
                    result.setExtra(extra);
                    result.setCrop(crop);
                    mappedRecognitions.add(result);
                }
            }
        } catch (IOException exception) {
            exception.printStackTrace();
        }

        return mappedRecognitions;
    }
}
